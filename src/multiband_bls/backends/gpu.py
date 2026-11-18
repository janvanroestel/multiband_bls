"""GPU backends (CuPy): the CUDA kernels in :mod:`multiband_bls.gpu` behind the engine contract.

``prepare`` validates, weights and uploads the data once; ``run`` launches the kernel for one
segment. Kernels only produce a periodogram (and, for the multiband ones, per-frequency
band counts), so the best fit comes from a single exact CPU refit at the global peak
(``refine``) on the host copy kept in the prepared state.
"""

from __future__ import annotations

__all__ = ["IMPLEMENTATIONS"]

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from .. import _eebls, _meebls  # type: ignore[attr-defined]
from .. import gpu as _k
from .._common import (
    _check_phase_range,
    _coarse_active,
    _coarse_to_fine,
    _flux_scale,
    _grid_params,
    _log_widths,
    _merge_bands,
    _preprocess_single,
    _scale_bands,
)
from .._prep import _prepare_bands
from .._validate import _validate_light_curve
from ..engine import (
    BestFit,
    CapabilityError,
    Implementation,
    Prepared,
    RunOptions,
    RunOutput,
    SegmentSpec,
)

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]


class _GpuPrep(Prepared):
    """Device arrays plus the host copy used for the exact best-fit refit."""

    def __init__(self, denom: float, labels: tuple[str, ...] | None = None,
                 chi2_red: Array | None = None) -> None:
        super().__init__(denom, labels, chi2_red)
        self.dev: dict[str, Any] = {}
        self.host: dict[str, Any] = {}


def _shift_epoch(t0: float, dur: float, freq: float, offset: float) -> float:
    return float((t0 + offset) % (1.0 / freq)) if dur > 0.0 else float(t0)


class _GpuBase(Implementation):
    backend = "gpu"
    fname: str

    def check_frequencies(self, prep: Prepared, freqs: Array, opts: RunOptions) -> None:
        p: Any = prep
        _check_phase_range(p.host["t_raw"], freqs)
        _k._check_int32_sizes(p.host["t_raw"].shape[0], freqs.size, self.fname)


# --------------------------------------------------------------------------- #
# Exact (binned) kernels                                                       #
# --------------------------------------------------------------------------- #
class BinnedGpuSingle(_GpuBase):
    method, multiband, exact = "binned", False, False
    fname = "eebls_gpu"

    def prepare(self, data: Any, rescale_errors: bool = False) -> _GpuPrep:
        import cupy as cp

        t, y, dy = _validate_light_curve(*data)
        _k._kernels()
        wx, w_hat, yy = _preprocess_single(y, dy)
        prep = _GpuPrep(yy)
        prep.host.update(t_raw=t, wx=wx, w_hat=w_hat)
        prep.dev.update(t=cp.asarray(t, dtype=cp.float64), wx=cp.asarray(wx, dtype=cp.float64),
                        w=cp.asarray(w_hat, dtype=cp.float64))
        return prep

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        import cupy as cp

        nbins = spec.nbins
        assert nbins is not None
        d = prep.dev
        f_d = cp.asarray(freqs, dtype=cp.float64)
        nf = int(f_d.size)
        kmi, kma = _grid_params(nbins, spec.q_min, spec.q_max)
        shared = 3 * nbins * 8  # ybin/rbin/cbin (float64)
        smem_cap = _k._single_kernel.max_dynamic_shared_size_bytes
        if shared > smem_cap:
            raise CapabilityError(
                f"eebls_gpu needs {shared} B of dynamic shared memory "
                f"(3 * nbins={nbins} * 8) but the device allows {smem_cap} B; "
                f"reduce nbins."
            )
        power_d = cp.empty(nf, dtype=cp.float64)
        _k._single_kernel(
            (min(nf, 2048),), (_k._THREADS,),
            (d["t"], d["wx"], d["w"], np.int32(d["t"].size), np.int32(nf), f_d,
             np.int32(nbins), np.int32(kmi), np.int32(kma),
             np.int32(min(opts.min_points, d["t"].size + 1)), power_d,
             np.float64(opts.sign)),
            shared_mem=shared)
        return RunOutput(cp.asnumpy(power_d))

    def refine(self, prep: Any, freq: float, spec: SegmentSpec, opts: RunOptions) -> BestFit:
        h = prep.host
        _, _, t0, dur, depth, sr = _eebls.eebls_grid(
            np.ascontiguousarray(h["t_raw"]), h["wx"], h["w_hat"], np.array([freq]),
            spec.nbins, float(spec.q_min), float(spec.q_max), opts.min_points, opts.sign)
        return BestFit(freq, float(t0), float(dur), float(depth), float(sr))


class BinnedGpuMulti(_GpuBase):
    method, multiband, exact = "binned", True, False
    fname = "multiband_eebls_gpu"

    def prepare(self, data: Any, rescale_errors: bool = False) -> _GpuPrep:
        import cupy as cp

        bands, chi2_red = _prepare_bands(data, rescale_errors)
        _k._kernels()
        t_all, wx, w, band, labels, w_totals, chi2_flat = _merge_bands(bands)
        prep = _GpuPrep(chi2_flat, labels, chi2_red)
        prep.host.update(t_raw=t_all, wx=wx, w=w, band=band, band_w=w_totals,
                         n_bands=len(labels))
        prep.n_bands = len(labels)  # type: ignore[attr-defined]
        if prep.n_bands > _k._MAX_BANDS:  # type: ignore[attr-defined]
            raise CapabilityError(f"GPU multiband kernel supports up to {_k._MAX_BANDS} bands")
        prep.dev.update(
            t=cp.asarray(t_all, dtype=cp.float64), wx=cp.asarray(wx, dtype=cp.float64),
            w=cp.asarray(w, dtype=cp.float64), band=cp.asarray(band, dtype=cp.int32),
            bw=cp.asarray(w_totals, dtype=cp.float64),
            bn=cp.asarray(np.bincount(band, minlength=len(labels)), dtype=cp.int32))
        return prep

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        import cupy as cp

        nbins = spec.nbins
        assert nbins is not None
        d, n_bands = prep.dev, prep.n_bands
        f_d = cp.asarray(freqs, dtype=cp.float64)
        nf = int(f_d.size)
        power_d = cp.empty(nf, dtype=cp.float64)
        k_d = cp.zeros(nf, dtype=cp.int32)
        kmi, kma = _grid_params(nbins, spec.q_min, spec.q_max)
        multi_kernel = _k._multi_kernel_for(n_bands)
        shared = 3 * n_bands * nbins * 8
        smem_cap = multi_kernel.max_dynamic_shared_size_bytes
        if shared > smem_cap:
            raise CapabilityError(
                f"multiband_eebls_gpu needs {shared} B of dynamic shared memory "
                f"(3 * n_bands={n_bands} * nbins={nbins} * 8) but the device allows "
                f"{smem_cap} B; reduce nbins or n_bands."
            )
        multi_kernel(
            (min(nf, 2048),), (_k._THREADS,),
            (d["t"], d["wx"], d["w"], d["band"], d["bw"], d["bn"], np.int32(d["t"].size),
             np.int32(n_bands), np.int32(nf), f_d, np.int32(nbins), np.int32(kmi),
             np.int32(kma), np.int32(min(opts.min_points, d["t"].size + 1)), power_d, k_d,
             np.float64(opts.sign)),
            shared_mem=shared)
        return RunOutput(cp.asnumpy(power_d), None, cp.asnumpy(k_d).astype(np.int64))

    def refine(self, prep: Any, freq: float, spec: SegmentSpec, opts: RunOptions) -> BestFit:
        h = prep.host
        _, _, t0, dur, depths, sr, _ = _meebls.meebls_grid(
            np.ascontiguousarray(h["t_raw"]), h["wx"], h["w"], h["band"], prep.n_bands,
            np.array([freq]), spec.nbins, float(spec.q_min), float(spec.q_max),
            opts.min_points, np.ascontiguousarray(h["band_w"]), opts.sign)
        return BestFit(freq, float(t0), float(dur), np.asarray(depths), float(sr))


# --------------------------------------------------------------------------- #
# Fast (prefix-sum, log-width, float32) kernels                                #
# --------------------------------------------------------------------------- #
class FastGpuSingle(_GpuBase):
    method, multiband, exact = "fast", False, False
    fname = "eebls_gpu_fast"

    def prepare(self, data: Any, rescale_errors: bool = False) -> _GpuPrep:
        import cupy as cp

        t, y, dy = _validate_light_curve(*data)
        if t.shape[0] >= _k._MAX_FAST_POINTS:
            raise CapabilityError(
                f"eebls_gpu_fast supports fewer than {_k._MAX_FAST_POINTS} points "
                f"(float32 point counts), got {t.shape[0]}; use eebls_gpu."
            )
        _k._kernels()
        # Work in units of the median uncertainty so sy**2 and the band weights stay
        # inside float32 range for any flux scale; the power is scale-invariant.
        f_scale = _flux_scale([dy])
        wx, w_hat, yy = _preprocess_single(np.asarray(y, dtype=np.float64) / f_scale,
                                           np.asarray(dy, dtype=np.float64) / f_scale)
        # Subtract the median so large JD values keep full precision in the two-float split.
        t_offset = float(np.median(t))
        t_hi, t_lo = _k._split_time(np.asarray(t, dtype=np.float64) - t_offset)
        prep = _GpuPrep(yy)
        wx_h, w_h, _ = _preprocess_single(y, dy)  # unscaled, for the exact refit
        prep.host.update(t_raw=t, t_offset=t_offset, t_c=np.asarray(t, dtype=np.float64)
                         - t_offset, wx=wx_h, w_hat=w_h)
        prep.dev.update(thi=cp.asarray(t_hi, dtype=cp.float32),
                        tlo=cp.asarray(t_lo, dtype=cp.float32),
                        wx=cp.asarray(wx, dtype=cp.float32),
                        w=cp.asarray(w_hat, dtype=cp.float32))
        return prep

    def check_frequencies(self, prep: Prepared, freqs: Array, opts: RunOptions) -> None:
        super().check_frequencies(prep, freqs, opts)
        p: Any = prep
        _k._check_fast_float32(p.host["t_c"], freqs, self.fname)

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        import cupy as cp

        nbins = spec.nbins
        assert nbins is not None
        d = prep.dev
        n = int(d["thi"].size)
        kmi, kma = _grid_params(nbins, spec.q_min, spec.q_max)
        widths_np = _log_widths(kmi, kma, opts.dlogq)
        widths_d = cp.asarray(widths_np, dtype=cp.int32)
        n_widths = int(widths_np.size)
        logger.debug("eebls_gpu_fast: nb=%d kmi=%d kma=%d n_widths=%d (linear=%d)",
                     nbins, kmi, kma, n_widths, kma - kmi + 1)
        shared = (3 * nbins + 8) * 4  # float32: ybin + rbin + cbin [nb] + warp_max[8]
        smem_cap = _k._fast_single_kernel.max_dynamic_shared_size_bytes
        if shared > smem_cap:
            raise CapabilityError(
                f"eebls_gpu_fast needs {shared} B of dynamic shared memory "
                f"((3 * nbins={nbins} + 8) * 4) but the device allows "
                f"{smem_cap} B; reduce nbins."
            )

        def evaluate(f_np: Array) -> Array:
            f_d = cp.asarray(f_np, dtype=cp.float64)
            nf = int(f_d.size)
            power_d = cp.empty(nf, dtype=cp.float64)
            _k._fast_single_kernel(
                (min(nf, 2048),), (_k._THREADS,),
                (d["thi"], d["tlo"], d["wx"], d["w"], np.int32(n), np.int32(nf), f_d,
                 np.int32(nbins), widths_d, np.int32(n_widths),
                 np.int32(min(opts.min_points, n + 1)), power_d, np.float32(opts.sign)),
                shared_mem=shared)
            return cp.asnumpy(power_d)

        return RunOutput(_coarse_to_fine(evaluate, freqs, opts.coarse_factor, opts.n_refine))

    def refine(self, prep: Any, freq: float, spec: SegmentSpec, opts: RunOptions) -> BestFit:
        h = prep.host
        _, _, t0, dur, depth, sr = _eebls.eebls_grid(
            np.ascontiguousarray(h["t_c"]), h["wx"], h["w_hat"], np.array([freq]),
            spec.nbins, float(spec.q_min), float(spec.q_max), opts.min_points, opts.sign)
        return BestFit(freq, _shift_epoch(t0, dur, freq, h["t_offset"]), float(dur),
                       float(depth), float(sr))


class FastGpuMulti(_GpuBase):
    method, multiband, exact = "fast", True, False
    fname = "multiband_eebls_gpu_fast"

    def prepare(self, data: Any, rescale_errors: bool = False) -> _GpuPrep:
        import cupy as cp

        bands, chi2_red = _prepare_bands(data, rescale_errors)
        _k._kernels()
        # Median-uncertainty units keep sy**2 and the band weights inside float32 range
        # for any flux scale (the power is scale-invariant); the refit uses unscaled bands.
        f_scale = _flux_scale([b[2] for b in bands.values()])
        t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(
            _scale_bands(bands, f_scale))
        n_bands = len(labels)
        if n_bands > _k._MAX_BANDS:
            raise CapabilityError(f"GPU multiband kernel supports up to {_k._MAX_BANDS} bands")
        if t_all.shape[0] >= _k._MAX_FAST_POINTS:
            raise CapabilityError(
                f"multiband_eebls_gpu_fast supports fewer than {_k._MAX_FAST_POINTS} points "
                f"(float32 point counts), got {t_all.shape[0]}; use multiband_eebls_gpu."
            )
        t_offset = float(np.median(t_all))
        t_hi, t_lo = _k._split_time(t_all - t_offset)
        prep = _GpuPrep(chi2_flat, labels, chi2_red)
        prep.n_bands = n_bands  # type: ignore[attr-defined]
        _, wx_h, w_h, band_h, _, w_tot_h, _ = _merge_bands(bands)
        prep.host.update(t_raw=t_all, t_offset=t_offset, t_c=t_all - t_offset, wx=wx_h, w=w_h,
                         band=band_h, band_w=w_tot_h)
        prep.dev.update(thi=cp.asarray(t_hi, dtype=cp.float32),
                        tlo=cp.asarray(t_lo, dtype=cp.float32),
                        wx=cp.asarray(wx_all, dtype=cp.float32),
                        w=cp.asarray(w_all, dtype=cp.float32),
                        band=cp.asarray(band_all, dtype=cp.int32),
                        bw=cp.asarray(w_totals, dtype=cp.float64))
        return prep

    def check_frequencies(self, prep: Prepared, freqs: Array, opts: RunOptions) -> None:
        super().check_frequencies(prep, freqs, opts)
        p: Any = prep
        _k._check_fast_float32(p.host["t_c"], freqs, self.fname)

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        import cupy as cp

        nbins = spec.nbins
        assert nbins is not None
        d, n_bands = prep.dev, prep.n_bands
        # n_bands_used is undefined at interpolated frequencies; it is still available when
        # the grid is too short for the coarse search to skip any frequency.
        with_k = opts.compute_n_bands_used and not _coarse_active(freqs.size,
                                                                  opts.coarse_factor)
        n = int(d["thi"].size)
        kmi, kma = _grid_params(nbins, spec.q_min, spec.q_max)
        widths_np = _log_widths(kmi, kma, opts.dlogq)
        widths_d = cp.asarray(widths_np, dtype=cp.int32)
        n_widths = int(widths_np.size)
        logger.debug("multiband_eebls_gpu_fast: nb=%d kmi=%d kma=%d n_widths=%d (linear=%d)",
                     nbins, kmi, kma, n_widths, kma - kmi + 1)
        # float32: ybin + rbin + cbin (per band) + warp_max[8] (+ int warp_k[8])
        n_extra = 16 if with_k else 8
        shared = (3 * n_bands * nbins + n_extra) * 4
        kernel = _k._fast_multi_kernel_for(n_bands, with_k, shared)
        smem_cap = kernel.max_dynamic_shared_size_bytes
        if shared > smem_cap:
            raise CapabilityError(
                f"multiband_eebls_gpu_fast needs {shared} B of dynamic shared memory "
                f"((3 * n_bands={n_bands} * nbins={nbins} + {n_extra}) * 4) but the "
                f"device allows {smem_cap} B; reduce nbins or n_bands."
            )
        k_out: list[npt.NDArray[np.int32]] = []

        def evaluate(f_np: Array) -> Array:
            f_d = cp.asarray(f_np, dtype=cp.float64)
            nf = int(f_d.size)
            power_d = cp.empty(nf, dtype=cp.float64)
            # k output; a 1-element placeholder when the kernel is compiled without WITH_K
            k_d = cp.zeros(nf if with_k else 1, dtype=cp.int32)
            kernel(
                (min(nf, 2048),), (_k._THREADS,),
                (d["thi"], d["tlo"], d["wx"], d["w"], d["band"], d["bw"], np.int32(n),
                 np.int32(n_bands), np.int32(nf), f_d, np.int32(nbins), widths_d,
                 np.int32(n_widths), np.int32(min(opts.min_points, n + 1)), power_d, k_d,
                 np.float32(opts.sign)),
                shared_mem=shared)
            if with_k:
                k_out.append(cp.asnumpy(k_d))
            return cp.asnumpy(power_d)

        power = _coarse_to_fine(evaluate, freqs, opts.coarse_factor, opts.n_refine)
        # (a full evaluation, i.e. no coarse search, is one call over the whole grid)
        return RunOutput(power, None, k_out[0].astype(np.int64) if with_k else None)

    def refine(self, prep: Any, freq: float, spec: SegmentSpec, opts: RunOptions) -> BestFit:
        h = prep.host
        _, _, t0, dur, depths, sr, _ = _meebls.meebls_grid(
            np.ascontiguousarray(h["t_c"]), h["wx"], h["w"], h["band"], prep.n_bands,
            np.array([freq]), spec.nbins, float(spec.q_min), float(spec.q_max),
            opts.min_points, np.ascontiguousarray(h["band_w"]), opts.sign)
        return BestFit(freq, _shift_epoch(t0, dur, freq, h["t_offset"]), float(dur),
                       np.asarray(depths), float(sr))


IMPLEMENTATIONS: list[type[Implementation]] = [
    BinnedGpuSingle, BinnedGpuMulti, FastGpuSingle, FastGpuMulti,
]
