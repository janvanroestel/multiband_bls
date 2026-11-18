"""CPU backends: the Cython cores behind the engine contract."""

from __future__ import annotations

__all__ = ["IMPLEMENTATIONS"]

from typing import Any

import numpy as np
import numpy.typing as npt

from .. import _eebls, _fast, _meebls, _msbls, _sbls  # type: ignore[attr-defined]
from .._common import (
    _check_phase_range,
    _coarse_eval,
    _grid_params,
    _log_widths,
    _merge_bands,
    _preprocess_single,
    _threaded_grid,
)
from .._prep import _prepare_bands
from .._validate import _validate_light_curve
from ..engine import (
    BestFit,
    Implementation,
    Prepared,
    RunOptions,
    RunOutput,
    SegmentSpec,
)

Array = npt.NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Prepared data                                                                #
# --------------------------------------------------------------------------- #
class SinglePrep(Prepared):
    """Validated single-band arrays; ``t`` is median-centred by ``t_offset``."""

    def __init__(self, t: Array, wx: Array, w_hat: Array, yy: float, t_offset: float,
                 t_raw: Array) -> None:
        super().__init__(denom=yy)
        self.t, self.wx, self.w_hat, self.t_offset = t, wx, w_hat, t_offset
        self.t_raw = t_raw  # uncentred times: the phase-precision check is on these


class MultiPrep(Prepared):
    """Merged multiband arrays (``t`` centred by ``t_offset``) and band weights."""

    def __init__(self, t: Array, wx: Array, w: Array, band: Array, labels: tuple[str, ...],
                 band_w: Array, chi2_flat: float, chi2_red: Array, t_offset: float,
                 t_raw: Array) -> None:
        super().__init__(denom=chi2_flat, labels=labels, band_chi2_red=chi2_red)
        self.t_raw = t_raw
        self.t, self.wx, self.w, self.band = t, wx, w, band
        self.band_w, self.t_offset = band_w, t_offset
        self.n_bands = len(labels)


def prepare_single(data: tuple[Array, Array, Array], center: bool) -> SinglePrep:
    t, y, dy = _validate_light_curve(*data)
    wx, w_hat, yy = _preprocess_single(y, dy)
    off = float(np.median(t)) if center else 0.0
    tc = np.ascontiguousarray(t - off if center else t, dtype=np.float64)
    return SinglePrep(tc, wx, w_hat, yy, off, t)


def prepare_multi(data: Any, rescale_errors: bool, center: bool) -> MultiPrep:
    bands, chi2_red = _prepare_bands(data, rescale_errors)
    t_all, wx, w, band, labels, w_totals, chi2_flat = _merge_bands(bands)
    off = float(np.median(t_all)) if center else 0.0
    tc = np.ascontiguousarray(t_all - off if center else t_all, dtype=np.float64)
    return MultiPrep(tc, wx, w, band, labels, np.ascontiguousarray(w_totals), chi2_flat,
                     chi2_red, off, t_all)


# --------------------------------------------------------------------------- #
# Implementations                                                              #
# --------------------------------------------------------------------------- #
class _CpuBase(Implementation):
    backend = "cpu"

    def check_frequencies(self, prep: Prepared, freqs: Array, opts: RunOptions) -> None:
        _check_phase_range(prep.t_raw, freqs)  # type: ignore[attr-defined]


class SparseSingle(_CpuBase):
    method, multiband, exact = "sparse", False, True

    def prepare(self, data: Any, rescale_errors: bool = False) -> SinglePrep:
        return prepare_single(data, center=False)

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        power, bf, t0, dur, depth, sr = _sbls.sbls_grid(
            prep.t, prep.wx, prep.w_hat, freqs, float(spec.q_min), float(spec.q_max),
            opts.min_points, opts.sign)
        return RunOutput(power, BestFit(float(bf), float(t0), float(dur), float(depth),
                                        float(sr)))


class BinnedSingle(_CpuBase):
    method, multiband, exact = "binned", False, True

    def prepare(self, data: Any, rescale_errors: bool = False) -> SinglePrep:
        return prepare_single(data, center=False)

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        power, bf, t0, dur, depth, sr = _eebls.eebls_grid(
            prep.t, prep.wx, prep.w_hat, freqs, spec.nbins, float(spec.q_min),
            float(spec.q_max), opts.min_points, opts.sign)
        return RunOutput(power, BestFit(float(bf), float(t0), float(dur), float(depth),
                                        float(sr)))


class FastSingle(_CpuBase):
    method, multiband, exact = "fast", False, False

    def prepare(self, data: Any, rescale_errors: bool = False) -> SinglePrep:
        return prepare_single(data, center=True)

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        assert spec.nbins is not None
        kmi, kma = _grid_params(spec.nbins, spec.q_min, spec.q_max)
        widths = np.ascontiguousarray(_log_widths(kmi, kma, opts.dlogq), dtype=np.int32)
        mp = min(opts.min_points, prep.t.shape[0] + 1)

        def kernel(f: Array) -> Array:
            return _fast.eebls_fast_grid(prep.t, prep.wx, prep.w_hat, np.ascontiguousarray(f),
                                         spec.nbins, widths, mp, opts.sign)

        power, evaluated = _coarse_eval(lambda f: _threaded_grid(kernel, f, opts.n_threads),
                                        freqs, opts.coarse_factor, opts.n_refine)
        return RunOutput(power, evaluated=evaluated)

    def refine(self, prep: Any, freq: float, spec: SegmentSpec, opts: RunOptions) -> BestFit:
        _, _, t0, dur, depth, sr = _eebls.eebls_grid(
            prep.t, prep.wx, prep.w_hat, np.array([freq]), spec.nbins, float(spec.q_min),
            float(spec.q_max), opts.min_points, opts.sign)
        if dur > 0.0:  # report the epoch on the caller's (uncentred) time axis
            t0 = float((t0 + prep.t_offset) % (1.0 / freq))
        return BestFit(freq, float(t0), float(dur), float(depth), float(sr))


class SparseMulti(_CpuBase):
    method, multiband, exact = "sparse", True, True

    def prepare(self, data: Any, rescale_errors: bool = False) -> MultiPrep:
        return prepare_multi(data, rescale_errors, center=False)

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        power, bf, t0, dur, depths, sr, k = _msbls.msbls_grid(
            prep.t, prep.wx, prep.w, prep.band, prep.n_bands, freqs, float(spec.q_min),
            float(spec.q_max), opts.min_points, prep.band_w, opts.sign)
        return RunOutput(power, BestFit(float(bf), float(t0), float(dur),
                                        np.asarray(depths), float(sr)), np.asarray(k))


class BinnedMulti(_CpuBase):
    method, multiband, exact = "binned", True, True

    def prepare(self, data: Any, rescale_errors: bool = False) -> MultiPrep:
        return prepare_multi(data, rescale_errors, center=False)

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        power, bf, t0, dur, depths, sr, k = _meebls.meebls_grid(
            prep.t, prep.wx, prep.w, prep.band, prep.n_bands, freqs, spec.nbins,
            float(spec.q_min), float(spec.q_max), opts.min_points, prep.band_w, opts.sign)
        return RunOutput(power, BestFit(float(bf), float(t0), float(dur),
                                        np.asarray(depths), float(sr)), np.asarray(k))


class FastMulti(_CpuBase):
    method, multiband, exact = "fast", True, False

    def prepare(self, data: Any, rescale_errors: bool = False) -> MultiPrep:
        prep = prepare_multi(data, rescale_errors, center=True)
        prep.band32 = np.ascontiguousarray(prep.band, dtype=np.int32)  # type: ignore[attr-defined]
        prep.band_n = np.bincount(prep.band32, minlength=prep.n_bands).astype(np.float64)  # type: ignore[attr-defined]
        return prep

    def run(self, prep: Any, freqs: Array, spec: SegmentSpec, opts: RunOptions) -> RunOutput:
        assert spec.nbins is not None
        kmi, kma = _grid_params(spec.nbins, spec.q_min, spec.q_max)
        widths = np.ascontiguousarray(_log_widths(kmi, kma, opts.dlogq), dtype=np.int32)
        mp = min(opts.min_points, prep.t.shape[0] + 1)

        def kernel(f: Array) -> Array:
            return _fast.meebls_fast_grid(prep.t, prep.wx, prep.w, prep.band32, prep.n_bands,
                                          np.ascontiguousarray(f), spec.nbins, widths, mp,
                                          prep.band_w, prep.band_n, opts.sign)

        power, evaluated = _coarse_eval(lambda f: _threaded_grid(kernel, f, opts.n_threads),
                                        freqs, opts.coarse_factor, opts.n_refine)
        return RunOutput(power, evaluated=evaluated)

    def refine(self, prep: Any, freq: float, spec: SegmentSpec, opts: RunOptions) -> BestFit:
        _, _, t0, dur, depths, sr, _ = _meebls.meebls_grid(
            prep.t, prep.wx, prep.w, prep.band, prep.n_bands, np.array([freq]), spec.nbins,
            float(spec.q_min), float(spec.q_max), opts.min_points, prep.band_w, opts.sign)
        if dur > 0.0:
            t0 = float((t0 + prep.t_offset) % (1.0 / freq))
        return BestFit(freq, float(t0), float(dur), np.asarray(depths), float(sr))


IMPLEMENTATIONS: list[type[Implementation]] = [
    SparseSingle, BinnedSingle, FastSingle, SparseMulti, BinnedMulti, FastMulti,
]
