"""GPU (CuPy ``RawKernel``) backends for the binned BLS methods.

One CUDA thread-block per trial frequency: the block bins the (small, shared)
light curve for its own frequency into shared-memory bins, runs the box search,
and writes one periodogram value. This mirrors the compiled binned cores
(:mod:`multiband_bls._eebls`, :mod:`multiband_bls._meebls`) and returns the same
``power = Delta chi^2 / chi^2_flat``.

The periodogram (the expensive part) is computed on the GPU; the best period's
``t0``/duration/depth are recomputed once on the CPU. Fallback is the caller's
responsibility -- :func:`gpu_available` reports whether a usable device exists.

CuPy is imported lazily so the package still imports without it / without a GPU.
"""

from __future__ import annotations

__all__ = [
    "gpu_available",
    "eebls_gpu",
    "multiband_eebls_gpu",
    "eebls_gpu_fast",
    "multiband_eebls_gpu_fast",
]

import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .api import _merge_bands, _preprocess_single, eebls, multiband_eebls
from .periodogram import BLSResult
from .reference import auto_nbins

logger = logging.getLogger(__name__)

Array = np.ndarray

_THREADS = 256  # threads per block (also the reduction array size in the kernels)
# Stack arrays in the CUDA multiband kernels are fixed-size; recompile after
# changing _MAX_BANDS (update the s[N]/r[N]/cnt[N] declarations in the kernel source).
_MAX_BANDS: int = 8

# --------------------------------------------------------------------------- #
# CUDA kernels                                                                 #
# --------------------------------------------------------------------------- #
_SINGLE_SRC = r"""
extern "C" __global__
void eebls_kernel(const double* t, const double* wx, const double* w, int n,
                  const double* freqs, int nb, int kmi, int kma,
                  int min_bin_points, double* power) {
    int f = blockIdx.x;
    double freq = freqs[f];
    extern __shared__ double sh[];           // ybin[nb], rbin[nb], cbin[nb]
    double* ybin = sh;
    double* rbin = sh + nb;
    double* cbin = sh + 2 * nb;
    int tid = threadIdx.x;

    for (int b = tid; b < nb; b += blockDim.x) { ybin[b] = 0; rbin[b] = 0; cbin[b] = 0; }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        double ph = t[i] * freq; ph -= floor(ph);
        int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
        atomicAdd(&ybin[ib], wx[i]);
        atomicAdd(&rbin[ib], w[i]);
        atomicAdd(&cbin[ib], 1.0);
    }
    __syncthreads();

    double best = 0.0;
    for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
        double s = 0.0, r = 0.0; int cnt = 0;
        for (int ww = 1; ww <= kma; ++ww) {
            int jj = i1 + ww - 1; if (jj >= nb) jj -= nb;
            s += ybin[jj]; r += rbin[jj]; cnt += (int)cbin[jj];
            if (ww < kmi) continue;
            if (cnt < min_bin_points) continue;
            double denom = r * (1.0 - r);
            if (denom <= 0.0) continue;
            double sr2 = s * s / denom;
            if (sr2 > best) best = sr2;
        }
    }
    __shared__ double red[256];
    red[tid] = best; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { if (red[tid + stride] > red[tid]) red[tid] = red[tid + stride]; }
        __syncthreads();
    }
    if (tid == 0) power[f] = red[0];
}
"""

_MULTI_SRC = r"""
extern "C" __global__
void meebls_kernel(const double* t, const double* wx, const double* w,
                   const int* band, const double* band_w,
                   int n, int n_bands,
                   const double* freqs, int nb, int kmi, int kma,
                   int min_points, double* power) {
    int f = blockIdx.x;
    double freq = freqs[f];
    extern __shared__ double sh[];           // ybin/rbin/cbin each n_bands*nb
    double* ybin = sh;
    double* rbin = sh + n_bands * nb;
    double* cbin = sh + 2 * n_bands * nb;
    int tid = threadIdx.x;
    int tot = n_bands * nb;

    for (int b = tid; b < tot; b += blockDim.x) { ybin[b] = 0; rbin[b] = 0; cbin[b] = 0; }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        double ph = t[i] * freq; ph -= floor(ph);
        int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
        int bb = band[i];
        atomicAdd(&ybin[bb * nb + ib], wx[i]);
        atomicAdd(&rbin[bb * nb + ib], w[i]);
        atomicAdd(&cbin[bb * nb + ib], 1.0);
    }
    __syncthreads();

    double best = 0.0;
    double s[8], r[8]; int cnt[8];
    for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
        for (int b = 0; b < n_bands; b++) { s[b] = 0; r[b] = 0; cnt[b] = 0; }
        int total = 0;
        for (int ww = 1; ww <= kma; ++ww) {
            int jj = i1 + ww - 1; if (jj >= nb) jj -= nb;
            for (int b = 0; b < n_bands; b++) {
                s[b] += ybin[b * nb + jj]; r[b] += rbin[b * nb + jj];
                cnt[b] += (int)cbin[b * nb + jj]; total += (int)cbin[b * nb + jj];
            }
            if (ww < kmi) continue;
            if (total < min_points) continue;
            double T3 = 0.0;
            for (int b = 0; b < n_bands; b++) {
                double denom = r[b] * (1.0 - r[b]);
                if (denom <= 0.0) continue;
                T3 += band_w[b] * s[b] * s[b] / denom;
            }
            if (T3 > best) best = T3;
        }
    }
    __shared__ double red[256];
    red[tid] = best; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { if (red[tid + stride] > red[tid]) red[tid] = red[tid + stride]; }
        __syncthreads();
    }
    if (tid == 0) power[f] = red[0];
}
"""

_FAST_BOX_SINGLE_SRC = r"""
// Incremental-accumulation box BLS kernel with log-spaced transit widths.
// Float32 bins; SR^2 comparison; warp-shuffle reduction; persistent grid-stride;
// __launch_bounds__ + __ldg__.
// min_r replaces a count check: weights are normalised to sum 1, so
// sum_r < min_r (= min_bin_points/n) reliably filters under-populated windows.
// Shared memory (float32):
//   ybin[nb], rbin[nb]              2*nb
//   warp_max[8]                     8
// Total: (2*nb + 8) * 4 bytes  (~4.0 KB for nb=500)
extern "C" __global__ __launch_bounds__(256, 6)
void eebls_fast_kernel(
    const double* __restrict__ t,
    const double* __restrict__ wx,
    const double* __restrict__ w,
    int n, int nf,
    const double* __restrict__ freqs, int nb,
    const int* __restrict__ widths, int n_widths,
    float min_r, double* power)
{
    extern __shared__ float sh[];
    float* ybin     = sh;
    float* rbin     = sh + nb;
    float* warp_max = sh + 2*nb;  // 8 floats (one per warp, 256/32=8)

    int tid = threadIdx.x;

    for (int f = blockIdx.x; f < nf; f += gridDim.x) {
        double freq = __ldg(&freqs[f]);

        // Phase 1: clear bins
        for (int b = tid; b < nb; b += blockDim.x) {
            ybin[b] = 0.f; rbin[b] = 0.f;
        }
        __syncthreads();

        // Phase 2: bin observations (double-precision phase, float32 accumulate)
        for (int i = tid; i < n; i += blockDim.x) {
            double ph = __ldg(&t[i]) * freq; ph -= floor(ph);
            int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
            atomicAdd(&ybin[ib], (float)__ldg(&wx[i]));
            atomicAdd(&rbin[ib], (float)__ldg(&w[i]));
        }
        __syncthreads();

        // Phase 3: window sweep - incremental accumulation over log-spaced widths.
        // Widths are sorted ascending so each bin is read at most once per i1.
        float best2 = 0.f;
        for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
            float sy = 0.f, sr = 0.f;
            int prev_w = 0;
            for (int wi = 0; wi < n_widths; wi++) {
                int w = __ldg(&widths[wi]);
                for (int k = prev_w; k < w; k++) {
                    int jj = i1 + k; if (jj >= nb) jj -= nb;
                    sy += ybin[jj]; sr += rbin[jj];
                }
                prev_w = w;
                if (sr < min_r) continue;
                float denom = sr * (1.f - sr);
                if (denom <= 0.f) continue;
                float sr2 = sy * sy / denom;
                if (sr2 > best2) best2 = sr2;
            }
        }

        // Phase 4: warp-shuffle reduction, then cross-warp via warp_max[8]
        for (int offset = 16; offset > 0; offset >>= 1)
            best2 = fmaxf(best2, __shfl_down_sync(0xffffffff, best2, offset));
        if ((tid & 31) == 0) warp_max[tid >> 5] = best2;
        __syncthreads();
        if (tid == 0) {
            best2 = fmaxf(fmaxf(fmaxf(warp_max[0], warp_max[1]),
                                fmaxf(warp_max[2], warp_max[3])),
                          fmaxf(fmaxf(warp_max[4], warp_max[5]),
                                fmaxf(warp_max[6], warp_max[7])));
            power[f] = (double)best2;
        }
        __syncthreads();
    }
}
"""

_FAST_BOX_MULTI_SRC = r"""
// Incremental-accumulation multiband box BLS with log-spaced widths.
// Handles all band counts (1-8).
// Float32 bins; band_w in registers; SR^2 comparison; warp-shuffle reduction;
// persistent grid-stride; __launch_bounds__ + __ldg__.
// Shared memory: (3*n_bands*nb + 8) * 4 bytes  (~35.2 KB for n_bands=6, nb=500)
extern "C" __global__ __launch_bounds__(256, 4)
void meebls_fast_kernel(
    const double* __restrict__ t,
    const double* __restrict__ wx,
    const double* __restrict__ w,
    const int* __restrict__ band,
    const double* __restrict__ band_w,
    int n, int n_bands, int nf,
    const double* __restrict__ freqs, int nb,
    const int* __restrict__ widths, int n_widths,
    int min_points, double* power)
{
    extern __shared__ float sh[];
    float* ybin    = sh;
    float* rbin    = sh + n_bands * nb;
    float* cbin    = sh + 2 * n_bands * nb;
    float* warp_max = sh + 3 * n_bands * nb;  // 8 floats (one per warp, 256/32=8)

    int tid = threadIdx.x;
    int tot = n_bands * nb;

    // Load band weights into registers (avoids repeated global-mem reads in hot loop)
    float bw[8];
    for (int b = 0; b < n_bands; b++) bw[b] = (float)__ldg(&band_w[b]);

    for (int f = blockIdx.x; f < nf; f += gridDim.x) {
        double freq = __ldg(&freqs[f]);

        for (int b = tid; b < tot; b += blockDim.x) {
            ybin[b] = 0.f; rbin[b] = 0.f; cbin[b] = 0.f;
        }
        __syncthreads();

        for (int i = tid; i < n; i += blockDim.x) {
            double ph = __ldg(&t[i]) * freq; ph -= floor(ph);
            int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
            int bb = __ldg(&band[i]);
            atomicAdd(&ybin[bb * nb + ib], (float)__ldg(&wx[i]));
            atomicAdd(&rbin[bb * nb + ib], (float)__ldg(&w[i]));
            atomicAdd(&cbin[bb * nb + ib], 1.f);
        }
        __syncthreads();

        int wmax = __ldg(&widths[n_widths - 1]);

        float best2 = 0.f;
        float s[8], r[8]; int cnt[8];
        for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
            for (int b = 0; b < n_bands; b++) { s[b] = 0.f; r[b] = 0.f; cnt[b] = 0; }
            int total = 0, wi = 0;
            for (int ww = 1; ww <= wmax; ++ww) {
                int jj = i1 + ww - 1; if (jj >= nb) jj -= nb;
                for (int b = 0; b < n_bands; b++) {
                    s[b] += ybin[b * nb + jj];
                    r[b] += rbin[b * nb + jj];
                    int c = (int)cbin[b * nb + jj]; cnt[b] += c; total += c;
                }
                if (wi >= n_widths || ww != __ldg(&widths[wi])) continue;
                wi++;
                if (total < min_points) continue;
                float sr2 = 0.f; bool ok = false;
                for (int b = 0; b < n_bands; b++) {
                    if (cnt[b] < 1) continue;
                    float denom = r[b] * (1.f - r[b]);
                    if (denom <= 0.f) continue;
                    sr2 += bw[b] * s[b] * s[b] / denom;
                    ok = true;
                }
                if (!ok) continue;
                if (sr2 > best2) best2 = sr2;
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            best2 = fmaxf(best2, __shfl_down_sync(0xffffffff, best2, offset));
        if ((tid & 31) == 0) warp_max[tid >> 5] = best2;
        __syncthreads();
        if (tid == 0) {
            best2 = fmaxf(fmaxf(fmaxf(warp_max[0], warp_max[1]),
                                fmaxf(warp_max[2], warp_max[3])),
                          fmaxf(fmaxf(warp_max[4], warp_max[5]),
                                fmaxf(warp_max[6], warp_max[7])));
            power[f] = (double)best2;
        }
        __syncthreads();
    }
}
"""

_single_kernel: Any = None
_multi_kernel: Any = None
_fast_single_kernel: Any = None
_fast_multi_kernel: Any = None
_max_smem: int = 65536  # updated at first _kernels() call from device attributes


def gpu_available() -> bool:
    """True if CuPy is importable and a CUDA device is usable."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _kernels() -> None:
    global _single_kernel, _multi_kernel
    global _fast_single_kernel, _fast_multi_kernel
    global _max_smem
    if _single_kernel is None:
        import cupy as cp

        _max_smem = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
        _single_kernel = cp.RawKernel(_SINGLE_SRC, "eebls_kernel")
        _multi_kernel = cp.RawKernel(_MULTI_SRC, "meebls_kernel")
        _fast_single_kernel = cp.RawKernel(_FAST_BOX_SINGLE_SRC, "eebls_fast_kernel")
        _fast_multi_kernel = cp.RawKernel(_FAST_BOX_MULTI_SRC, "meebls_fast_kernel")
        _multi_kernel.max_dynamic_shared_size_bytes = 65536
        _fast_multi_kernel.max_dynamic_shared_size_bytes = _max_smem
    # fast kernels are accessed as module globals by the fast entry points


def _log_widths(kmi: int, kma: int, dlogq: float = 0.3) -> np.ndarray:
    """Geometrically-spaced integer window widths between kmi and kma (inclusive).

    Reduces window evaluations from O(kma - kmi) to O(log(kma/kmi) / log(1+dlogq)).
    Smaller dlogq = more widths = higher fidelity but slower.
    """
    from math import ceil

    widths: set[int] = set()
    w = kmi
    while w <= kma:
        widths.add(w)
        w = max(w + 1, int(ceil(w * (1.0 + dlogq))))
    widths.add(kma)
    return np.array(sorted(widths), dtype=np.int32)


def _grid_params(nbins: int, qmin: float, qmax: float) -> tuple[int, int]:
    kmi = max(1, int(qmin * nbins))
    kma = min(nbins, int(qmax * nbins) + 1)
    return kmi, kma


def _result(
    power_raw: np.ndarray,
    chi2_flat: float,
    frequencies: np.ndarray,
    refine_fn: Callable[[float], BLSResult],
    bands: tuple[str, ...] | None = None,
) -> BLSResult:
    power = power_raw / chi2_flat
    best_idx = int(np.argmax(power))
    best_freq = float(frequencies[best_idx])
    ref = refine_fn(best_freq)
    return BLSResult(
        frequency=frequencies,
        power=power,
        best_frequency=best_freq,
        best_period=1.0 / best_freq,
        best_t0=ref.best_t0,
        best_duration=ref.best_duration,
        best_depth=ref.best_depth,
        best_power=float(power[best_idx]),
        bands=bands,
    )


def eebls_gpu(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    qmin: float = 0.01,
    qmax: float = 0.10,
    min_points: int = 3,
) -> BLSResult:
    """GPU binned BLS (single band). Mirrors :func:`multiband_bls.eebls`.

    Parameters
    ----------
    t :
        Observation times (days), 1-D float array.
    y :
        Magnitudes or fluxes, 1-D float array, same length as ``t``.
    dy :
        1-sigma uncertainties, 1-D float array, same length as ``t``.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / qmin, 50, 500)``
        if ``None``.
    qmin :
        Minimum transit duration as a fraction of the period.
    qmax :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required.
    """
    import cupy as cp

    if nbins is None:
        nbins = auto_nbins(qmin)
    _kernels()
    wx, w_hat, yy = _preprocess_single(y, dy)

    t_d = cp.asarray(t, dtype=cp.float64)
    wx_d = cp.asarray(wx, dtype=cp.float64)
    w_d = cp.asarray(w_hat, dtype=cp.float64)
    f_d = cp.asarray(frequencies, dtype=cp.float64)
    nf = int(f_d.size)
    kmi, kma = _grid_params(nbins, qmin, qmax)
    shared = 3 * nbins * 8  # ybin/rbin/cbin (float64)

    power_d = cp.empty(nf, dtype=cp.float64)
    _single_kernel((nf,), (_THREADS,),
                   (t_d, wx_d, w_d, np.int32(t_d.size), f_d, np.int32(nbins),
                    np.int32(kmi), np.int32(kma), np.int32(min_points), power_d),
                   shared_mem=shared)
    power_np = cp.asnumpy(power_d)
    freqs_np = np.asarray(frequencies, dtype=np.float64)
    return _result(power_np, yy, freqs_np,
                   lambda bf: eebls(t, y, dy, np.array([bf]), nbins=nbins,
                                    qmin=qmin, qmax=qmax,
                                    min_points=min_points))


def multiband_eebls_gpu(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    qmin: float = 0.01,
    qmax: float = 0.10,
    min_points: int = 3,
) -> BLSResult:
    """GPU binned multiband BLS. Mirrors :func:`multiband_bls.multiband_eebls`.

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch, and duration.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / qmin, 50, 500)``
        if ``None``.
    qmin :
        Minimum transit duration as a fraction of the period.
    qmax :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required across all bands.
    """
    import cupy as cp

    if nbins is None:
        nbins = auto_nbins(qmin)
    _kernels()
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    n_bands = len(labels)
    if n_bands > _MAX_BANDS:
        raise ValueError(f"GPU multiband kernel supports up to {_MAX_BANDS} bands")

    t_d = cp.asarray(t_all, dtype=cp.float64)
    wx_d = cp.asarray(wx_all, dtype=cp.float64)
    w_d = cp.asarray(w_all, dtype=cp.float64)
    band_d = cp.asarray(band_all, dtype=cp.int32)
    bw_d = cp.asarray(w_totals, dtype=cp.float64)
    f_d = cp.asarray(frequencies, dtype=cp.float64)
    nf = int(f_d.size)
    power_d = cp.empty(nf, dtype=cp.float64)
    kmi, kma = _grid_params(nbins, qmin, qmax)

    shared = 3 * n_bands * nbins * 8
    _multi_kernel((nf,), (_THREADS,),
                  (t_d, wx_d, w_d, band_d, bw_d, np.int32(t_d.size), np.int32(n_bands),
                   f_d, np.int32(nbins), np.int32(kmi), np.int32(kma),
                   np.int32(min_points), power_d),
                  shared_mem=shared)
    power = cp.asnumpy(power_d)
    return _result(power, chi2_flat, np.asarray(frequencies, dtype=float),
                   lambda bf: multiband_eebls(bands, np.array([bf]), nbins=nbins,
                                              qmin=qmin, qmax=qmax,
                                              min_points=min_points),
                   bands=labels)


def eebls_gpu_fast(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    qmin: float = 0.01,
    qmax: float = 0.10,
    min_points: int = 3,
    dlogq: float = 0.3,
) -> BLSResult:
    """Fast GPU binned BLS using incremental accumulation and log-spaced transit widths.

    Instead of evaluating all integer widths from ``kmi`` to ``kma``, only
    evaluates a geometrically-spaced subset, reducing window evaluations from
    O(kma - kmi) to O(log(kma/kmi) / log(1 + dlogq)).

    Parameters
    ----------
    t :
        Observation times (days), 1-D float array.
    y :
        Magnitudes or fluxes, 1-D float array, same length as ``t``.
    dy :
        1-sigma uncertainties, 1-D float array, same length as ``t``.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / qmin, 50, 500)``
        if ``None``.
    qmin :
        Minimum transit duration as a fraction of the period.
    qmax :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required.
    dlogq :
        Logarithmic spacing between trial widths. Smaller = more widths = higher
        fidelity but slower. ``dlogq=0`` recovers the full linear search.
        Typical range: 0.1 – 0.5.

    Notes
    -----
    Introduces a small (~1–2%) loss in peak power because the exact optimal
    integer width may not fall in the log-spaced set. For detection this is
    negligible; for precise parameter estimation use :func:`eebls_gpu` or
    :func:`eebls` at the recovered period.
    """
    import cupy as cp

    if nbins is None:
        nbins = auto_nbins(qmin)

    _kernels()  # ensure compiled
    wx, w_hat, yy = _preprocess_single(y, dy)

    # Subtract the median before binning so that large JD values (t ~ 2.4e6)
    # don't exhaust float32 precision in the shared-memory bin accumulators.
    t_offset = float(np.median(t))
    t_d = cp.asarray(np.asarray(t, dtype=np.float64) - t_offset, dtype=cp.float64)
    wx_d = cp.asarray(wx, dtype=cp.float64)
    w_d = cp.asarray(w_hat, dtype=cp.float64)
    f_d = cp.asarray(frequencies, dtype=cp.float64)
    nf = int(f_d.size)
    kmi, kma = _grid_params(nbins, qmin, qmax)

    widths_np = _log_widths(kmi, kma, dlogq)
    widths_d = cp.asarray(widths_np, dtype=cp.int32)
    n_widths = int(widths_np.size)
    logger.debug("eebls_gpu_fast: nb=%d kmi=%d kma=%d n_widths=%d (linear=%d)",
                 nbins, kmi, kma, n_widths, kma - kmi + 1)

    power_d = cp.empty(nf, dtype=cp.float64)
    min_r = np.float32(min_points / t_d.size)
    shared = (2 * nbins + 8) * 4  # float32: ybin[nb] + rbin[nb] + warp_max[8]
    n_blocks = min(nf, 2048)
    _fast_single_kernel(
        (n_blocks,), (_THREADS,),
        (t_d, wx_d, w_d, np.int32(t_d.size), np.int32(nf), f_d, np.int32(nbins),
         widths_d, np.int32(n_widths), min_r, power_d),
        shared_mem=shared,
    )
    power_np = cp.asnumpy(power_d)
    freqs_np = np.asarray(frequencies, dtype=np.float64)
    return _result(power_np, yy, freqs_np,
                   lambda bf: eebls(t, y, dy, np.array([bf]), nbins=nbins,
                                    qmin=qmin, qmax=qmax,
                                    min_points=min_points))


def multiband_eebls_gpu_fast(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    qmin: float = 0.01,
    qmax: float = 0.10,
    min_points: int = 3,
    dlogq: float = 0.3,
) -> BLSResult:
    """Fast GPU binned multiband BLS using incremental accumulation and log-spaced widths.

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch, and duration.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / qmin, 50, 500)``
        if ``None``.
    qmin :
        Minimum transit duration as a fraction of the period.
    qmax :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required across all bands.
    dlogq :
        Logarithmic width spacing. Smaller = more widths = higher fidelity.
        Typical range: 0.1 – 0.5.

    Notes
    -----
    Introduces a small (~1–2%) loss in peak power because the exact optimal
    integer width may not fall in the log-spaced set.
    """
    import cupy as cp

    if nbins is None:
        nbins = auto_nbins(qmin)

    _kernels()  # ensure compiled
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    n_bands = len(labels)
    if n_bands > _MAX_BANDS:
        raise ValueError(f"GPU multiband kernel supports up to {_MAX_BANDS} bands")

    # Subtract the median before binning so that large JD values (t ~ 2.4e6)
    # don't exhaust float32 precision in the shared-memory bin accumulators.
    t_offset = float(np.median(t_all))
    t_d = cp.asarray(t_all - t_offset, dtype=cp.float64)
    wx_d = cp.asarray(wx_all, dtype=cp.float64)
    w_d = cp.asarray(w_all, dtype=cp.float64)
    band_d = cp.asarray(band_all, dtype=cp.int32)
    bw_d = cp.asarray(w_totals, dtype=cp.float64)
    f_d = cp.asarray(frequencies, dtype=cp.float64)
    nf = int(f_d.size)
    kmi, kma = _grid_params(nbins, qmin, qmax)

    widths_np = _log_widths(kmi, kma, dlogq)
    widths_d = cp.asarray(widths_np, dtype=cp.int32)
    n_widths = int(widths_np.size)
    logger.debug("multiband_eebls_gpu_fast: nb=%d kmi=%d kma=%d n_widths=%d (linear=%d)",
                 nbins, kmi, kma, n_widths, kma - kmi + 1)

    power_d = cp.empty(nf, dtype=cp.float64)
    n_blocks = min(nf, 2048)
    shared = (3 * n_bands * nbins + 8) * 4  # float32: ybin + rbin + cbin (per band) + warp_max[8]
    _fast_multi_kernel(
        (n_blocks,), (_THREADS,),
        (t_d, wx_d, w_d, band_d, bw_d, np.int32(t_d.size), np.int32(n_bands),
         np.int32(nf), f_d, np.int32(nbins), widths_d, np.int32(n_widths),
         np.int32(min_points), power_d),
        shared_mem=shared,
    )
    power = cp.asnumpy(power_d)
    return _result(power, chi2_flat, np.asarray(frequencies, dtype=float),
                   lambda bf: multiband_eebls(bands, np.array([bf]), nbins=nbins,
                                              qmin=qmin, qmax=qmax,
                                              min_points=min_points),
                   bands=labels)
