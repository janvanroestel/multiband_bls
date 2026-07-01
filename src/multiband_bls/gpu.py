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

_THREADS = 256  # threads per block. Hard-coded into the kernels: the exact
# kernels size their static reduction as `red[256]`, and the fast kernels assume
# exactly 256/32 = 8 warps (`warp_max[8]`). Changing this requires editing those
# reduction sizes in the kernel source and recompiling.
# Stack arrays in the CUDA multiband kernels are fixed-size; recompile after
# changing _MAX_BANDS (update every fixed [8] per-band array declaration in the
# kernel sources: bw, s/r/cnt in the exact kernel; bw, cy0/cr0/toty/totr,
# sy_b/sr_b in the fast kernel).
_MAX_BANDS: int = 8

# --------------------------------------------------------------------------- #
# CUDA kernels                                                                 #
# --------------------------------------------------------------------------- #
_SINGLE_SRC = r"""
// Exact single-band binned BLS: full linear width sweep in double precision.
// Persistent grid-stride over frequencies (blocks capped in Python) so global
// inputs stay hot in L2 across trials; __restrict__ + __ldg on all global reads.
extern "C" __global__ __launch_bounds__(256)
void eebls_kernel(const double* __restrict__ t, const double* __restrict__ wx,
                  const double* __restrict__ w, int n, int nf,
                  const double* __restrict__ freqs, int nb, int kmi, int kma,
                  int min_bin_points, double* power) {
    extern __shared__ double sh[];           // ybin[nb], rbin[nb], cbin[nb]
    double* ybin = sh;
    double* rbin = sh + nb;
    double* cbin = sh + 2 * nb;
    int tid = threadIdx.x;
    __shared__ double red[256];              // sized for _THREADS = 256

    for (int f = blockIdx.x; f < nf; f += gridDim.x) {
        double freq = __ldg(&freqs[f]);

        for (int b = tid; b < nb; b += blockDim.x) { ybin[b] = 0; rbin[b] = 0; cbin[b] = 0; }
        __syncthreads();

        for (int i = tid; i < n; i += blockDim.x) {
            double ph = __ldg(&t[i]) * freq; ph -= floor(ph);
            int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
            atomicAdd(&ybin[ib], __ldg(&wx[i]));
            atomicAdd(&rbin[ib], __ldg(&w[i]));
            atomicAdd(&cbin[ib], 1.0);
        }
        __syncthreads();

        double best = 0.0;
        for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
            double s = 0.0, r = 0.0; int cnt = 0;
            for (int ww = 1; ww <= kma; ++ww) {
                int jj = i1 + ww - 1; if (jj >= nb) jj -= nb;  // single wrap safe: kma <= nb
                s += ybin[jj]; r += rbin[jj]; cnt += (int)cbin[jj];
                if (ww < kmi) continue;
                if (cnt < min_bin_points) continue;
                double denom = r * (1.0 - r);
                if (denom <= 0.0) continue;
                double sr2 = s * s / denom;
                if (sr2 > best) best = sr2;
            }
        }
        red[tid] = best; __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) { if (red[tid + stride] > red[tid]) red[tid] = red[tid + stride]; }
            __syncthreads();
        }
        if (tid == 0) power[f] = red[0];
        __syncthreads();  // ensure red[0] is read before next f reuses it
    }
}
"""

_MULTI_SRC = r"""
// Exact multiband binned BLS in double precision. Persistent grid-stride over
// frequencies (blocks capped in Python) with __restrict__ + __ldg on global reads;
// band weights cached in registers.
extern "C" __global__ __launch_bounds__(256)
void meebls_kernel(const double* __restrict__ t, const double* __restrict__ wx,
                   const double* __restrict__ w,
                   const int* __restrict__ band, const double* __restrict__ band_w,
                   int n, int n_bands, int nf,
                   const double* __restrict__ freqs, int nb, int kmi, int kma,
                   int min_points, double* power) {
    extern __shared__ double sh[];           // ybin/rbin/cbin each n_bands*nb
    double* ybin = sh;
    double* rbin = sh + n_bands * nb;
    double* cbin = sh + 2 * n_bands * nb;
    int tid = threadIdx.x;
    int tot = n_bands * nb;
    __shared__ double red[256];              // sized for _THREADS = 256

    double bw[8];                            // band weights cached in registers
    for (int b = 0; b < n_bands; b++) bw[b] = __ldg(&band_w[b]);

    for (int f = blockIdx.x; f < nf; f += gridDim.x) {
        double freq = __ldg(&freqs[f]);

        for (int b = tid; b < tot; b += blockDim.x) { ybin[b] = 0; rbin[b] = 0; cbin[b] = 0; }
        __syncthreads();

        for (int i = tid; i < n; i += blockDim.x) {
            double ph = __ldg(&t[i]) * freq; ph -= floor(ph);
            int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
            int bb = __ldg(&band[i]);
            atomicAdd(&ybin[bb * nb + ib], __ldg(&wx[i]));
            atomicAdd(&rbin[bb * nb + ib], __ldg(&w[i]));
            atomicAdd(&cbin[bb * nb + ib], 1.0);
        }
        __syncthreads();

        double best = 0.0;
        double s[8], r[8]; int cnt[8];
        for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
            for (int b = 0; b < n_bands; b++) { s[b] = 0; r[b] = 0; cnt[b] = 0; }
            int total = 0;
            for (int ww = 1; ww <= kma; ++ww) {
                int jj = i1 + ww - 1; if (jj >= nb) jj -= nb;  // single wrap safe: kma <= nb
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
                    T3 += bw[b] * s[b] * s[b] / denom;
                }
                if (T3 > best) best = T3;
            }
        }
        red[tid] = best; __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) { if (red[tid + stride] > red[tid]) red[tid] = red[tid + stride]; }
            __syncthreads();
        }
        if (tid == 0) power[f] = red[0];
        __syncthreads();  // ensure red[0] is read before next f reuses it
    }
}
"""

_FAST_HELPERS = r"""
// float-float ("double-single") phase fold: t and freq are split into
// (hi, lo) float32 pairs; the exact error of the hi*hi product is recovered
// with an fmaf twoProd and the cross terms added to it. Absolute phase error
// ~1e-10 (bin width is >= 1/500), at full FP32 rate instead of the 1/64-rate
// FP64 multiply+floor on consumer GPUs. Requires |t| reduced (median
// subtracted on the host) so t*freq stays well inside float range.
__device__ __forceinline__ float fold_ff(float t_hi, float t_lo,
                                         float f_hi, float f_lo) {
    float p = t_hi * f_hi;
    float e = fmaf(t_hi, f_hi, -p);       // exact rounding error of hi*hi
    e += t_hi * f_lo + t_lo * f_hi;       // cross terms (~ulp-scale)
    float ph = (p - floorf(p)) + e;
    ph -= floorf(ph);                     // wrap to [0, 1)
    return ph;
}

// inclusive warp scan (Hillis-Steele over lanes)
__device__ __forceinline__ float warp_scan1(float v, int lane) {
    for (int off = 1; off < 32; off <<= 1) {
        float a = __shfl_up_sync(0xffffffff, v, off);
        if (lane >= off) v += a;
    }
    return v;
}
"""

_FAST_BOX_SINGLE_SRC = _FAST_HELPERS + r"""
// Prefix-sum box BLS kernel with log-spaced transit widths.
// After binning, ybin/rbin are turned in-place into inclusive cumulative sums
// (one warp per array), so every window sum is an O(1) two-element difference
// and the sweep touches only the log-spaced widths instead of every integer
// width. Float32 bins; float-float phase fold; SR^2 comparison; warp-shuffle
// reduction; persistent grid-stride; __launch_bounds__ + __ldg.
// min_r replaces a count check: weights are normalised to sum 1, so
// sum_r < min_r (= min_bin_points/n) reliably filters under-populated windows.
// Shared memory (float32):
//   ybin[nb], rbin[nb]              2*nb   (bins, then their cumsums)
//   warp_max[8]                     8
// Total: (2*nb + 8) * 4 bytes  (~4.0 KB for nb=500)
extern "C" __global__ __launch_bounds__(256, 6)
void eebls_fast_kernel(
    const float* __restrict__ t_hi,
    const float* __restrict__ t_lo,
    const float* __restrict__ wx,
    const float* __restrict__ w,
    int n, int nf,
    const double* __restrict__ freqs, int nb,
    const int* __restrict__ widths, int n_widths,
    float min_r, double* power)
{
    extern __shared__ float sh[];
    float* ybin     = sh;
    float* rbin     = sh + nb;
    float* warp_max = sh + 2*nb;  // 8 floats (one per warp, 256/32=8)

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    for (int f = blockIdx.x; f < nf; f += gridDim.x) {
        double fd = __ldg(&freqs[f]);
        float f_hi = (float)fd;
        float f_lo = (float)(fd - (double)f_hi);

        // Phase 1: clear bins
        for (int b = tid; b < 2 * nb; b += blockDim.x) sh[b] = 0.f;
        __syncthreads();

        // Phase 2: bin observations (float-float phase, float32 accumulate)
        for (int i = tid; i < n; i += blockDim.x) {
            float ph = fold_ff(__ldg(&t_hi[i]), __ldg(&t_lo[i]), f_hi, f_lo);
            int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
            atomicAdd(&ybin[ib], __ldg(&wx[i]));
            atomicAdd(&rbin[ib], __ldg(&w[i]));
        }
        __syncthreads();

        // Phase 3a: in-place inclusive scan of ybin and rbin (one warp each;
        // the carry is propagated serially across 32-lane chunks).
        for (int a = warp; a < 2; a += (blockDim.x >> 5)) {
            float* arr = sh + a * nb;
            float carry = 0.f;
            for (int chunk = 0; chunk < nb; chunk += 32) {
                int i = chunk + lane;
                float v = (i < nb) ? arr[i] : 0.f;
                v = warp_scan1(v, lane);
                if (i < nb) arr[i] = v + carry;
                carry += __shfl_sync(0xffffffff, v, 31);
            }
        }
        __syncthreads();

        // Phase 3b: window sweep - O(1) cumsum differences per (i1, width).
        // Wrap-around windows use the array totals (single wrap: kma <= nb).
        float best2 = 0.f;
        for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
            float cy0  = (i1 > 0) ? ybin[i1 - 1] : 0.f;
            float cr0  = (i1 > 0) ? rbin[i1 - 1] : 0.f;
            float toty = ybin[nb - 1];
            float totr = rbin[nb - 1];
            for (int wi = 0; wi < n_widths; wi++) {
                int ww = __ldg(&widths[wi]);
                int j2 = i1 + ww - 1;
                float sy, sr;
                if (j2 < nb) {
                    sy = ybin[j2] - cy0;
                    sr = rbin[j2] - cr0;
                } else {
                    j2 -= nb;
                    sy = (toty - cy0) + ybin[j2];
                    sr = (totr - cr0) + rbin[j2];
                }
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

_FAST_BOX_MULTI_SRC = _FAST_HELPERS + r"""
// Prefix-sum multiband box BLS with log-spaced widths. Handles 1-8 bands.
// Same design as the single-band fast kernel: per-band ybin/rbin are scanned
// in place into inclusive cumsums (one warp per array), window sums are O(1)
// differences, and only the log-spaced widths are evaluated. The per-bin
// count array of the previous version is gone: min_points is enforced as a
// weighted threshold on the summed normalised weight across bands
// (min_r = min_points / n), like the single-band fast kernel.
// Float32 bins; float-float phase fold; band_w in registers; SR^2 comparison;
// warp-shuffle reduction; persistent grid-stride; __launch_bounds__ + __ldg.
// Shared memory: (2*n_bands*nb + 8) * 4 bytes  (~23.5 KB for n_bands=6, nb=500)
extern "C" __global__ __launch_bounds__(256, 4)
void meebls_fast_kernel(
    const float* __restrict__ t_hi,
    const float* __restrict__ t_lo,
    const float* __restrict__ wx,
    const float* __restrict__ w,
    const int* __restrict__ band,
    const double* __restrict__ band_w,
    int n, int n_bands, int nf,
    const double* __restrict__ freqs, int nb,
    const int* __restrict__ widths, int n_widths,
    float min_r, double* power)
{
    extern __shared__ float sh[];             // ybin[b][nb] then rbin[b][nb]
    float* warp_max = sh + 2 * n_bands * nb;  // 8 floats (one per warp)

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int tot  = n_bands * nb;

    // Load band weights into registers (avoids repeated global-mem reads in hot loop)
    float bw[8];
    for (int b = 0; b < n_bands; b++) bw[b] = (float)__ldg(&band_w[b]);

    for (int f = blockIdx.x; f < nf; f += gridDim.x) {
        double fd = __ldg(&freqs[f]);
        float f_hi = (float)fd;
        float f_lo = (float)(fd - (double)f_hi);

        for (int b = tid; b < 2 * tot; b += blockDim.x) sh[b] = 0.f;
        __syncthreads();

        for (int i = tid; i < n; i += blockDim.x) {
            float ph = fold_ff(__ldg(&t_hi[i]), __ldg(&t_lo[i]), f_hi, f_lo);
            int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
            int bb = __ldg(&band[i]);
            atomicAdd(&sh[bb * nb + ib],       __ldg(&wx[i]));
            atomicAdd(&sh[tot + bb * nb + ib], __ldg(&w[i]));
        }
        __syncthreads();

        // In-place inclusive scan of each of the 2*n_bands arrays
        // (one warp per array, strided over the 8 warps).
        for (int a = warp; a < 2 * n_bands; a += (blockDim.x >> 5)) {
            float* arr = sh + a * nb;
            float carry = 0.f;
            for (int chunk = 0; chunk < nb; chunk += 32) {
                int i = chunk + lane;
                float v = (i < nb) ? arr[i] : 0.f;
                v = warp_scan1(v, lane);
                if (i < nb) arr[i] = v + carry;
                carry += __shfl_sync(0xffffffff, v, 31);
            }
        }
        __syncthreads();

        float best2 = 0.f;
        for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
            // hoist the left cumsum edge and per-band totals out of the width loop
            float cy0[8], cr0[8], toty[8], totr[8];
            for (int b = 0; b < n_bands; b++) {
                const float* cy = sh + b * nb;
                const float* cr = sh + (n_bands + b) * nb;
                cy0[b]  = (i1 > 0) ? cy[i1 - 1] : 0.f;
                cr0[b]  = (i1 > 0) ? cr[i1 - 1] : 0.f;
                toty[b] = cy[nb - 1];
                totr[b] = cr[nb - 1];
            }
            for (int wi = 0; wi < n_widths; wi++) {
                int ww = __ldg(&widths[wi]);
                int j2 = i1 + ww - 1;
                float sr_tot = 0.f, sy_b[8], sr_b[8];
                if (j2 < nb) {
                    for (int b = 0; b < n_bands; b++) {
                        sy_b[b] = sh[b * nb + j2] - cy0[b];
                        sr_b[b] = sh[(n_bands + b) * nb + j2] - cr0[b];
                        sr_tot += sr_b[b];
                    }
                } else {                       // single wrap safe: kma <= nb
                    int j2w = j2 - nb;
                    for (int b = 0; b < n_bands; b++) {
                        sy_b[b] = (toty[b] - cy0[b]) + sh[b * nb + j2w];
                        sr_b[b] = (totr[b] - cr0[b]) + sh[(n_bands + b) * nb + j2w];
                        sr_tot += sr_b[b];
                    }
                }
                if (sr_tot < min_r) continue;
                float sr2 = 0.f;
                for (int b = 0; b < n_bands; b++) {
                    float denom = sr_b[b] * (1.f - sr_b[b]);
                    if (denom <= 0.f) continue;
                    sr2 += bw[b] * sy_b[b] * sy_b[b] / denom;
                }
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
        # Opt into the device maximum for dynamic shared memory. The requested
        # dynamic size must leave room for the kernel's own static shared memory
        # (the exact kernels carry a static `red[256]` reduction buffer), so
        # subtract that from the opt-in ceiling.
        _multi_kernel.max_dynamic_shared_size_bytes = (
            _max_smem - _multi_kernel.attributes["shared_size_bytes"]
        )
        _fast_single_kernel.max_dynamic_shared_size_bytes = (
            _max_smem - _fast_single_kernel.attributes["shared_size_bytes"]
        )
        _fast_multi_kernel.max_dynamic_shared_size_bytes = (
            _max_smem - _fast_multi_kernel.attributes["shared_size_bytes"]
        )
    # fast kernels are accessed as module globals by the fast entry points


def _split_time(t: Array) -> tuple[Array, Array]:
    """Split times into (hi, lo) float32 pairs with ``t ~= hi + lo``.

    Feeds the float-float phase fold in the fast kernels. ``t`` must already
    be median-reduced so the products ``t * freq`` stay well inside the range
    where the two-float representation holds full double-like precision.
    """
    t64 = np.asarray(t, dtype=np.float64)
    t_hi = t64.astype(np.float32)
    t_lo = (t64 - t_hi.astype(np.float64)).astype(np.float32)
    return t_hi, t_lo


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


def _grid_params(nbins: int, q_min: float, q_max: float) -> tuple[int, int]:
    kmi = max(1, int(q_min * nbins))
    kma = min(nbins, int(q_max * nbins) + 1)
    # Load-bearing invariant: every kernel wraps the circular window with a single
    # subtraction (`if (jj >= nb) jj -= nb`), which is only correct when the widest
    # window (kma) does not exceed nbins. Violating this would silently read/write
    # out of bounds in shared memory.
    assert kma <= nbins, f"kma ({kma}) must not exceed nbins ({nbins})"
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
    q_min: float = 0.01,
    q_max: float = 0.10,
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
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required.
    """
    import cupy as cp

    if nbins is None:
        nbins = auto_nbins(q_min)
    _kernels()
    wx, w_hat, yy = _preprocess_single(y, dy)

    t_d = cp.asarray(t, dtype=cp.float64)
    wx_d = cp.asarray(wx, dtype=cp.float64)
    w_d = cp.asarray(w_hat, dtype=cp.float64)
    f_d = cp.asarray(frequencies, dtype=cp.float64)
    nf = int(f_d.size)
    kmi, kma = _grid_params(nbins, q_min, q_max)
    shared = 3 * nbins * 8  # ybin/rbin/cbin (float64)

    power_d = cp.empty(nf, dtype=cp.float64)
    n_blocks = min(nf, 2048)
    _single_kernel((n_blocks,), (_THREADS,),
                   (t_d, wx_d, w_d, np.int32(t_d.size), np.int32(nf), f_d,
                    np.int32(nbins), np.int32(kmi), np.int32(kma),
                    np.int32(min_points), power_d),
                   shared_mem=shared)
    power_np = cp.asnumpy(power_d)
    freqs_np = np.asarray(frequencies, dtype=np.float64)
    return _result(power_np, yy, freqs_np,
                   lambda bf: eebls(t, y, dy, np.array([bf]), nbins=nbins,
                                    q_min=q_min, q_max=q_max,
                                    min_points=min_points))


def multiband_eebls_gpu(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
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
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required across all bands.
    """
    import cupy as cp

    if nbins is None:
        nbins = auto_nbins(q_min)
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
    kmi, kma = _grid_params(nbins, q_min, q_max)

    shared = 3 * n_bands * nbins * 8
    smem_cap = _multi_kernel.max_dynamic_shared_size_bytes
    if shared > smem_cap:
        raise ValueError(
            f"multiband_eebls_gpu needs {shared} B of dynamic shared memory "
            f"(3 * n_bands={n_bands} * nbins={nbins} * 8) but the device allows "
            f"{smem_cap} B; reduce nbins or n_bands."
        )
    n_blocks = min(nf, 2048)
    _multi_kernel((n_blocks,), (_THREADS,),
                  (t_d, wx_d, w_d, band_d, bw_d, np.int32(t_d.size), np.int32(n_bands),
                   np.int32(nf), f_d, np.int32(nbins), np.int32(kmi), np.int32(kma),
                   np.int32(min_points), power_d),
                  shared_mem=shared)
    power = cp.asnumpy(power_d)
    return _result(power, chi2_flat, np.asarray(frequencies, dtype=float),
                   lambda bf: multiband_eebls(bands, np.array([bf]), nbins=nbins,
                                              q_min=q_min, q_max=q_max,
                                              min_points=min_points),
                   bands=labels)


def eebls_gpu_fast(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    dlogq: float = 0.3,
) -> BLSResult:
    """Fast GPU binned BLS using prefix-sum bins and log-spaced transit widths.

    Instead of evaluating all integer widths from ``kmi`` to ``kma``, only
    evaluates a geometrically-spaced subset, reducing window evaluations from
    O(kma - kmi) to O(log(kma/kmi) / log(1 + dlogq)). The phase bins are
    turned into cumulative sums so each window is an O(1) lookup, and the
    phase fold runs in float-float (two-float32) arithmetic, which is several
    times faster than float64 on consumer GPUs at ~1e-10 phase error.

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
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
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

    Unlike the exact :func:`eebls_gpu`, ``min_points`` is applied here as a
    *weighted-fraction* threshold: this kernel tracks no per-bin count, so it
    rejects windows whose summed normalised weight is below ``min_points / n``.
    This matches an exact in-transit point count only when all points share the
    same uncertainty; with heteroscedastic ``dy`` the two filters diverge.
    """
    import cupy as cp

    if nbins is None:
        nbins = auto_nbins(q_min)

    _kernels()  # ensure compiled
    wx, w_hat, yy = _preprocess_single(y, dy)

    # Subtract the median so large JD values (t ~ 2.4e6) keep full precision
    # in the two-float split (and the float32 bin accumulators).
    t_offset = float(np.median(t))
    t_hi, t_lo = _split_time(np.asarray(t, dtype=np.float64) - t_offset)
    thi_d = cp.asarray(t_hi, dtype=cp.float32)
    tlo_d = cp.asarray(t_lo, dtype=cp.float32)
    wx_d = cp.asarray(wx, dtype=cp.float32)
    w_d = cp.asarray(w_hat, dtype=cp.float32)
    f_d = cp.asarray(frequencies, dtype=cp.float64)
    nf = int(f_d.size)
    n = int(thi_d.size)
    kmi, kma = _grid_params(nbins, q_min, q_max)

    widths_np = _log_widths(kmi, kma, dlogq)
    widths_d = cp.asarray(widths_np, dtype=cp.int32)
    n_widths = int(widths_np.size)
    logger.debug("eebls_gpu_fast: nb=%d kmi=%d kma=%d n_widths=%d (linear=%d)",
                 nbins, kmi, kma, n_widths, kma - kmi + 1)

    power_d = cp.empty(nf, dtype=cp.float64)
    min_r = np.float32(min_points / n)
    shared = (2 * nbins + 8) * 4  # float32: ybin[nb] + rbin[nb] + warp_max[8]
    smem_cap = _fast_single_kernel.max_dynamic_shared_size_bytes
    if shared > smem_cap:
        raise ValueError(
            f"eebls_gpu_fast needs {shared} B of dynamic shared memory "
            f"((2 * nbins={nbins} + 8) * 4) but the device allows "
            f"{smem_cap} B; reduce nbins."
        )
    n_blocks = min(nf, 2048)
    _fast_single_kernel(
        (n_blocks,), (_THREADS,),
        (thi_d, tlo_d, wx_d, w_d, np.int32(n), np.int32(nf), f_d, np.int32(nbins),
         widths_d, np.int32(n_widths), min_r, power_d),
        shared_mem=shared,
    )
    power_np = cp.asnumpy(power_d)
    freqs_np = np.asarray(frequencies, dtype=np.float64)
    return _result(power_np, yy, freqs_np,
                   lambda bf: eebls(t, y, dy, np.array([bf]), nbins=nbins,
                                    q_min=q_min, q_max=q_max,
                                    min_points=min_points))


def multiband_eebls_gpu_fast(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    dlogq: float = 0.3,
) -> BLSResult:
    """Fast GPU binned multiband BLS using prefix-sum bins and log-spaced widths.

    Same design as :func:`eebls_gpu_fast`: cumulative-sum phase bins (O(1)
    window sums), a geometrically-spaced width subset, and a float-float
    phase fold.

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch, and duration.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
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

    Like :func:`eebls_gpu_fast`, ``min_points`` is applied as a
    *weighted-fraction* threshold on the summed per-band normalised weights
    (``min_r = min_points * n_bands / n``), not an exact in-transit count;
    the two coincide when all points share the same uncertainty.
    """
    import cupy as cp

    if nbins is None:
        nbins = auto_nbins(q_min)

    _kernels()  # ensure compiled
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    n_bands = len(labels)
    if n_bands > _MAX_BANDS:
        raise ValueError(f"GPU multiband kernel supports up to {_MAX_BANDS} bands")

    # Subtract the median so large JD values (t ~ 2.4e6) keep full precision
    # in the two-float split (and the float32 bin accumulators).
    t_offset = float(np.median(t_all))
    t_hi, t_lo = _split_time(t_all - t_offset)
    thi_d = cp.asarray(t_hi, dtype=cp.float32)
    tlo_d = cp.asarray(t_lo, dtype=cp.float32)
    wx_d = cp.asarray(wx_all, dtype=cp.float32)
    w_d = cp.asarray(w_all, dtype=cp.float32)
    band_d = cp.asarray(band_all, dtype=cp.int32)
    bw_d = cp.asarray(w_totals, dtype=cp.float64)
    f_d = cp.asarray(frequencies, dtype=cp.float64)
    nf = int(f_d.size)
    n = int(thi_d.size)
    kmi, kma = _grid_params(nbins, q_min, q_max)

    widths_np = _log_widths(kmi, kma, dlogq)
    widths_d = cp.asarray(widths_np, dtype=cp.int32)
    n_widths = int(widths_np.size)
    logger.debug("multiband_eebls_gpu_fast: nb=%d kmi=%d kma=%d n_widths=%d (linear=%d)",
                 nbins, kmi, kma, n_widths, kma - kmi + 1)

    power_d = cp.empty(nf, dtype=cp.float64)
    n_blocks = min(nf, 2048)
    # min_points -> weighted threshold on the summed per-band normalised
    # weights. Each band's weights sum to 1, so the average per-point weight
    # is n_bands / n and min_r = min_points * n_bands / n matches an exact
    # count when all points carry equal uncertainty.
    min_r = np.float32(min_points * n_bands / n)
    shared = (2 * n_bands * nbins + 8) * 4  # float32: ybin + rbin (per band) + warp_max[8]
    smem_cap = _fast_multi_kernel.max_dynamic_shared_size_bytes
    if shared > smem_cap:
        raise ValueError(
            f"multiband_eebls_gpu_fast needs {shared} B of dynamic shared memory "
            f"((2 * n_bands={n_bands} * nbins={nbins} + 8) * 4) but the device "
            f"allows {smem_cap} B; reduce nbins or n_bands."
        )
    _fast_multi_kernel(
        (n_blocks,), (_THREADS,),
        (thi_d, tlo_d, wx_d, w_d, band_d, bw_d, np.int32(n), np.int32(n_bands),
         np.int32(nf), f_d, np.int32(nbins), widths_d, np.int32(n_widths),
         min_r, power_d),
        shared_mem=shared,
    )
    power = cp.asnumpy(power_d)
    return _result(power, chi2_flat, np.asarray(frequencies, dtype=float),
                   lambda bf: multiband_eebls(bands, np.array([bf]), nbins=nbins,
                                              q_min=q_min, q_max=q_max,
                                              min_points=min_points),
                   bands=labels)
