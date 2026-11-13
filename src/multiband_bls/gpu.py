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
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from .api import (
    _check_dlogq,
    _grid_params,
    _log_widths,
    _merge_bands,
    _preprocess_single,
    _result,
    eebls,
    multiband_eebls,
)
from .periodogram import BLSResult
from .reference import (
    _check_phase_range,
    _prepare_bands,
    _validate_light_curve,
    _validate_search_params,
    auto_nbins,
)

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]

_THREADS = 256  # threads per block. Hard-coded into the kernels: the exact
# kernels size their static reduction as `red[256]`, and the fast kernels assume
# exactly 256/32 = 8 warps (`warp_max[8]`). Changing this requires editing those
# reduction sizes in the kernel source and recompiling.
# Stack arrays in the CUDA multiband kernels are fixed-size; recompile after
# changing _MAX_BANDS (update every fixed [8] per-band array declaration in the
# kernel sources: bw, s/r/cnt in the exact kernel; bw in the fast kernel).
_MAX_BANDS: int = 8
# The fast kernels count in-transit points in float32 bins; integer sums are
# exact only below 2**24, so larger light curves are rejected.
_MAX_FAST_POINTS: int = 2**24
# The kernels take point and frequency counts as C int; larger inputs would wrap.
_MAX_INT32: int = int(np.iinfo(np.int32).max)
# The fast kernels hold median-centred times and frequencies in float32; keep
# both far below float32's ~3.4e38 so the casts and fold products stay finite.
_MAX_FAST_FLOAT32: float = 1e30
# The float-float fold's absolute phase error grows ~ 7e-15 per cycle of
# max|t_centred| * max(f) (measured: <= 4e-8 up to 1e6 cycles, 7e-7 at 1e8, 7e-5 at
# 1e10); beyond 1e8 cycles it is no longer small against a 1/nbins bin.
_MAX_FAST_CYCLES: float = 1e8


def _check_int32_sizes(n: int, nf: int, name: str) -> None:
    """Raise unless the point and frequency counts fit the kernels' 32-bit ints."""
    if n > _MAX_INT32 or nf > _MAX_INT32:
        raise ValueError(
            f"{name}: {n} points and {nf} frequencies; both must be at most "
            f"{_MAX_INT32} (32-bit kernel indices). Split the frequency grid."
        )


def _check_fast_float32(t_centered: Array, frequencies: Array, name: str) -> None:
    """Raise unless centred times and frequencies fit the fast kernels' float32."""
    t_max = float(np.max(np.abs(t_centered)))
    f_max = float(np.max(frequencies))
    if t_max > _MAX_FAST_FLOAT32 or f_max > _MAX_FAST_FLOAT32:
        raise ValueError(
            f"{name}: median-centred |t| ({t_max:.3g}) and frequencies "
            f"({f_max:.3g}) must stay below {_MAX_FAST_FLOAT32:.0e} for the "
            "float32 kernel; use the exact GPU entry point."
        )
    if t_max * f_max > _MAX_FAST_CYCLES:
        raise ValueError(
            f"{name}: median-centred max|t| * max(f) = {t_max * f_max:.3g} cycles "
            f"exceeds {_MAX_FAST_CYCLES:.0e}, beyond which the float32 phase fold "
            "loses accuracy; use the exact GPU entry point."
        )


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
                if (cnt >= n) continue;  // no out-of-transit point: depth not identifiable
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
                   const int* __restrict__ band_n,
                   int n, int n_bands, int nf,
                   const double* __restrict__ freqs, int nb, int kmi, int kma,
                   int min_points, double* power, int* kout) {
    extern __shared__ double sh[];           // ybin/rbin/cbin each n_bands*nb
    double* ybin = sh;
    double* rbin = sh + n_bands * nb;
    double* cbin = sh + 2 * n_bands * nb;
    int tid = threadIdx.x;
    int tot = n_bands * nb;
    __shared__ double red[256];              // sized for _THREADS = 256
    __shared__ int red_k[256];               // bands used by red[] (n_bands_used)

    double bw[8];                            // band weights cached in registers
    int bn[8];                               // points per band
    for (int b = 0; b < n_bands; b++) { bw[b] = __ldg(&band_w[b]); bn[b] = __ldg(&band_n[b]); }

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

        double best = 0.0; int best_k = 0;
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
                double T3 = 0.0; int kk = 0;
                for (int b = 0; b < n_bands; b++) {
                    // identifiable only with points both in and out of the window
                    if (cnt[b] < 1 || cnt[b] >= bn[b]) continue;
                    double denom = r[b] * (1.0 - r[b]);
                    if (denom <= 0.0) continue;
                    T3 += bw[b] * s[b] * s[b] / denom; kk++;
                }
                if (T3 > best) { best = T3; best_k = kk; }
            }
        }
        red[tid] = best; red_k[tid] = best_k; __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && red[tid + stride] > red[tid]) {
                red[tid] = red[tid + stride]; red_k[tid] = red_k[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) { power[f] = red[0]; kout[f] = red_k[0]; }
        __syncthreads();  // ensure red[0] is read before next f reuses it
    }
}
"""

_FAST_HELPERS = r"""
// float-float ("double-single") phase fold: t and freq are split into
// (hi, lo) float32 pairs; the exact error of the hi*hi product is recovered
// with an fmaf twoProd and the cross terms added to it. Absolute phase error
// <= ~4e-8 (float32 resolution of the result) up to 1e6 cycles, growing ~7e-15 per
// cycle beyond (the host rejects > 1e8 cycles), at full FP32 rate instead of the 1/64-rate
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
// After binning, ybin/rbin/cbin are turned in-place into inclusive cumulative
// sums (one warp per array), so every window sum is an O(1) two-element
// difference and the sweep touches only the log-spaced widths instead of every
// integer width. cbin counts points, so min_points is an exact in-transit point
// count (float32 integer sums are exact for n < 2^24, enforced in Python).
// Float32 bins; float-float phase fold; SR^2 comparison; warp-shuffle
// reduction; persistent grid-stride; __launch_bounds__ + __ldg.
// Shared memory (float32):
//   ybin[nb], rbin[nb], cbin[nb]    3*nb   (bins, then their cumsums)
//   warp_max[8]                     8
// Total: (3*nb + 8) * 4 bytes  (~5.9 KB for nb=500)
extern "C" __global__ __launch_bounds__(256, 6)
void eebls_fast_kernel(
    const float* __restrict__ t_hi,
    const float* __restrict__ t_lo,
    const float* __restrict__ wx,
    const float* __restrict__ w,
    int n, int nf,
    const double* __restrict__ freqs, int nb,
    const int* __restrict__ widths, int n_widths,
    int min_points, double* power)
{
    extern __shared__ float sh[];
    float* ybin     = sh;
    float* rbin     = sh + nb;
    float* cbin     = sh + 2*nb;
    float* warp_max = sh + 3*nb;  // 8 floats (one per warp, 256/32=8)

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float min_cnt = (float)min_points - 0.5f;  // integer counts: cnt >= min_points
    float max_cnt = (float)n - 0.5f;           // cnt <= n - 1: keep an out-of-transit point

    for (int f = blockIdx.x; f < nf; f += gridDim.x) {
        double fd = __ldg(&freqs[f]);
        float f_hi = (float)fd;
        float f_lo = (float)(fd - (double)f_hi);

        // Phase 1: clear bins
        for (int b = tid; b < 3 * nb; b += blockDim.x) sh[b] = 0.f;
        __syncthreads();

        // Phase 2: bin observations (float-float phase, float32 accumulate)
        for (int i = tid; i < n; i += blockDim.x) {
            float ph = fold_ff(__ldg(&t_hi[i]), __ldg(&t_lo[i]), f_hi, f_lo);
            int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
            atomicAdd(&ybin[ib], __ldg(&wx[i]));
            atomicAdd(&rbin[ib], __ldg(&w[i]));
            atomicAdd(&cbin[ib], 1.f);
        }
        __syncthreads();

        // Phase 3a: in-place inclusive scan of ybin, rbin and cbin (one warp
        // each; the carry is propagated serially across 32-lane chunks).
        for (int a = warp; a < 3; a += (blockDim.x >> 5)) {
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
            float cc0  = (i1 > 0) ? cbin[i1 - 1] : 0.f;
            float toty = ybin[nb - 1];
            float totr = rbin[nb - 1];
            float totc = cbin[nb - 1];
            for (int wi = 0; wi < n_widths; wi++) {
                int ww = __ldg(&widths[wi]);
                int j2 = i1 + ww - 1;
                float sy, sr, sc;
                if (j2 < nb) {
                    sc = cbin[j2] - cc0;
                    if (sc < min_cnt || sc > max_cnt) continue;
                    sy = ybin[j2] - cy0;
                    sr = rbin[j2] - cr0;
                } else {
                    j2 -= nb;
                    sc = (totc - cc0) + cbin[j2];
                    if (sc < min_cnt || sc > max_cnt) continue;
                    sy = (toty - cy0) + ybin[j2];
                    sr = (totr - cr0) + rbin[j2];
                }
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
// Same design as the single-band fast kernel: per-band ybin/rbin/cbin are
// scanned in place into inclusive cumsums (one warp per array), window sums are
// O(1) differences, and only the log-spaced widths are evaluated. As in the CPU
// cores, min_points is an exact *total* in-transit point count across bands,
// and a band contributes only if it has >= 1 point in the window (which also
// drops float32 cumsum-cancellation residue in empty windows).
// Float32 bins; float-float phase fold; band_w in registers; SR^2 comparison;
// warp-shuffle reduction; persistent grid-stride; __launch_bounds__ + __ldg.
// Compile with -DWITH_K to also write the contributing-band count of each
// frequency's best window to kout (n_bands_used); without it kout is unused.
// Shared memory: (3*n_bands*nb + 8) * 4 bytes, + 32 B with WITH_K
// (~35.2 KB for n_bands=6, nb=500)
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
    int min_points, double* power, int* kout)
{
    extern __shared__ float sh[];             // ybin[b][nb], rbin[b][nb], cbin[b][nb]
    float* warp_max = sh + 3 * n_bands * nb;  // 8 floats (one per warp)
#ifdef WITH_K
    int* warp_k = (int*)(warp_max + 8);       // 8 ints: bands used by warp_max
#endif

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int tot  = n_bands * nb;
    float min_cnt = (float)min_points - 0.5f;  // integer counts: cnt >= min_points

    // Load band weights into registers (avoids repeated global-mem reads in hot loop)
    float bw[8];
    for (int b = 0; b < n_bands; b++) bw[b] = (float)__ldg(&band_w[b]);

    for (int f = blockIdx.x; f < nf; f += gridDim.x) {
        double fd = __ldg(&freqs[f]);
        float f_hi = (float)fd;
        float f_lo = (float)(fd - (double)f_hi);

        for (int b = tid; b < 3 * tot; b += blockDim.x) sh[b] = 0.f;
        __syncthreads();

        for (int i = tid; i < n; i += blockDim.x) {
            float ph = fold_ff(__ldg(&t_hi[i]), __ldg(&t_lo[i]), f_hi, f_lo);
            int ib = (int)(nb * ph); if (ib >= nb) ib = nb - 1;
            int bb = __ldg(&band[i]);
            atomicAdd(&sh[bb * nb + ib],           __ldg(&wx[i]));
            atomicAdd(&sh[tot + bb * nb + ib],     __ldg(&w[i]));
            atomicAdd(&sh[2 * tot + bb * nb + ib], 1.f);
        }
        __syncthreads();

        // In-place inclusive scan of each of the 3*n_bands arrays
        // (one warp per array, strided over the 8 warps).
        for (int a = warp; a < 3 * n_bands; a += (blockDim.x >> 5)) {
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
#ifdef WITH_K
        int best_k = 0;
#endif
        for (int i1 = tid; i1 < nb; i1 += blockDim.x) {
            for (int wi = 0; wi < n_widths; wi++) {
                int ww = __ldg(&widths[wi]);
                int j2 = i1 + ww - 1;
                bool wrap = (j2 >= nb);        // single wrap safe: kma <= nb
                int j2w = wrap ? j2 - nb : j2;
                // One pass over bands: accumulate the total count and the
                // statistic of the bands that have points in the window; the
                // total-count gate is applied afterwards (same result as the
                // CPU's gate-then-sum, without per-band scratch arrays).
                float cnt_tot = 0.f, sr2 = 0.f; int kk = 0;
                for (int b = 0; b < n_bands; b++) {
                    const float* cy = sh + b * nb;
                    const float* cr = sh + tot + b * nb;
                    const float* cc = sh + 2 * tot + b * nb;
                    float y0 = (i1 > 0) ? cy[i1 - 1] : 0.f;
                    float r0 = (i1 > 0) ? cr[i1 - 1] : 0.f;
                    float c0 = (i1 > 0) ? cc[i1 - 1] : 0.f;
                    float sy, sr, sc;
                    if (!wrap) {
                        sy = cy[j2w] - y0;
                        sr = cr[j2w] - r0;
                        sc = cc[j2w] - c0;
                    } else {
                        sy = (cy[nb - 1] - y0) + cy[j2w];
                        sr = (cr[nb - 1] - r0) + cr[j2w];
                        sc = (cc[nb - 1] - c0) + cc[j2w];
                    }
                    cnt_tot += sc;
                    // identifiable only with points both in and out of the
                    // window (cc[nb - 1] is the band's total point count)
                    if (sc < 0.5f || sc > cc[nb - 1] - 0.5f) continue;
                    float denom = sr * (1.f - sr);
                    if (denom <= 0.f) continue;
                    sr2 += bw[b] * sy * sy / denom; kk++;
                }
                if (cnt_tot < min_cnt) continue;
                #ifdef WITH_K
                if (sr2 > best2) { best2 = sr2; best_k = kk; }
#else
                (void)kk;
                if (sr2 > best2) best2 = sr2;
#endif
            }
        }

#ifdef WITH_K
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, best2, offset);
            int ok = __shfl_down_sync(0xffffffff, best_k, offset);
            if (ov > best2) { best2 = ov; best_k = ok; }
        }
        if ((tid & 31) == 0) { warp_max[tid >> 5] = best2; warp_k[tid >> 5] = best_k; }
        __syncthreads();
        if (tid == 0) {
            int kb = warp_k[0];
            best2 = warp_max[0];
            for (int q = 1; q < 8; q++)
                if (warp_max[q] > best2) { best2 = warp_max[q]; kb = warp_k[q]; }
            power[f] = (double)best2;
            kout[f] = kb;
        }
#else
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
#endif
        __syncthreads();
    }
}
"""

_single_kernel: Any = None
_multi_kernel: Any = None
_fast_single_kernel: Any = None
_fast_multi_kernel: Any = None
_fast_multi_kernel_k: Any = None  # WITH_K variant (n_bands_used)
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
    global _fast_single_kernel, _fast_multi_kernel, _fast_multi_kernel_k
    global _max_smem
    if _single_kernel is None:
        import cupy as cp

        _max_smem = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
        _single_kernel = cp.RawKernel(_SINGLE_SRC, "eebls_kernel")
        _multi_kernel = cp.RawKernel(_MULTI_SRC, "meebls_kernel")
        _fast_single_kernel = cp.RawKernel(_FAST_BOX_SINGLE_SRC, "eebls_fast_kernel")
        _fast_multi_kernel = cp.RawKernel(_FAST_BOX_MULTI_SRC, "meebls_fast_kernel")
        _fast_multi_kernel_k = cp.RawKernel(_FAST_BOX_MULTI_SRC, "meebls_fast_kernel",
                                            options=("-DWITH_K",))
        # Opt into the device maximum for dynamic shared memory. The requested
        # dynamic size must leave room for the kernel's own static shared memory
        # (the exact kernels carry a static `red[256]` reduction buffer), so
        # subtract that from the opt-in ceiling.
        _single_kernel.max_dynamic_shared_size_bytes = (
            _max_smem - _single_kernel.attributes["shared_size_bytes"]
        )
        _multi_kernel.max_dynamic_shared_size_bytes = (
            _max_smem - _multi_kernel.attributes["shared_size_bytes"]
        )
        _fast_single_kernel.max_dynamic_shared_size_bytes = (
            _max_smem - _fast_single_kernel.attributes["shared_size_bytes"]
        )
        _fast_multi_kernel.max_dynamic_shared_size_bytes = (
            _max_smem - _fast_multi_kernel.attributes["shared_size_bytes"]
        )
        _fast_multi_kernel_k.max_dynamic_shared_size_bytes = (
            _max_smem - _fast_multi_kernel_k.attributes["shared_size_bytes"]
        )
    # fast kernels are accessed as module globals by the fast entry points


def _split_time(t: Array) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Split times into (hi, lo) float32 pairs with ``t ~= hi + lo``.

    Feeds the float-float phase fold in the fast kernels. ``t`` must already
    be median-reduced so the products ``t * freq`` stay well inside the range
    where the two-float representation holds full double-like precision.
    """
    t64 = np.asarray(t, dtype=np.float64)
    t_hi = t64.astype(np.float32)
    t_lo = (t64 - t_hi.astype(np.float64)).astype(np.float32)
    return t_hi, t_lo


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

    freqs_np, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    t, y, dy = _validate_light_curve(t, y, dy)
    _check_phase_range(t, freqs_np)
    _check_int32_sizes(t.shape[0], freqs_np.size, "eebls_gpu")
    _kernels()
    wx, w_hat, yy = _preprocess_single(y, dy)

    t_d = cp.asarray(t, dtype=cp.float64)
    wx_d = cp.asarray(wx, dtype=cp.float64)
    w_d = cp.asarray(w_hat, dtype=cp.float64)
    f_d = cp.asarray(freqs_np, dtype=cp.float64)
    nf = int(f_d.size)
    kmi, kma = _grid_params(nbins, q_min, q_max)
    shared = 3 * nbins * 8  # ybin/rbin/cbin (float64)
    smem_cap = _single_kernel.max_dynamic_shared_size_bytes
    if shared > smem_cap:
        raise ValueError(
            f"eebls_gpu needs {shared} B of dynamic shared memory "
            f"(3 * nbins={nbins} * 8) but the device allows {smem_cap} B; "
            f"reduce nbins."
        )

    power_d = cp.empty(nf, dtype=cp.float64)
    n_blocks = min(nf, 2048)
    _single_kernel((n_blocks,), (_THREADS,),
                   (t_d, wx_d, w_d, np.int32(t_d.size), np.int32(nf), f_d,
                    np.int32(nbins), np.int32(kmi), np.int32(kma),
                    np.int32(min(min_points, t_d.size + 1)), power_d),
                   shared_mem=shared)
    power_np = cp.asnumpy(power_d)
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
    rescale_errors: bool = False,
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
    rescale_errors :
        As in :func:`multiband_bls.multiband_sparse_bls`.

    Notes
    -----
    Fills :attr:`BLSResult.n_bands_used` like the CPU core. Where several windows
    tie exactly for the best statistic, the GPU reduction may keep a different one
    than the CPU's first-found window, so ``k`` can then differ.
    """
    import cupy as cp

    freqs_np, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    bands, chi2_red = _prepare_bands(bands, rescale_errors)
    _kernels()
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    _check_phase_range(t_all, freqs_np)
    _check_int32_sizes(t_all.shape[0], freqs_np.size, "multiband_eebls_gpu")
    n_bands = len(labels)
    if n_bands > _MAX_BANDS:
        raise ValueError(f"GPU multiband kernel supports up to {_MAX_BANDS} bands")

    t_d = cp.asarray(t_all, dtype=cp.float64)
    wx_d = cp.asarray(wx_all, dtype=cp.float64)
    w_d = cp.asarray(w_all, dtype=cp.float64)
    band_d = cp.asarray(band_all, dtype=cp.int32)
    bw_d = cp.asarray(w_totals, dtype=cp.float64)
    bn_d = cp.asarray(np.bincount(band_all, minlength=n_bands), dtype=cp.int32)
    f_d = cp.asarray(freqs_np, dtype=cp.float64)
    nf = int(f_d.size)
    power_d = cp.empty(nf, dtype=cp.float64)
    k_d = cp.zeros(nf, dtype=cp.int32)
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
                  (t_d, wx_d, w_d, band_d, bw_d, bn_d, np.int32(t_d.size), np.int32(n_bands),
                   np.int32(nf), f_d, np.int32(nbins), np.int32(kmi), np.int32(kma),
                   np.int32(min(min_points, t_d.size + 1)), power_d, k_d),
                  shared_mem=shared)
    power = cp.asnumpy(power_d)
    n_bands_used = cp.asnumpy(k_d).astype(np.int64)
    # Refine on the already-prepared (possibly rescaled) bands so power and the
    # refined parameters share one objective.
    return _result(power, chi2_flat, freqs_np,
                   lambda bf: multiband_eebls(bands, np.array([bf]), nbins=nbins,
                                              q_min=q_min, q_max=q_max,
                                              min_points=min_points),
                   bands=labels, band_chi2_red=chi2_red, n_bands_used=n_bands_used)


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
    times faster than float64 on consumer GPUs, with an absolute phase error of
    <= ~4e-8 for up to 1e6 cycles (median-centred ``max|t| * max(f)``); inputs beyond
    1e8 cycles are rejected.

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

    ``min_points`` is an exact global in-transit point count (same semantics
    as :func:`multiband_bls.eebls` and :func:`eebls_gpu`). Bins and sums are
    float32, so low-power (sub-peak) periodogram values are less accurate than
    the exact kernels'. ``best_t0``/``best_duration``/``best_depth`` are
    recomputed exactly on the CPU at the best frequency, while ``power`` and
    ``best_power`` stay the fast kernel's values; the two need not describe
    exactly the same box.
    Light curves with ``n >= 2**24`` points are rejected (float32 counts).
    """
    import cupy as cp

    freqs_np, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    t, y, dy = _validate_light_curve(t, y, dy)
    _check_phase_range(t, freqs_np)
    _check_int32_sizes(t.shape[0], freqs_np.size, "eebls_gpu_fast")
    if t.shape[0] >= _MAX_FAST_POINTS:
        raise ValueError(
            f"eebls_gpu_fast supports fewer than {_MAX_FAST_POINTS} points "
            f"(float32 point counts), got {t.shape[0]}; use eebls_gpu."
        )

    dlogq = _check_dlogq(dlogq, "eebls_gpu_fast")
    _kernels()  # ensure compiled
    wx, w_hat, yy = _preprocess_single(y, dy)

    # Subtract the median so large JD values (t ~ 2.4e6) keep full precision
    # in the two-float split (and the float32 bin accumulators).
    t_offset = float(np.median(t))
    _check_fast_float32(t - t_offset, freqs_np, "eebls_gpu_fast")
    t_hi, t_lo = _split_time(np.asarray(t, dtype=np.float64) - t_offset)
    thi_d = cp.asarray(t_hi, dtype=cp.float32)
    tlo_d = cp.asarray(t_lo, dtype=cp.float32)
    wx_d = cp.asarray(wx, dtype=cp.float32)
    w_d = cp.asarray(w_hat, dtype=cp.float32)
    f_d = cp.asarray(freqs_np, dtype=cp.float64)
    nf = int(f_d.size)
    n = int(thi_d.size)
    kmi, kma = _grid_params(nbins, q_min, q_max)

    widths_np = _log_widths(kmi, kma, dlogq)
    widths_d = cp.asarray(widths_np, dtype=cp.int32)
    n_widths = int(widths_np.size)
    logger.debug("eebls_gpu_fast: nb=%d kmi=%d kma=%d n_widths=%d (linear=%d)",
                 nbins, kmi, kma, n_widths, kma - kmi + 1)

    power_d = cp.empty(nf, dtype=cp.float64)
    shared = (3 * nbins + 8) * 4  # float32: ybin + rbin + cbin [nb] + warp_max[8]
    smem_cap = _fast_single_kernel.max_dynamic_shared_size_bytes
    if shared > smem_cap:
        raise ValueError(
            f"eebls_gpu_fast needs {shared} B of dynamic shared memory "
            f"((3 * nbins={nbins} + 8) * 4) but the device allows "
            f"{smem_cap} B; reduce nbins."
        )
    n_blocks = min(nf, 2048)
    _fast_single_kernel(
        (n_blocks,), (_THREADS,),
        (thi_d, tlo_d, wx_d, w_d, np.int32(n), np.int32(nf), f_d, np.int32(nbins),
         widths_d, np.int32(n_widths), np.int32(min(min_points, n + 1)), power_d),
        shared_mem=shared,
    )
    power_np = cp.asnumpy(power_d)
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
    rescale_errors: bool = False,
    compute_n_bands_used: bool = True,
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
    rescale_errors :
        As in :func:`multiband_bls.multiband_sparse_bls`.
    compute_n_bands_used :
        Fill :attr:`BLSResult.n_bands_used` (default). Pass False to leave it
        ``None`` and skip the count, which costs up to ~15% kernel time for sparse
        light curves with few bands (measured: 3 bands, n = 1k–3k points) and
        ~1–5% otherwise.

    Notes
    -----
    Introduces a small (~1–2%) loss in peak power because the exact optimal
    integer width may not fall in the log-spaced set.

    ``min_points`` is an exact global in-transit point count across all bands,
    and a band contributes only with >= 1 point in the window (same semantics
    as :func:`multiband_bls.multiband_eebls` and :func:`multiband_eebls_gpu`).
    Bins and sums are float32, so low-power (sub-peak) periodogram values are
    less accurate than the exact kernels'. The best period's parameters are
    recomputed exactly on the CPU, while ``power``/``best_power`` stay the fast
    kernel's values (the two need not describe exactly the same box). Light
    curves with ``n >= 2**24`` points in total are rejected (float32 counts).

    ``n_bands_used`` counts the bands of the window this kernel
    selects. It matches :func:`multiband_bls.multiband_eebls` when both search
    the same windows, but this function bins median-centred times (so its phase
    bins start elsewhere) and evaluates only log-spaced widths, so its best window,
    and hence ``k``, can differ at some frequencies.
    """
    import cupy as cp

    freqs_np, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    bands, chi2_red = _prepare_bands(bands, rescale_errors)

    dlogq = _check_dlogq(dlogq, "multiband_eebls_gpu_fast")
    _kernels()  # ensure compiled
    kernel = _fast_multi_kernel_k if compute_n_bands_used else _fast_multi_kernel
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    _check_phase_range(t_all, freqs_np)
    _check_int32_sizes(t_all.shape[0], freqs_np.size, "multiband_eebls_gpu_fast")
    n_bands = len(labels)
    if n_bands > _MAX_BANDS:
        raise ValueError(f"GPU multiband kernel supports up to {_MAX_BANDS} bands")
    if t_all.shape[0] >= _MAX_FAST_POINTS:
        raise ValueError(
            f"multiband_eebls_gpu_fast supports fewer than {_MAX_FAST_POINTS} points "
            f"(float32 point counts), got {t_all.shape[0]}; use multiband_eebls_gpu."
        )

    # Subtract the median so large JD values (t ~ 2.4e6) keep full precision
    # in the two-float split (and the float32 bin accumulators).
    t_offset = float(np.median(t_all))
    _check_fast_float32(t_all - t_offset, freqs_np, "multiband_eebls_gpu_fast")
    t_hi, t_lo = _split_time(t_all - t_offset)
    thi_d = cp.asarray(t_hi, dtype=cp.float32)
    tlo_d = cp.asarray(t_lo, dtype=cp.float32)
    wx_d = cp.asarray(wx_all, dtype=cp.float32)
    w_d = cp.asarray(w_all, dtype=cp.float32)
    band_d = cp.asarray(band_all, dtype=cp.int32)
    bw_d = cp.asarray(w_totals, dtype=cp.float64)
    f_d = cp.asarray(freqs_np, dtype=cp.float64)
    nf = int(f_d.size)
    n = int(thi_d.size)
    kmi, kma = _grid_params(nbins, q_min, q_max)

    widths_np = _log_widths(kmi, kma, dlogq)
    widths_d = cp.asarray(widths_np, dtype=cp.int32)
    n_widths = int(widths_np.size)
    logger.debug("multiband_eebls_gpu_fast: nb=%d kmi=%d kma=%d n_widths=%d (linear=%d)",
                 nbins, kmi, kma, n_widths, kma - kmi + 1)

    power_d = cp.empty(nf, dtype=cp.float64)
    # k output; a 1-element placeholder when the kernel is compiled without WITH_K
    k_d = cp.zeros(nf if compute_n_bands_used else 1, dtype=cp.int32)
    n_blocks = min(nf, 2048)
    # float32: ybin + rbin + cbin (per band) + warp_max[8] (+ int warp_k[8])
    n_extra = 16 if compute_n_bands_used else 8
    shared = (3 * n_bands * nbins + n_extra) * 4
    smem_cap = kernel.max_dynamic_shared_size_bytes
    if shared > smem_cap:
        raise ValueError(
            f"multiband_eebls_gpu_fast needs {shared} B of dynamic shared memory "
            f"((3 * n_bands={n_bands} * nbins={nbins} + {n_extra}) * 4) but the "
            f"device allows {smem_cap} B; reduce nbins or n_bands."
        )
    kernel(
        (n_blocks,), (_THREADS,),
        (thi_d, tlo_d, wx_d, w_d, band_d, bw_d, np.int32(n), np.int32(n_bands),
         np.int32(nf), f_d, np.int32(nbins), widths_d, np.int32(n_widths),
         np.int32(min(min_points, n + 1)), power_d, k_d),
        shared_mem=shared,
    )
    power = cp.asnumpy(power_d)
    n_bands_used = cp.asnumpy(k_d).astype(np.int64) if compute_n_bands_used else None
    # Refine on the already-prepared (possibly rescaled) bands so power and the
    # refined parameters share one objective.
    return _result(power, chi2_flat, freqs_np,
                   lambda bf: multiband_eebls(bands, np.array([bf]), nbins=nbins,
                                              q_min=q_min, q_max=q_max,
                                              min_points=min_points),
                   bands=labels, band_chi2_red=chi2_red, n_bands_used=n_bands_used)
