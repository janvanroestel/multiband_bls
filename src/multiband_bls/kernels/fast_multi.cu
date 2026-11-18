
// Prefix-sum multiband box BLS with log-spaced widths. Handles 1-8 bands.
// Same design as the single-band fast kernel: per-band ybin/rbin/cbin are
// scanned in place into inclusive cumsums (one warp per array), window sums are
// O(1) differences, and only the log-spaced widths are evaluated. As in the CPU
// cores, min_points is an exact *total* in-transit point count across bands,
// and a band contributes only if it has >= 1 point in the window (which also
// drops float32 cumsum-cancellation residue in empty windows).
// Float32 bins; float-float phase fold; band_w in registers; SR^2 comparison;
// warp-shuffle reduction; persistent grid-stride; __launch_bounds__ + __ldg.
// Compile with -DNB=<n_bands> to specialise on the band count (band loops unroll and
// the per-band arrays live in registers; without it n_bands is a runtime argument).
// Compile with -DMINB=<k> to set the minimum blocks per SM for __launch_bounds__.
// Compile with -DWITH_K to also write the contributing-band count of each
// frequency's best window to kout (n_bands_used); without it kout is unused.
// Shared memory: (3*n_bands*nb + 8) * 4 bytes, + 32 B with WITH_K
// (~35.2 KB for n_bands=6, nb=500)
// One (start i1, width wi) window of the multiband sweep: updates best2 (and
// best_k with WITH_K). sh holds ybin/rbin/cbin (inclusive cumsums) per band.
__device__ __forceinline__ void eval_window(
    const float* __restrict__ sh, const int* __restrict__ widths, const float* bw,
    int i1, int wi, int nb, int tot, int n_bands, float min_cnt, float sg,
    float& best2
#ifdef WITH_K
    , int& best_k
#endif
    )
{
    const int ww = __ldg(&widths[wi]);
    const int j2 = i1 + ww - 1;
    const bool wrap = (j2 >= nb);              // single wrap safe: kma <= nb
    const int j2w = wrap ? j2 - nb : j2;
    // One pass over bands: accumulate the total count and the statistic of the
    // bands that have points in the window; the total-count gate is applied
    // afterwards (same result as the CPU's gate-then-sum, without per-band
    // scratch arrays).
    float cnt_tot = 0.f, sr2 = 0.f; int kk = 0;
    #pragma unroll
    for (int b = 0; b < n_bands; b++) {
        const float* cy = sh + b * nb;
        const float* cr = sh + tot + b * nb;
        const float* cc = sh + 2 * tot + b * nb;
        const float y0 = (i1 > 0) ? cy[i1 - 1] : 0.f;
        const float r0 = (i1 > 0) ? cr[i1 - 1] : 0.f;
        const float c0 = (i1 > 0) ? cc[i1 - 1] : 0.f;
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
        // identifiable only with points both in and out of the window
        // (cc[nb - 1] is the band's total point count)
        if (sc < 0.5f || sc > cc[nb - 1] - 0.5f) continue;
        const float denom = sr * (1.f - sr);
        if (denom <= 0.f || sg * sy < 0.f) continue;  // per-band sign cut
        sr2 += bw[b] * sy * sy / denom; kk++;
    }
    if (cnt_tot < min_cnt) return;
#ifdef WITH_K
    if (sr2 > best2) { best2 = sr2; best_k = kk; }
#else
    (void)kk;
    if (sr2 > best2) best2 = sr2;
#endif
}

// Below this many bins the sweep distributes (start, width) pairs over threads
// (measured crossover ~200-256 bins on an RTX 3050 Ti; flat wins 1.05-1.5x below).
#define FLAT_NB 224
#ifdef WITH_K
#define EVAL(i1_, wi_) eval_window(sh, widths, bw, i1_, wi_, nb, tot, n_bands, \
                                   min_cnt, sg, best2, best_k)
#else
#define EVAL(i1_, wi_) eval_window(sh, widths, bw, i1_, wi_, nb, tot, n_bands, \
                                   min_cnt, sg, best2)
#endif
// Minimum resident blocks per SM promised to the register allocator. The host picks it
// from what shared memory allows (-DMINB=...): demanding 4 blocks when only 2 fit
// caps registers for nothing (measured ~11% slower at 6-8 bands, nb = 500).
#ifndef MINB
#define MINB 4
#endif
extern "C" __global__ __launch_bounds__(256, MINB)
void meebls_fast_kernel(
    const float* __restrict__ t_hi,
    const float* __restrict__ t_lo,
    const float* __restrict__ wx,
    const float* __restrict__ w,
    const int* __restrict__ band,
    const double* __restrict__ band_w,
    int n, int n_bands_in, int nf,
    const double* __restrict__ freqs, int nb,
    const int* __restrict__ widths, int n_widths,
    int min_points, double* power, int* kout, float sg)
{
#ifdef NB
    const int n_bands = NB;                   // compile-time: unrolled loops, bw[] in regs
#else
    const int n_bands = n_bands_in;
#endif
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
    #pragma unroll
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
        if (nb < FLAT_NB) {
            // Few starts: spread the (start, width) pairs over all threads
            // (p = wi * nb + i1, consecutive threads -> consecutive starts) so
            // nb < blockDim.x does not leave threads idle through the sweep.
            // Ties between equal statistics are broken by thread order, as
            // below; n_bands_used was checked identical on test grids.
            const int n_pairs = nb * n_widths;
            for (int p = tid; p < n_pairs; p += blockDim.x) {
                const int wi = p / nb;
                EVAL(p - wi * nb, wi);
            }
        } else {
            // Enough starts to keep the block busy: loop widths per start, which
            // reloads the per-start prefix values once rather than per width.
            for (int i1 = tid; i1 < nb; i1 += blockDim.x)
                for (int wi = 0; wi < n_widths; wi++) EVAL(i1, wi);
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
