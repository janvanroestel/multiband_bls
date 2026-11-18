
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
    int min_points, double* power, float sg)
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
                if (sg * sy < 0.f) continue;  // sign cut (sg = 0: both signs)
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
