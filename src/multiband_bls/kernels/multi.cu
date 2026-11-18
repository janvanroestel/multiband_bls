
// Exact multiband binned BLS in double precision. Persistent grid-stride over
// frequencies (blocks capped in Python) with __restrict__ + __ldg on global reads;
// band weights cached in registers. Compile with -DNB=<n_bands> to specialise on
// the band count (unrolled band loops; per-band arrays in registers).
extern "C" __global__ __launch_bounds__(256)
void meebls_kernel(const double* __restrict__ t, const double* __restrict__ wx,
                   const double* __restrict__ w,
                   const int* __restrict__ band, const double* __restrict__ band_w,
                   const int* __restrict__ band_n,
                   int n, int n_bands_in, int nf,
                   const double* __restrict__ freqs, int nb, int kmi, int kma,
                   int min_points, double* power, int* kout, double sg) {
#ifdef NB
    const int n_bands = NB;                  // compile-time: band loops unroll and the
#else                                        // per-band arrays stay in registers
    const int n_bands = n_bands_in;
#endif
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
    #pragma unroll
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
            #pragma unroll
            for (int b = 0; b < n_bands; b++) { s[b] = 0; r[b] = 0; cnt[b] = 0; }
            int total = 0;
            for (int ww = 1; ww <= kma; ++ww) {
                int jj = i1 + ww - 1; if (jj >= nb) jj -= nb;  // single wrap safe: kma <= nb
                #pragma unroll
                for (int b = 0; b < n_bands; b++) {
                    s[b] += ybin[b * nb + jj]; r[b] += rbin[b * nb + jj];
                    cnt[b] += (int)cbin[b * nb + jj]; total += (int)cbin[b * nb + jj];
                }
                if (ww < kmi) continue;
                if (total < min_points) continue;
                double T3 = 0.0; int kk = 0;
                #pragma unroll
                for (int b = 0; b < n_bands; b++) {
                    // identifiable only with points both in and out of the window
                    if (cnt[b] < 1 || cnt[b] >= bn[b]) continue;
                    double denom = r[b] * (1.0 - r[b]);
                    if (denom <= 0.0 || sg * s[b] < 0.0) continue;  // per-band sign cut
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
