
// Exact single-band binned BLS: full linear width sweep in double precision.
// Persistent grid-stride over frequencies (blocks capped in Python) so global
// inputs stay hot in L2 across trials; __restrict__ + __ldg on all global reads.
extern "C" __global__ __launch_bounds__(256)
void eebls_kernel(const double* __restrict__ t, const double* __restrict__ wx,
                  const double* __restrict__ w, int n, int nf,
                  const double* __restrict__ freqs, int nb, int kmi, int kma,
                  int min_bin_points, double* power, double sg) {
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
                if (sg * s < 0.0) continue;  // sign cut (sg = 0: both signs)
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
