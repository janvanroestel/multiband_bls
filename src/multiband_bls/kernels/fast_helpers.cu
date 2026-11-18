
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
