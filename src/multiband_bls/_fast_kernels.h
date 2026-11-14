/* Per-frequency kernels of the CPU prefix-sum BLS (see _fast.pyx).
 *
 * Plain C so the compiler can vectorise the loops, which Cython's generated code
 * does not allow. On x86-64 Linux with GCC/Clang the window evaluators are
 * multiversioned (Haswell: AVX2+FMA / baseline) and marked noinline so the
 * dispatcher is actually used; wheels stay portable while AVX2 machines get
 * 4-wide double lanes.
 *
 * Bin arrays hold inclusive-prefix sums with a leading zero: c[k] = sum of bins
 * [0, k), length nb + 1. A circular window of width w starting at bin i is
 *
 *     c[i + w] - c[i]                     if i + w <= nb   (no wrap)
 *     (c[nb] - c[i]) + c[i + w - nb]      otherwise        (wraps once; w <= nb)
 *
 * Both are evaluated as c[i + off] + add - c[i] with (off, add) = (w, 0) for
 * the non-wrapping starts i in [0, nb - w] and (w - nb, c[nb]) for the wrapping
 * starts i in [nb - w + 1, nb).
 *
 * The sweep first lists the start bins whose (pooled) point count passes the
 * min_points gate -- cheap, and most of them in sparse data do not -- and only then
 * forms the chi^2 statistic, on the survivors.
 */
#ifndef MULTIBAND_BLS_FAST_KERNELS_H
#define MULTIBAND_BLS_FAST_KERNELS_H

#include <string.h>

#if defined(_MSC_VER) && !defined(restrict)
#define restrict __restrict
#endif

#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__) && !defined(MBLS_NO_CLONES)
#define MBLS_CLONES __attribute__((noinline, target_clones("arch=haswell", "default")))
#else
#define MBLS_CLONES
#endif

/* Phase bins ib[i] = floor(nb * frac(t[i] * f)), clamped to nb - 1, for all points.
 * Truncation to int32 is exact because |t * f| < 2**30 (checked by the caller); the
 * loop is a separate pass so it vectorises (4 doubles per AVX2 instruction). frac()
 * is ph - trunc(ph) shifted into [0, 1]; the arithmetic matches ph - floor(ph). */
MBLS_CLONES
static void mbls_fold(const double *restrict t, long n, double f, int nb,
                      int *restrict ib)
{
    for (long i = 0; i < n; i++) {
        double ph = t[i] * f;
        ph -= (double)(int)ph;            /* trunc toward zero */
        ph += (ph < 0.0);                 /* wrap negatives into [0, 1] */
        const int b = (int)(nb * ph);
        ib[i] = b >= nb ? nb - 1 : b;
    }
}

/* In-place inclusive cumsum of c[1..nb] given c[0] = 0, for three arrays at once
 * (three independent dependency chains). */
static inline void mbls_cumsum3(double *restrict y, double *restrict r,
                                double *restrict c, int nb)
{
    double sy = 0.0, sr = 0.0, sc = 0.0;
    for (int k = 1; k <= nb; k++) {
        sy += y[k]; sr += r[k]; sc += c[k];
        y[k] = sy; r[k] = sr; c[k] = sc;
    }
}

/* Fold + bin + prefix-sum one (single) band: fills cy/cr/cc[0..nb]. */
static void mbls_prepare_single(const double *restrict t, const double *restrict wx,
                                const double *restrict w, long n, double f, int nb,
                                double *restrict cy, double *restrict cr,
                                double *restrict cc, int *restrict ibuf)
{
    memset(cy, 0, (nb + 1) * sizeof(double));
    memset(cr, 0, (nb + 1) * sizeof(double));
    memset(cc, 0, (nb + 1) * sizeof(double));
    mbls_fold(t, n, f, nb, ibuf);
    for (long i = 0; i < n; i++) {
        const int ib = ibuf[i] + 1;
        cy[ib] += wx[i];
        cr[ib] += w[i];
        cc[ib] += 1.0;
    }
    mbls_cumsum3(cy, cr, cc, nb);
}

/* Fold + bin + prefix-sum all bands (band-major: array b at offset b * (nb + 1))
 * and the pooled count prefix ct[0..nb]. */
static void mbls_prepare_multi(const double *restrict t, const double *restrict wx,
                               const double *restrict w, const int *restrict band,
                               long n, int n_bands, double f, int nb,
                               double *restrict cy, double *restrict cr,
                               double *restrict cc, double *restrict ct,
                               int *restrict ibuf)
{
    const long L = nb + 1;
    memset(cy, 0, n_bands * L * sizeof(double));
    memset(cr, 0, n_bands * L * sizeof(double));
    memset(cc, 0, n_bands * L * sizeof(double));
    mbls_fold(t, n, f, nb, ibuf);
    for (long i = 0; i < n; i++) {
        const long ib = band[i] * L + ibuf[i] + 1;
        cy[ib] += wx[i];
        cr[ib] += w[i];
        cc[ib] += 1.0;
    }
    memset(ct, 0, L * sizeof(double));
    for (int b = 0; b < n_bands; b++) {
        mbls_cumsum3(cy + b * L, cr + b * L, cc + b * L, nb);
        const double *restrict cb = cc + b * L;
        for (long k = 0; k < L; k++) ct[k] += cb[k];
    }
}

/* ---- single band ------------------------------------------------------- */

/* Start bins i in [i0, i1) whose window count c[i + off] + add - c[i] lies in
 * [mp, nmax]; returns how many were stored in idx. */
MBLS_CLONES
static int mbls_single_gate(const double *restrict cc, int i0, int i1, int off,
                            double add, double mp, double nmax, int *restrict idx)
{
    int m = 0;
    for (int i = i0; i < i1; i++) {
        const double sc = cc[i + off] + add - cc[i];
        idx[m] = i;
        m += (sc >= mp) & (sc <= nmax);
    }
    return m;
}

/* Max of sy^2 / (sr (1 - sr)) over the listed starts (sr (1 - sr) > 0 only). */
MBLS_CLONES
static double mbls_single_eval(const double *restrict cy, const double *restrict cr,
                               const int *restrict idx, int m, int off,
                               double addy, double addr)
{
    double best = 0.0;
    for (int j = 0; j < m; j++) {
        const int i = idx[j];
        const double sy = cy[i + off] + addy - cy[i];
        const double sr = cr[i + off] + addr - cr[i];
        const double denom = sr * (1.0 - sr);
        const double v = sy * sy / (denom > 0.0 ? denom : 1.0);
        const double g = denom > 0.0 ? v : 0.0;
        best = g > best ? g : best;
    }
    return best;
}

/* Whole sweep of one frequency's prepared bins over all widths. */
static double mbls_sweep_single(const double *cy, const double *cr, const double *cc,
                                int nb, const int *widths, int nw, double mp,
                                double nmax, int *idx)
{
    double best = 0.0;
    for (int wi = 0; wi < nw; wi++) {
        const int ww = widths[wi];
        for (int part = 0; part < 2; part++) {
            const int i0 = part ? nb - ww + 1 : 0, i1 = part ? nb : nb - ww + 1;
            if (i1 <= i0) continue;
            const int off = part ? ww - nb : ww;
            const int m = mbls_single_gate(cc, i0, i1, off, part ? cc[nb] : 0.0, mp,
                                           nmax, idx);
            if (m == 0) continue;
            const double v = mbls_single_eval(cy, cr, idx, m, off, part ? cy[nb] : 0.0,
                                              part ? cr[nb] : 0.0);
            best = v > best ? v : best;
        }
    }
    return best;
}

/* ---- multiband --------------------------------------------------------- */

/* Start bins i in [i0, i1) whose pooled (all-band) window count is >= mp. */
MBLS_CLONES
static int mbls_multi_gate(const double *restrict ct, int i0, int i1, int off,
                           double add, double mp, int *restrict idx)
{
    int m = 0;
    for (int i = i0; i < i1; i++) {
        const double sc = ct[i + off] + add - ct[i];
        idx[m] = i;
        m += (sc >= mp);
    }
    return m;
}

/* max over listed starts of sum_b W_b sy^2 / (sr (1 - sr)), a band counting only
 * with 1 <= count < band_n and sr (1 - sr) > 0. acc is scratch of length >= m. */
MBLS_CLONES
static double mbls_multi_eval(const double *restrict cy, const double *restrict cr,
                              const double *restrict cc, int n_bands, int nb,
                              const int *restrict idx, int m, int off, int wrap,
                              const double *restrict band_w,
                              const double *restrict band_n, double *restrict acc)
{
    const long L = nb + 1;
    for (int j = 0; j < m; j++) acc[j] = 0.0;
    for (int b = 0; b < n_bands; b++) {
        const double *y = cy + b * L, *r = cr + b * L, *c = cc + b * L;
        const double bw = band_w[b], bn = band_n[b];
        const double ay = wrap ? y[nb] : 0.0, ar = wrap ? r[nb] : 0.0,
                     ac = wrap ? c[nb] : 0.0;
        for (int j = 0; j < m; j++) {
            const int i = idx[j];
            const double sc = c[i + off] + ac - c[i];
            const double sy = y[i + off] + ay - y[i];
            const double sr = r[i + off] + ar - r[i];
            const double denom = sr * (1.0 - sr);
            const double v = bw * sy * sy / (denom > 0.0 ? denom : 1.0);
            acc[j] += ((sc >= 1.0) & (sc < bn) & (denom > 0.0)) ? v : 0.0;
        }
    }
    double best = 0.0;
    for (int j = 0; j < m; j++) best = acc[j] > best ? acc[j] : best;
    return best;
}

static double mbls_sweep_multi(const double *cy, const double *cr, const double *cc,
                               const double *ct, int n_bands, int nb, const int *widths,
                               int nw, double mp, const double *band_w,
                               const double *band_n, int *idx, double *acc)
{
    double best = 0.0;
    for (int wi = 0; wi < nw; wi++) {
        const int ww = widths[wi];
        for (int part = 0; part < 2; part++) {
            const int i0 = part ? nb - ww + 1 : 0, i1 = part ? nb : nb - ww + 1;
            if (i1 <= i0) continue;
            const int off = part ? ww - nb : ww;
            const int m = mbls_multi_gate(ct, i0, i1, off, part ? ct[nb] : 0.0, mp, idx);
            if (m == 0) continue;
            const double v = mbls_multi_eval(cy, cr, cc, n_bands, nb, idx, m, off, part,
                                             band_w, band_n, acc);
            best = v > best ? v : best;
        }
    }
    return best;
}

#endif
