# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled classic binned BLS with edge-effect handling (``eeBLS``).

This is the original Box-fitting Least Squares of Kovacs, Zucker & Mazeh (2002,
the FORTRAN ``eebls.f`` routine): the phase-folded light curve is binned onto
``nb`` phase bins, and the in-transit box is searched over *bins*. The
"edge effect" (ee) is the wrap-around of the bin window past phase 1, so transits
straddling the phase-0 boundary are not missed.

Unlike the unbinned Sparse BLS, the per-period cost is ``O(N)`` (one binning pass)
plus a constant ``O(nb * kma)`` bin loop, i.e. **linear** in ``N`` for a fixed
bin count -- so for large ``N`` it eventually overtakes the ``O(N^2)`` SBLS, the
crossover discussed in Panahi & Zucker (2021).

To keep the statistic directly comparable to the SBLS cores, the same normalised
weights are used: ``s = sum_in w_hat*x_tilde``, ``r = sum_in w_hat``,
``SR = sqrt(s^2 / (r(1-r)))``.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport floor

cnp.import_array()


cdef void _eebls_period(const double* t, const double* wx, const double* w,
                        Py_ssize_t n, double period, int nb,
                        int kmi, int kma, int min_bin_points, double sign,
                        double* ybin, double* rbin, cnp.int64_t* cbin,
                        double* out) noexcept nogil:
    """Fill ``out = [SR, t0, duration, depth]`` for one trial period (box model)."""
    cdef Py_ssize_t i
    cdef int j, jj, ib, i1, width
    cdef cnp.int64_t cnt  # point counts: 64-bit, like the count buffers
    cdef double ph, s, r, denom, sr, phi_mid
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0, best_depth = 0.0

    # --- binning ----------------------------------------------------------------
    for j in range(nb):
        ybin[j] = 0.0
        rbin[j] = 0.0
        cbin[j] = 0

    for i in range(n):
        ph = t[i] / period
        ph -= floor(ph)
        ib = <int>(nb * ph)
        if ib >= nb:
            ib = nb - 1
        ybin[ib] += wx[i]
        rbin[ib] += w[i]
        cbin[ib] += 1

    # --- window search ----------------------------------------------------------
    for i1 in range(nb):
        s = 0.0
        r = 0.0
        cnt = 0
        width = 0
        for j in range(i1, i1 + kma):
            jj = j
            if jj >= nb:
                jj -= nb
            s += ybin[jj]
            r += rbin[jj]
            cnt += cbin[jj]
            width += 1
            if width < kmi:
                continue
            if cnt < min_bin_points:
                continue
            if cnt >= n:
                continue  # no out-of-transit point: depth not identifiable
            if sign * s < 0.0:
                continue  # wrong-signed window (sign = 0: keep both)
            denom = r * (1.0 - r)
            if denom <= 0.0:
                continue
            sr = s * s / denom
            if sr > best_sr:
                best_sr = sr
                best_depth = s / denom
                best_dur = (<double>width) / nb * period
                phi_mid = (i1 + 0.5 * width) / nb
                phi_mid -= floor(phi_mid)
                best_t0 = phi_mid * period

    out[0] = best_sr
    out[1] = best_t0
    out[2] = best_dur
    out[3] = best_depth


def eebls_grid(double[::1] t, double[::1] wx, double[::1] w_hat,
               double[::1] freqs, int nb, double qmin, double qmax,
               int min_bin_points, double sign=0.0):
    """Scan a frequency grid; return ``(power, best_freq, t0, dur, depth, sr)``.

    ``nb`` is the number of phase bins; ``qmin``/``qmax`` bound the fractional
    transit duration (converted to a min/max number of bins).
    """
    cdef Py_ssize_t nf = freqs.shape[0]
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t fi

    cdef int kmi = <int>(qmin * nb)
    if kmi < 1:
        kmi = 1
    cdef int kma = <int>(qmax * nb) + 1
    if kma > nb:
        kma = nb

    cdef cnp.ndarray[double, ndim=1] power = np.zeros(nf, dtype=np.float64)
    cdef double[::1] pw = power
    cdef double[::1] ybin = np.empty(nb, dtype=np.float64)
    cdef double[::1] rbin = np.empty(nb, dtype=np.float64)
    cdef cnp.int64_t[::1] cbin = np.empty(nb, dtype=np.int64)

    cdef double out[4]
    cdef double period, f
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0, best_depth = 0.0
    cdef double best_freq = freqs[0]

    for fi in range(nf):
        f = freqs[fi]
        period = 1.0 / f
        with nogil:
            _eebls_period(&t[0], &wx[0], &w_hat[0], n, period, nb,
                          kmi, kma, min_bin_points, sign,
                          &ybin[0], &rbin[0], &cbin[0], out)
        pw[fi] = out[0]
        if out[0] > best_sr:
            best_sr = out[0]
            best_t0 = out[1]
            best_dur = out[2]
            best_depth = out[3]
            best_freq = f

    return power, best_freq, best_t0, best_dur, best_depth, best_sr
