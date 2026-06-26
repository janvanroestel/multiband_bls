# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled single-band Sparse BLS core (Panahi & Zucker 2021).

Mirrors :func:`multiband_bls.reference.sparse_bls_reference` exactly, with the
double in-transit loop in nogil C.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport floor

cnp.import_array()


cdef void _single_period(const double* phi, const double* wx, const double* w,
                         Py_ssize_t n, double period, double q_max,
                         int min_points, double* out) noexcept nogil:
    """Fill ``out = [SR, t0, duration, depth]`` for one trial period.

    Arrays are sorted ascending in phase. The in-transit block is a contiguous
    (wrap-around) run grown from ``i1``.
    """
    cdef Py_ssize_t i1, k, i2, i1m, i2p
    cdef int cnt
    cdef double s, r, frac, phi_ing, phi_eg, denom, sr
    cdef double p_i1, p_i1m, p_i2, p_i2p, phi_mid
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0, best_depth = 0.0

    for i1 in range(n):
        i1m = i1 - 1
        if i1m < 0:
            i1m += n
        p_i1 = phi[i1]
        p_i1m = phi[i1m]
        if p_i1m < p_i1:
            phi_ing = 0.5 * (p_i1m + p_i1)
        else:
            phi_ing = 0.5 * (p_i1m + p_i1) - 0.5

        s = 0.0
        r = 0.0
        cnt = 0
        for k in range(n - 1):
            i2 = i1 + k
            if i2 >= n:
                i2 -= n
            s += wx[i2]
            r += w[i2]
            cnt += 1

            i2p = i2 + 1
            if i2p >= n:
                i2p -= n
            p_i2 = phi[i2]
            p_i2p = phi[i2p]
            if p_i2 < p_i2p:
                phi_eg = 0.5 * (p_i2 + p_i2p)
            else:
                phi_eg = 0.5 * (p_i2 + p_i2p) + 0.5

            frac = phi_eg - phi_ing
            frac -= floor(frac)
            if frac >= q_max:
                break
            if cnt < min_points:
                continue
            denom = r * (1.0 - r)
            if denom <= 0.0:
                continue
            sr = s * s / denom
            if sr > best_sr:
                best_sr = sr
                best_depth = s / denom
                best_dur = frac * period
                phi_mid = phi_ing + 0.5 * frac
                phi_mid -= floor(phi_mid)
                best_t0 = phi_mid * period

    out[0] = best_sr
    out[1] = best_t0
    out[2] = best_dur
    out[3] = best_depth


def sbls_grid(double[::1] t, double[::1] wx, double[::1] w_hat,
              double[::1] freqs, double q_max, int min_points):
    """Scan a frequency grid; return ``(power, best_freq, t0, dur, depth, sr)``.

    ``wx = w_hat * x_tilde`` and ``w_hat`` are the normalised-weight products
    from :func:`multiband_bls.reference.preprocess`.
    """
    cdef Py_ssize_t nf = freqs.shape[0]
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t fi, i, j

    cdef cnp.ndarray[double, ndim=1] power = np.zeros(nf, dtype=np.float64)
    cdef double[::1] pw = power

    cdef cnp.ndarray[double, ndim=1] phi_np = np.empty(n, dtype=np.float64)
    cdef double[::1] phi = phi_np
    cdef double[::1] phi_s = np.empty(n, dtype=np.float64)
    cdef double[::1] wx_s = np.empty(n, dtype=np.float64)
    cdef double[::1] w_s = np.empty(n, dtype=np.float64)
    cdef cnp.intp_t[::1] order

    cdef double out[4]
    cdef double period, f, ph
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0, best_depth = 0.0
    cdef double best_freq = freqs[0]

    for fi in range(nf):
        f = freqs[fi]
        period = 1.0 / f
        for i in range(n):
            ph = t[i] / period
            phi[i] = ph - floor(ph)
        order = np.argsort(phi_np)
        for i in range(n):
            j = order[i]
            phi_s[i] = phi[j]
            wx_s[i] = wx[j]
            w_s[i] = w_hat[j]
        with nogil:
            _single_period(&phi_s[0], &wx_s[0], &w_s[0], n, period,
                           q_max, min_points, out)
        pw[fi] = out[0]
        if out[0] > best_sr:
            best_sr = out[0]
            best_t0 = out[1]
            best_dur = out[2]
            best_depth = out[3]
            best_freq = f

    return power, best_freq, best_t0, best_dur, best_depth, best_sr
