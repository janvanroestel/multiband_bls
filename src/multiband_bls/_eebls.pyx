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

Trapezoid extension
-------------------
The optional ``f_arr`` parameter searches over transit shapes parameterised by
``tau`` in [0, 1]:

- ``tau = 0`` → box (fast incremental path, identical to the original kernel).
- ``tau = 1`` → triangle (ingress + egress fill the entire window).
- ``0 < tau < 1`` → trapezoid with ingress/egress fraction ``tau``.

Bin ``k`` in a window of width ``W`` carries trapezoid weight::

    phi = (k + 0.5) / W
    f_k = phi / (tau/2)         if phi < tau/2           (ingress)
        = (1 - phi) / (tau/2)   if phi > 1 - tau/2       (egress)
        = 1.0                   otherwise                 (flat bottom)

The modified statistic uses three accumulators A, B, C over the window::

    A = sum_k f_k * ybin[k]
    B = sum_k f_k^2 * rbin[k]
    C = sum_k f_k * rbin[k]
    SR = |A| / sqrt(B - C^2)

This reduces exactly to sqrt(s^2 / (r(1-r))) for tau=0 (box).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport floor, ceil
from libc.stdlib cimport malloc, free

cnp.import_array()


cdef inline double _trap_w(double phi, double tau) noexcept nogil:
    """Trapezoid weight for normalised bin-centre phi in (0,1) and shape tau."""
    cdef double half
    if tau <= 0.0:
        return 1.0
    half = 0.5 * tau
    if phi < half:
        return phi / half
    if phi > 1.0 - half:
        return (1.0 - phi) / half
    return 1.0


cdef void _eebls_period_trap(const double* t, const double* wx, const double* w,
                             Py_ssize_t n, double period, int nb,
                             int kmi, int kma, int min_bin_points, double tau,
                             double* ybin, double* rbin, long* cbin,
                             double* out) noexcept nogil:
    """Fill ``out = [SR, t0, duration, depth]`` for one trial period and shape.

    For ``tau == 0`` uses the original fast incremental accumulation (box).
    For ``tau > 0`` uses a non-incremental triple loop with trapezoid weights.
    """
    cdef Py_ssize_t i
    cdef int j, jj, ib, i1, width, cnt, k
    cdef double ph, s, r, denom, sr, phi_mid, fk, A, B, C
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0, best_depth = 0.0
    # prefix-sum variables (used only for tau > 0 path)
    cdef double* S0y = NULL
    cdef double* S1y = NULL
    cdef double* S0r = NULL
    cdef double* S1r = NULL
    cdef double* S2r = NULL
    cdef long*   S0c = NULL
    cdef int size2, k_in, k_eg, je, jf, jw
    cdef double m, c2, c2sq, i1d, wd, yv, rv
    cdef double A_in, A_flat, A_eg, C_in, C_flat, C_eg, B_in, B_flat, B_eg
    cdef double DS0y_in, DS1y_in, DS0y_flat, DS0y_eg, DS1y_eg
    cdef double DS0r_in, DS1r_in, DS2r_in, DS0r_flat, DS0r_eg, DS1r_eg, DS2r_eg
    cdef double ref_in, ref_eg
    cdef long cnt_l

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
    if tau <= 0.0:
        # Fast path: box model — incremental accumulation (identical to original).
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
    else:
        # Trapezoid path: prefix-sum O(nb * kma) algorithm.
        # Precompute 6 prefix arrays on the doubled circular bin array so each
        # (i1, width) evaluation of (A, B, C) costs O(1) instead of O(width).
        size2 = 2 * nb + 2
        S0y = <double*>malloc(size2 * sizeof(double))
        S1y = <double*>malloc(size2 * sizeof(double))
        S0r = <double*>malloc(size2 * sizeof(double))
        S1r = <double*>malloc(size2 * sizeof(double))
        S2r = <double*>malloc(size2 * sizeof(double))
        S0c = <long*>malloc(size2 * sizeof(long))
        if (S0y == NULL or S1y == NULL or S0r == NULL or
                S1r == NULL or S2r == NULL or S0c == NULL):
            free(S0y); free(S1y); free(S0r)
            free(S1r); free(S2r); free(S0c)
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0; out[3] = 0.0
            return
        S0y[0] = 0.0; S1y[0] = 0.0; S0r[0] = 0.0
        S1r[0] = 0.0; S2r[0] = 0.0; S0c[0] = 0
        for j in range(2 * nb):
            jj = j % nb
            yv = ybin[jj]
            rv = rbin[jj]
            wd = <double>j       # reuse wd as loop variable here (j as double)
            S0y[j+1] = S0y[j] + yv
            S1y[j+1] = S1y[j] + wd * yv
            S0r[j+1] = S0r[j] + rv
            S1r[j+1] = S1r[j] + wd * rv
            S2r[j+1] = S2r[j] + wd * wd * rv
            S0c[j+1] = S0c[j] + cbin[jj]

        for i1 in range(nb):
            i1d = <double>i1
            for width in range(kmi, kma + 1):
                wd = <double>width
                m = tau * wd * 0.5
                k_in = <int>ceil(m - 0.5) if m > 0.5 else 0
                if k_in > width:
                    k_in = width
                k_eg = width - k_in

                je = i1 + k_in    # ingress end / flat start
                jf = i1 + k_eg   # flat end / egress start
                jw = i1 + width   # window end

                cnt_l = S0c[jw] - S0c[i1]
                if cnt_l < min_bin_points:
                    continue

                DS0y_in  = S0y[je] - S0y[i1]
                DS1y_in  = S1y[je] - S1y[i1]
                DS0y_flat = S0y[jf] - S0y[je]
                DS0y_eg  = S0y[jw] - S0y[jf]
                DS1y_eg  = S1y[jw] - S1y[jf]

                DS0r_in  = S0r[je] - S0r[i1]
                DS1r_in  = S1r[je] - S1r[i1]
                DS2r_in  = S2r[je] - S2r[i1]
                DS0r_flat = S0r[jf] - S0r[je]
                DS0r_eg  = S0r[jw] - S0r[jf]
                DS1r_eg  = S1r[jw] - S1r[jf]
                DS2r_eg  = S2r[jw] - S2r[jf]

                c2 = 2.0 / (tau * wd)
                c2sq = c2 * c2

                A_in   = c2 * (DS1y_in + (0.5 - i1d) * DS0y_in)
                A_flat = DS0y_flat
                A_eg   = c2 * ((wd - 0.5 + i1d) * DS0y_eg - DS1y_eg)
                A = A_in + A_flat + A_eg

                C_in   = c2 * (DS1r_in + (0.5 - i1d) * DS0r_in)
                C_flat = DS0r_flat
                C_eg   = c2 * ((wd - 0.5 + i1d) * DS0r_eg - DS1r_eg)
                C = C_in + C_flat + C_eg

                ref_in = i1d - 0.5
                ref_eg = i1d + wd - 0.5
                B_in   = c2sq * (DS2r_in - (2.0*i1d - 1.0)*DS1r_in + ref_in*ref_in*DS0r_in)
                B_flat = DS0r_flat
                B_eg   = c2sq * (DS2r_eg - 2.0*ref_eg*DS1r_eg + ref_eg*ref_eg*DS0r_eg)
                B = B_in + B_flat + B_eg

                denom = B - C * C
                if denom <= 0.0:
                    continue
                sr = A * A / denom
                if sr > best_sr:
                    best_sr = sr
                    best_depth = A / denom
                    best_dur = wd / nb * period
                    phi_mid = (i1 + 0.5 * width) / nb
                    phi_mid -= floor(phi_mid)
                    best_t0 = phi_mid * period

        free(S0y); free(S1y); free(S0r); free(S1r); free(S2r); free(S0c)

    out[0] = best_sr
    out[1] = best_t0
    out[2] = best_dur
    out[3] = best_depth


def eebls_grid(double[::1] t, double[::1] wx, double[::1] w_hat,
               double[::1] freqs, int nb, double qmin, double qmax,
               int min_bin_points, double[::1] f_arr=None):
    """Scan a frequency grid; return ``(power, best_freq, t0, dur, depth, sr, best_tau)``.

    ``nb`` is the number of phase bins; ``qmin``/``qmax`` bound the fractional
    transit duration (converted to a min/max number of bins).

    ``f_arr`` is an optional array of shape parameters (tau in [0, 1]).  If
    omitted or empty the default ``[0.0]`` (box) is used.  When multiple values
    are supplied, ``power[fi]`` stores the best SR over all tau values at that
    frequency and ``best_tau`` records the tau at the global optimum.
    """
    cdef Py_ssize_t nf = freqs.shape[0]
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t fi, ti, n_tau

    cdef int kmi = <int>(qmin * nb)
    if kmi < 1:
        kmi = 1
    cdef int kma = <int>(qmax * nb) + 1
    if kma > nb:
        kma = nb

    # Resolve tau array
    cdef double[::1] tau_arr
    if f_arr is None or f_arr.shape[0] == 0:
        tau_arr = np.array([0.0], dtype=np.float64)
    else:
        tau_arr = f_arr
    n_tau = tau_arr.shape[0]

    cdef cnp.ndarray[double, ndim=1] power = np.zeros(nf, dtype=np.float64)
    cdef double[::1] pw = power
    cdef double[::1] ybin = np.empty(nb, dtype=np.float64)
    cdef double[::1] rbin = np.empty(nb, dtype=np.float64)
    cdef long[::1] cbin = np.empty(nb, dtype=np.int64)

    cdef double out[4]
    cdef double tau, period, f
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0, best_depth = 0.0
    cdef double best_tau = tau_arr[0]
    cdef double best_freq = freqs[0]

    for fi in range(nf):
        f = freqs[fi]
        period = 1.0 / f
        for ti in range(n_tau):
            tau = tau_arr[ti]
            with nogil:
                _eebls_period_trap(&t[0], &wx[0], &w_hat[0], n, period, nb,
                                   kmi, kma, min_bin_points, tau,
                                   &ybin[0], &rbin[0], &cbin[0], out)
            if out[0] > pw[fi]:
                pw[fi] = out[0]
            if out[0] > best_sr:
                best_sr = out[0]
                best_t0 = out[1]
                best_dur = out[2]
                best_depth = out[3]
                best_freq = f
                best_tau = tau

    return power, best_freq, best_t0, best_dur, best_depth, best_sr, best_tau
