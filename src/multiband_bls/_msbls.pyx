# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled multiband Sparse BLS core.

Generalises the single-band SBLS to ``B`` bands that share one trial period,
epoch and duration, but each fit their own baseline and depth. The combined
statistic is ``sqrt(sum_b band_w[b] * s_b^2 / (r_b (1 - r_b)))``. ``min_points`` is the
minimum *total* number of in-transit points; any band with >= 1 in-transit
point contributes. Mirrors
:func:`multiband_bls.reference.multiband_sparse_bls_reference`.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport floor, NAN

cnp.import_array()


cdef void _multiband_period(const double* phi, const double* wx,
                            const double* w, const cnp.int64_t* band,
                            Py_ssize_t n, int n_bands, double period,
                            double q_min, double q_max, int min_points, double sign,
                            const double* band_w, const cnp.int64_t* band_n,
                            double* s, double* r, cnp.int64_t* cnt, double* contrib,
                            double* depths, double* out) noexcept nogil:
    """Fill ``out = [SR, t0, duration, k]`` and ``depths[n_bands]`` for one period.

    ``k`` is the number of bands contributing to the best window (0 if no
    window was selected); depths of non-contributing bands are NaN.

    ``s, r, cnt, contrib, depths`` are caller-provided scratch buffers of length
    ``n_bands`` (contents are overwritten).

    Adding one sample changes only its own band, so each band's term of the
    combined statistic is cached in ``contrib`` (< 0: band not contributing) and
    the running total is updated in O(1) per window extension instead of
    re-summing all bands; per-band depths are extracted only for new best
    windows. The reference implementation performs the same operations in the
    same order.
    """
    cdef Py_ssize_t i1, k, i2, i1m, i2p
    cdef int bb
    cdef cnp.int64_t total  # point counts: 64-bit, like the count buffers
    cdef cnp.int64_t b
    cdef double frac, phi_ing, phi_eg, denom, sr2, c
    cdef double p_i1, p_i1m, p_i2, p_i2p, phi_mid
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0
    cdef int kk, best_k = 0

    for bb in range(n_bands):
        depths[bb] = NAN

    for i1 in range(n):
        i1m = i1 - 1
        if i1m < 0:
            i1m += n
        p_i1 = phi[i1]
        p_i1m = phi[i1m]
        if p_i1m == p_i1:
            # i1 sits inside a group of identical phases: an ingress edge here
            # would put points at the same phase both in and out of transit.
            continue
        # Wrap-around is decided by index (sorted phases), not by comparing
        # values, so tied phases cannot be mistaken for a wrap.
        if i1 > 0:
            phi_ing = 0.5 * (p_i1m + p_i1)
        else:
            phi_ing = 0.5 * (p_i1m + p_i1) - 0.5

        for bb in range(n_bands):
            s[bb] = 0.0
            r[bb] = 0.0
            cnt[bb] = 0
            contrib[bb] = -1.0
        total = 0
        sr2 = 0.0
        kk = 0

        for k in range(n - 1):
            i2 = i1 + k
            if i2 >= n:
                i2 -= n
            b = band[i2]
            s[b] += wx[i2]
            r[b] += w[i2]
            cnt[b] += 1
            total += 1

            # refresh band b's cached contribution (identifiable only with points
            # both in and out of the window, and the right sign)
            if contrib[b] >= 0.0:
                sr2 -= contrib[b]
                kk -= 1
                contrib[b] = -1.0
            if cnt[b] < band_n[b]:
                denom = r[b] * (1.0 - r[b])
                if denom > 0.0 and sign * s[b] >= 0.0:
                    c = band_w[b] * s[b] * s[b] / denom
                    contrib[b] = c
                    sr2 += c
                    kk += 1
            if kk == 0:
                sr2 = 0.0  # drop accumulated rounding once nothing contributes

            i2p = i2 + 1
            if i2p >= n:
                i2p -= n
            p_i2 = phi[i2]
            p_i2p = phi[i2p]
            if i2p != 0:
                phi_eg = 0.5 * (p_i2 + p_i2p)
            else:
                phi_eg = 0.5 * (p_i2 + p_i2p) + 0.5

            frac = phi_eg - phi_ing
            frac -= floor(frac)
            if frac >= q_max:
                break
            if p_i2p == p_i2:
                continue  # egress edge would split identical phases
            if frac < q_min or total < min_points:
                continue
            if kk == 0:
                continue
            if sr2 > best_sr:
                best_sr = sr2
                best_k = kk
                best_dur = frac * period
                phi_mid = phi_ing + 0.5 * frac
                phi_mid -= floor(phi_mid)
                best_t0 = phi_mid * period
                for bb in range(n_bands):
                    if contrib[bb] >= 0.0:
                        depths[bb] = s[bb] / (r[bb] * (1.0 - r[bb]))
                    else:
                        depths[bb] = NAN

    out[0] = best_sr
    out[1] = best_t0
    out[2] = best_dur
    out[3] = best_k


def msbls_grid(double[::1] t, double[::1] wx, double[::1] w_hat,
               cnp.int64_t[::1] band, int n_bands, double[::1] freqs,
               double q_min, double q_max, int min_points, double[::1] band_w,
               double sign=0.0):
    """Scan a frequency grid for the multiband search.

    Inputs are the merged (all-band) arrays: ``t``, ``wx = w_hat * x_tilde``
    with per-band normalised weights, ``w_hat``, and integer ``band`` labels.
    Returns ``(power, best_freq, t0, dur, depths[n_bands], sr, k_used[nf])``.
    """
    cdef Py_ssize_t nf = freqs.shape[0]
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t fi, i, j

    cdef cnp.ndarray[double, ndim=1] power = np.zeros(nf, dtype=np.float64)
    cdef double[::1] pw = power
    cdef cnp.ndarray[double, ndim=1] best_depths = np.full(n_bands, np.nan, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] k_used = np.zeros(nf, dtype=np.int64)
    cdef cnp.int64_t[::1] ku = k_used
    cdef double[::1] bd = best_depths

    cdef cnp.ndarray[double, ndim=1] phi_np = np.empty(n, dtype=np.float64)
    cdef double[::1] phi = phi_np
    cdef double[::1] phi_s = np.empty(n, dtype=np.float64)
    cdef double[::1] wx_s = np.empty(n, dtype=np.float64)
    cdef double[::1] w_s = np.empty(n, dtype=np.float64)
    cdef cnp.int64_t[::1] band_s = np.empty(n, dtype=np.int64)
    cdef cnp.intp_t[::1] order

    # scratch buffers (length n_bands)
    cdef double[::1] s_buf = np.empty(n_bands, dtype=np.float64)
    cdef double[::1] r_buf = np.empty(n_bands, dtype=np.float64)
    cdef cnp.int64_t[::1] cnt_buf = np.empty(n_bands, dtype=np.int64)
    cdef double[::1] contrib_buf = np.empty(n_bands, dtype=np.float64)
    cdef cnp.int64_t[::1] band_n = np.bincount(np.asarray(band), minlength=n_bands).astype(np.int64)
    cdef double[::1] dep_buf = np.empty(n_bands, dtype=np.float64)

    cdef double out[4]
    cdef double period, f, ph
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0
    cdef double best_freq = freqs[0]
    cdef int bb

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
            band_s[i] = band[j]
        with nogil:
            _multiband_period(&phi_s[0], &wx_s[0], &w_s[0], &band_s[0], n,
                              n_bands, period, q_min, q_max, min_points, sign,
                              &band_w[0], &band_n[0], &s_buf[0], &r_buf[0], &cnt_buf[0],
                              &contrib_buf[0], &dep_buf[0], out)
        pw[fi] = out[0]
        ku[fi] = <cnp.int64_t>out[3]
        if out[0] > best_sr:
            best_sr = out[0]
            best_t0 = out[1]
            best_dur = out[2]
            best_freq = f
            for bb in range(n_bands):
                bd[bb] = dep_buf[bb]

    return power, best_freq, best_t0, best_dur, best_depths, best_sr, k_used
