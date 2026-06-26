# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled binned multiband BLS (``multiband eeBLS``).

A hybrid of the classic binned eeBLS (:mod:`multiband_bls._eebls`) and the
unbinned multiband SBLS (:mod:`multiband_bls._msbls`): every band is binned onto
the *same* ``nb`` phase bins (bin ``k`` spans phase ``[k/nb, (k+1)/nb)`` for all
bands), and the box is searched over a shared window of bins while each band
keeps its own baseline and depth. The combined statistic is
``SR = sqrt(sum_b s_b^2 / (r_b (1 - r_b)))``.

Per-period cost is ``O(N)`` binning + ``O(nb * kma * B)`` bin loop -- linear in
``N`` for a fixed bin count, like the single-band eeBLS but with the ``O(B)``
multiband factor. Mirrors
:func:`multiband_bls.reference.multiband_eebls_reference`.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport floor

cnp.import_array()


cdef void _meebls_period(const double* t, const double* wx, const double* w,
                         const long* band, Py_ssize_t n, int n_bands, int nb,
                         double period, int kmi, int kma, int min_points,
                         const double* band_w,
                         double* ybin, double* rbin, long* cbin,
                         double* s, double* r, long* cnt,
                         double* depths, double* out) noexcept nogil:
    """Fill ``out = [SR, t0, duration]`` and ``depths[n_bands]`` for one period.

    ``ybin, rbin, cbin`` are ``n_bands*nb`` flattened bin buffers; ``s, r, cnt,
    depths`` are length-``n_bands`` scratch buffers (all overwritten).
    """
    cdef Py_ssize_t i
    cdef int j, jj, ib, i1, width, bb, total
    cdef long b
    cdef double ph, denom, sr2, phi_mid
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0
    cdef bint ok

    for bb in range(n_bands):
        depths[bb] = 0.0
    for j in range(n_bands * nb):
        ybin[j] = 0.0
        rbin[j] = 0.0
        cbin[j] = 0

    for i in range(n):
        ph = t[i] / period
        ph -= floor(ph)
        ib = <int>(nb * ph)
        if ib >= nb:
            ib = nb - 1
        b = band[i]
        ybin[b * nb + ib] += wx[i]
        rbin[b * nb + ib] += w[i]
        cbin[b * nb + ib] += 1

    for i1 in range(nb):
        for bb in range(n_bands):
            s[bb] = 0.0
            r[bb] = 0.0
            cnt[bb] = 0
        total = 0
        width = 0
        for j in range(i1, i1 + kma):
            jj = j
            if jj >= nb:
                jj -= nb
            for bb in range(n_bands):
                s[bb] += ybin[bb * nb + jj]
                r[bb] += rbin[bb * nb + jj]
                cnt[bb] += cbin[bb * nb + jj]
                total += <int>cbin[bb * nb + jj]
            width += 1
            if width < kmi:
                continue
            if total < min_points:
                continue
            sr2 = 0.0
            ok = False
            for bb in range(n_bands):
                if cnt[bb] < 1:
                    continue
                denom = r[bb] * (1.0 - r[bb])
                if denom <= 0.0:
                    continue
                sr2 += band_w[bb] * s[bb] * s[bb] / denom
                ok = True
            if not ok:
                continue
            if sr2 > best_sr:
                best_sr = sr2
                best_dur = (<double>width) / nb * period
                phi_mid = (i1 + 0.5 * width) / nb
                phi_mid -= floor(phi_mid)
                best_t0 = phi_mid * period
                for bb in range(n_bands):
                    denom = r[bb] * (1.0 - r[bb])
                    if cnt[bb] >= 1 and denom > 0.0:
                        depths[bb] = s[bb] / denom
                    else:
                        depths[bb] = 0.0

    out[0] = best_sr
    out[1] = best_t0
    out[2] = best_dur


def meebls_grid(double[::1] t, double[::1] wx, double[::1] w_hat,
                long[::1] band, int n_bands, double[::1] freqs, int nb,
                double qmin, double qmax, int min_points, double[::1] band_w):
    """Scan a frequency grid for the binned multiband search.

    Inputs are the merged (all-band) arrays: ``t``, ``wx = w_hat * x_tilde`` with
    per-band normalised weights, ``w_hat``, and integer ``band`` labels. Returns
    ``(power, best_freq, t0, dur, depths[n_bands], sr)``.
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
    cdef cnp.ndarray[double, ndim=1] best_depths = np.zeros(n_bands, dtype=np.float64)
    cdef double[::1] bd = best_depths

    cdef double[::1] ybin = np.empty(n_bands * nb, dtype=np.float64)
    cdef double[::1] rbin = np.empty(n_bands * nb, dtype=np.float64)
    cdef long[::1] cbin = np.empty(n_bands * nb, dtype=np.int64)
    cdef double[::1] s_buf = np.empty(n_bands, dtype=np.float64)
    cdef double[::1] r_buf = np.empty(n_bands, dtype=np.float64)
    cdef long[::1] cnt_buf = np.empty(n_bands, dtype=np.int64)
    cdef double[::1] dep_buf = np.empty(n_bands, dtype=np.float64)

    cdef double out[3]
    cdef double period, f
    cdef double best_sr = 0.0, best_t0 = 0.0, best_dur = 0.0
    cdef double best_freq = freqs[0]
    cdef int bb

    for fi in range(nf):
        f = freqs[fi]
        period = 1.0 / f
        with nogil:
            _meebls_period(&t[0], &wx[0], &w_hat[0], &band[0], n, n_bands, nb,
                           period, kmi, kma, min_points, &band_w[0],
                           &ybin[0], &rbin[0], &cbin[0], &s_buf[0], &r_buf[0],
                           &cnt_buf[0], &dep_buf[0], out)
        pw[fi] = out[0]
        if out[0] > best_sr:
            best_sr = out[0]
            best_t0 = out[1]
            best_dur = out[2]
            best_freq = f
            for bb in range(n_bands):
                bd[bb] = dep_buf[bb]

    return power, best_freq, best_t0, best_dur, best_depths, best_sr
