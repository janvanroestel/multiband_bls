# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled prefix-sum binned BLS with log-spaced widths (CPU ``_fast`` cores).

CPU counterparts of the fast GPU kernels (:func:`multiband_bls.eebls_gpu_fast`,
:func:`multiband_bls.multiband_eebls_gpu_fast`): after binning, every bin array
is turned into an inclusive cumulative sum, so each window sum is an O(1)
difference, and only the caller's (log-spaced) widths are evaluated. The
per-frequency cost is ``O(N + nb * n_widths * B)`` instead of the
``O(N + nb * kma * B)`` of :mod:`multiband_bls._eebls` /
:mod:`multiband_bls._meebls`.

Unlike the GPU kernels these run in double precision (no float-float fold, exact
64-bit point counts). The whole frequency loop runs without the GIL, so callers
can split the frequency grid over threads. Only the periodogram is returned;
the entry points recompute ``t0``/duration/depth exactly at the best frequency.

Gates (same as the GPU fast kernels and the exact cores):

* single band: ``min_points <= count <= n - 1`` in the window;
* multiband: total count over bands ``>= min_points``; a band contributes only
  with ``1 <= count < band_n`` in the window, and a window needs at least one
  contributing band.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport floor

cnp.import_array()


cdef inline int _bin_of(double t, double f, int nb) noexcept nogil:
    cdef double ph = t * f
    ph -= floor(ph)
    cdef int ib = <int>(nb * ph)
    if ib >= nb:
        ib = nb - 1
    return ib


def eebls_fast_grid(double[::1] t, double[::1] wx, double[::1] w_hat,
                    double[::1] freqs, int nb, cnp.int32_t[::1] widths,
                    long long min_points):
    """Periodogram ``SR^2`` of the single-band prefix-sum BLS over ``freqs``.

    ``t`` should be median-centred by the caller (as the GPU fast kernels do);
    ``wx = w_hat * x_tilde``. ``widths`` are window widths in bins, each in
    ``[1, nb]``. Returns a float64 array of length ``len(freqs)``.
    """
    cdef Py_ssize_t nf = freqs.shape[0]
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t nw = widths.shape[0]
    cdef cnp.ndarray[double, ndim=1] power = np.zeros(nf, dtype=np.float64)
    cdef double[::1] pw = power
    # cumulative arrays with a leading zero: c[k] = sum of bins [0, k)
    cdef double[::1] cy = np.empty(nb + 1, dtype=np.float64)
    cdef double[::1] cr = np.empty(nb + 1, dtype=np.float64)
    cdef cnp.int64_t[::1] cc = np.empty(nb + 1, dtype=np.int64)
    cdef Py_ssize_t i, fi
    cdef int k, ib, i1, wi, ww, j2
    cdef double sy, sr, denom, sr2, best
    cdef cnp.int64_t sc
    cdef cnp.int64_t nmax = n - 1

    with nogil:
        for fi in range(nf):
            for k in range(nb + 1):
                cy[k] = 0.0
                cr[k] = 0.0
                cc[k] = 0
            for i in range(n):
                ib = _bin_of(t[i], freqs[fi], nb) + 1
                cy[ib] += wx[i]
                cr[ib] += w_hat[i]
                cc[ib] += 1
            for k in range(1, nb + 1):
                cy[k] += cy[k - 1]
                cr[k] += cr[k - 1]
                cc[k] += cc[k - 1]
            best = 0.0
            for i1 in range(nb):
                for wi in range(nw):
                    ww = widths[wi]
                    j2 = i1 + ww          # exclusive end
                    if j2 <= nb:
                        sc = cc[j2] - cc[i1]
                        if sc < min_points or sc > nmax:
                            continue
                        sy = cy[j2] - cy[i1]
                        sr = cr[j2] - cr[i1]
                    else:
                        j2 -= nb
                        sc = (cc[nb] - cc[i1]) + cc[j2]
                        if sc < min_points or sc > nmax:
                            continue
                        sy = (cy[nb] - cy[i1]) + cy[j2]
                        sr = (cr[nb] - cr[i1]) + cr[j2]
                    denom = sr * (1.0 - sr)
                    if denom <= 0.0:
                        continue
                    sr2 = sy * sy / denom
                    if sr2 > best:
                        best = sr2
            pw[fi] = best
    return power


def meebls_fast_grid(double[::1] t, double[::1] wx, double[::1] w_hat,
                     cnp.int64_t[::1] band, int n_bands, double[::1] freqs, int nb,
                     cnp.int32_t[::1] widths, long long min_points, double[::1] band_w):
    """Periodogram ``sum_b W_b s_b^2 / (r_b (1 - r_b))`` of the multiband
    prefix-sum BLS over ``freqs``.

    Inputs are the merged (all-band) arrays of :func:`multiband_bls.api._merge_bands`,
    with ``t`` median-centred by the caller. Returns a float64 array of length
    ``len(freqs)``.
    """
    cdef Py_ssize_t nf = freqs.shape[0]
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t nw = widths.shape[0]
    cdef int nb1 = nb + 1
    cdef cnp.ndarray[double, ndim=1] power = np.zeros(nf, dtype=np.float64)
    cdef double[::1] pw = power
    # per band b: c[b * (nb + 1) + k] = sum of that band's bins [0, k)
    cdef double[::1] cy = np.empty(n_bands * nb1, dtype=np.float64)
    cdef double[::1] cr = np.empty(n_bands * nb1, dtype=np.float64)
    cdef cnp.int64_t[::1] cc = np.empty(n_bands * nb1, dtype=np.int64)
    cdef cnp.int64_t[::1] band_n = np.bincount(np.asarray(band), minlength=n_bands).astype(np.int64)
    cdef Py_ssize_t i, fi
    cdef int k, ib, i1, wi, ww, j2, bb, o, kk
    cdef bint wrap
    cdef double sy, sr, denom, sr2, best
    cdef cnp.int64_t sc, cnt_tot

    with nogil:
        for fi in range(nf):
            for k in range(n_bands * nb1):
                cy[k] = 0.0
                cr[k] = 0.0
                cc[k] = 0
            for i in range(n):
                ib = band[i] * nb1 + _bin_of(t[i], freqs[fi], nb) + 1
                cy[ib] += wx[i]
                cr[ib] += w_hat[i]
                cc[ib] += 1
            for bb in range(n_bands):
                o = bb * nb1
                for k in range(1, nb1):
                    cy[o + k] += cy[o + k - 1]
                    cr[o + k] += cr[o + k - 1]
                    cc[o + k] += cc[o + k - 1]
            best = 0.0
            for i1 in range(nb):
                for wi in range(nw):
                    ww = widths[wi]
                    j2 = i1 + ww          # exclusive end
                    wrap = j2 > nb
                    if wrap:
                        j2 -= nb
                    cnt_tot = 0
                    sr2 = 0.0
                    kk = 0
                    for bb in range(n_bands):
                        o = bb * nb1
                        if not wrap:
                            sc = cc[o + j2] - cc[o + i1]
                            sy = cy[o + j2] - cy[o + i1]
                            sr = cr[o + j2] - cr[o + i1]
                        else:
                            sc = (cc[o + nb] - cc[o + i1]) + cc[o + j2]
                            sy = (cy[o + nb] - cy[o + i1]) + cy[o + j2]
                            sr = (cr[o + nb] - cr[o + i1]) + cr[o + j2]
                        cnt_tot += sc
                        if sc < 1 or sc >= band_n[bb]:
                            continue
                        denom = sr * (1.0 - sr)
                        if denom <= 0.0:
                            continue
                        sr2 += band_w[bb] * sy * sy / denom
                        kk += 1
                    if cnt_tot < min_points or kk == 0:
                        continue
                    if sr2 > best:
                        best = sr2
            pw[fi] = best
    return power
