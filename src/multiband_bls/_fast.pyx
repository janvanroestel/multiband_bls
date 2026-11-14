# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled prefix-sum binned BLS with log-spaced widths (CPU ``_fast`` cores).

CPU counterparts of the fast GPU kernels (:func:`multiband_bls.eebls_gpu_fast`,
:func:`multiband_bls.multiband_eebls_gpu_fast`): after binning, every bin array
is turned into an inclusive cumulative sum, so each window sum is an O(1)
difference, and only the caller's (log-spaced) widths are evaluated. The
per-frequency cost is ``O(N + nb * n_widths * B)`` instead of the
``O(N + nb * kma * B)`` of :mod:`multiband_bls._eebls` /
:mod:`multiband_bls._meebls`.

The per-frequency work (fold, bin, prefix sums, window sweep) is plain C in
``_fast_kernels.h``: the sweep is width-first, first listing the start bins whose
(pooled) point count passes the ``min_points`` gate -- most windows in sparse data do
not -- and only then forming the statistic on the survivors; the window evaluators are
AVX2-multiversioned on x86-64 Linux. This module only allocates scratch buffers and
loops over frequencies.

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

cnp.import_array()

cdef extern from "_fast_kernels.h" nogil:
    void mbls_prepare_single(const double* t, const double* wx, const double* w, long n,
                             double f, int nb, double* cy, double* cr, double* cc, int* ibuf)
    void mbls_prepare_multi(const double* t, const double* wx, const double* w,
                            const int* band, long n, int n_bands, double f, int nb,
                            double* cy, double* cr, double* cc, double* ct, int* ibuf)
    double mbls_sweep_single(const double* cy, const double* cr, const double* cc, int nb,
                             const int* widths, int nw, double mp, double nmax, int* idx)
    double mbls_sweep_multi(const double* cy, const double* cr, const double* cc,
                            const double* ct, int n_bands, int nb, const int* widths,
                            int nw, double mp, const double* band_w, const double* band_n,
                            int* idx, double* acc)


cdef _check_phase_int32(double[::1] t, double[::1] freqs):
    """The fold truncates ``t * f`` to int32: require ``max|t| * max(f) < 2**30``."""
    cdef double tmax = float(np.max(np.abs(t))) if t.shape[0] else 0.0
    cdef double fmax = float(np.max(freqs))
    if tmax * fmax >= 1073741824.0:
        raise ValueError("fast BLS core: max|t| * max(frequency) must be < 2**30 cycles")


def eebls_fast_grid(double[::1] t, double[::1] wx, double[::1] w_hat,
                    double[::1] freqs, int nb, cnp.int32_t[::1] widths,
                    long long min_points):
    """Periodogram ``SR^2`` of the single-band prefix-sum BLS over ``freqs``.

    ``t`` should be median-centred by the caller (as the GPU fast kernels do);
    ``wx = w_hat * x_tilde``. ``widths`` are window widths in bins, each in
    ``[1, nb]``. Returns a float64 array of length ``len(freqs)``.
    """
    cdef Py_ssize_t nf = freqs.shape[0]
    cdef long n = t.shape[0]
    cdef int nw = <int>widths.shape[0]
    cdef cnp.ndarray[double, ndim=1] power = np.zeros(nf, dtype=np.float64)
    cdef double[::1] pw = power
    # c[k] = sum of bins [0, k); counts are doubles (exact for n < 2**53)
    cdef double[::1] cy = np.empty(nb + 1, dtype=np.float64)
    cdef double[::1] cr = np.empty(nb + 1, dtype=np.float64)
    cdef double[::1] cc = np.empty(nb + 1, dtype=np.float64)
    cdef cnp.int32_t[::1] idx = np.empty(nb + 1, dtype=np.int32)
    cdef cnp.int32_t[::1] ibuf = np.empty(max(n, 1), dtype=np.int32)
    cdef Py_ssize_t fi
    cdef double mp = <double>min_points
    cdef double nmax = <double>(n - 1)    # keep an out-of-transit point

    if nw == 0 or nf == 0:
        return power
    _check_phase_int32(t, freqs)
    with nogil:
        for fi in range(nf):
            mbls_prepare_single(&t[0], &wx[0], &w_hat[0], n, freqs[fi], nb,
                                &cy[0], &cr[0], &cc[0], <int*>&ibuf[0])
            pw[fi] = mbls_sweep_single(&cy[0], &cr[0], &cc[0], nb, <const int*>&widths[0], nw,
                                       mp, nmax, <int*>&idx[0])
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
    cdef long n = t.shape[0]
    cdef int nw = <int>widths.shape[0]
    cdef Py_ssize_t L = nb + 1
    cdef cnp.ndarray[double, ndim=1] power = np.zeros(nf, dtype=np.float64)
    cdef double[::1] pw = power
    # per band b: c[b * L + k] = sum of that band's bins [0, k); ct = pooled counts
    cdef double[::1] cy = np.empty(n_bands * L, dtype=np.float64)
    cdef double[::1] cr = np.empty(n_bands * L, dtype=np.float64)
    cdef double[::1] cc = np.empty(n_bands * L, dtype=np.float64)
    cdef double[::1] ct = np.empty(L, dtype=np.float64)
    cdef double[::1] acc = np.empty(nb + 1, dtype=np.float64)
    cdef cnp.int32_t[::1] idx = np.empty(nb + 1, dtype=np.int32)
    cdef cnp.int32_t[::1] ibuf = np.empty(max(n, 1), dtype=np.int32)
    cdef cnp.int32_t[::1] band32 = np.ascontiguousarray(band, dtype=np.int32)
    cdef double[::1] band_n = np.bincount(np.asarray(band), minlength=n_bands).astype(np.float64)
    cdef Py_ssize_t fi
    cdef double mp = <double>min_points

    if nw == 0 or nf == 0:
        return power
    _check_phase_int32(t, freqs)
    with nogil:
        for fi in range(nf):
            mbls_prepare_multi(&t[0], &wx[0], &w_hat[0], <const int*>&band32[0], n, n_bands,
                               freqs[fi], nb, &cy[0], &cr[0], &cc[0], &ct[0],
                               <int*>&ibuf[0])
            pw[fi] = mbls_sweep_multi(&cy[0], &cr[0], &cc[0], &ct[0], n_bands, nb,
                                      <const int*>&widths[0], nw, mp, &band_w[0], &band_n[0],
                                      <int*>&idx[0], &acc[0])
    return power
