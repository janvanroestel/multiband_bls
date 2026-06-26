"""Fast entry points backed by the compiled Cython cores.

- :func:`sparse_bls` / :func:`multiband_sparse_bls` — unbinned, O(N²/freq), precise
- :func:`eebls` / :func:`multiband_eebls` — phase-binned, O(N/freq), fast for large N

All functions return :class:`~multiband_bls.SBLSResult` with
``power = Δχ²/χ²_flat ∈ [0, 1]`` (fraction of variance explained, maximised at
the best period). Drop-in replacements for the reference implementations in
:mod:`multiband_bls.reference`.
"""

from __future__ import annotations

__all__ = ["sparse_bls", "eebls", "multiband_sparse_bls", "multiband_eebls"]

import logging
from collections.abc import Mapping

import numpy as np

from . import _eebls, _meebls, _msbls, _sbls
from .periodogram import SBLSResult
from .reference import auto_nbins, preprocess, variance_explained

logger = logging.getLogger(__name__)

Array = np.ndarray


def _preprocess_single(
    y: Array, dy: Array
) -> tuple[Array, Array, float]:
    """Preprocess a single-band light curve for the Cython cores.

    Returns ``(wx, w_hat, yy)`` where ``wx = w_hat * x_tilde``,
    ``w_hat`` are the normalised weights (sum to 1), and ``yy`` is the
    flat-model weighted variance (``Δχ²`` denominator).
    """
    w_hat, x_tilde, _ = preprocess(y, dy)
    yy = float(np.sum(w_hat * x_tilde ** 2))
    wx = np.ascontiguousarray(w_hat * x_tilde, dtype=np.float64)
    w_hat = np.ascontiguousarray(w_hat, dtype=np.float64)
    return wx, w_hat, yy


def _merge_bands(
    bands: Mapping[str, tuple[Array, Array, Array]]
) -> tuple[Array, Array, Array, Array, tuple[str, ...], Array, float]:
    """Preprocess each band and merge into contiguous all-band arrays.

    Returns ``(t, wx, w_hat, band_idx, labels, w_totals, chi2_flat)`` where
    ``wx = w_hat * x_tilde`` uses each band's own normalised weights,
    ``band_idx`` labels each point with its band index, ``w_totals[b] =
    sum_i 1/sigma^2`` per band, and ``chi2_flat`` is the total flat-model chi^2
    (``sum_b sum_i (x_tilde/sigma)^2``). Shared by the multiband entry points.
    """
    labels = tuple(bands.keys())
    t_parts: list[Array] = []
    wx_parts: list[Array] = []
    w_parts: list[Array] = []
    band_parts: list[Array] = []
    w_totals: list[float] = []
    chi2_flat = 0.0
    for bi, label in enumerate(labels):
        t_b, y_b, dy_b = bands[label]
        w_hat, x_tilde, _ = preprocess(y_b, dy_b)
        w_raw = 1.0 / np.asarray(dy_b, dtype=np.float64) ** 2
        w_totals.append(float(w_raw.sum()))
        chi2_flat += float(np.sum(w_raw * x_tilde ** 2))
        t_parts.append(np.asarray(t_b, dtype=np.float64))
        wx_parts.append(w_hat * x_tilde)
        w_parts.append(w_hat)
        band_parts.append(np.full(len(t_b), bi, dtype=np.int64))
    return (
        np.ascontiguousarray(np.concatenate(t_parts), dtype=np.float64),
        np.ascontiguousarray(np.concatenate(wx_parts), dtype=np.float64),
        np.ascontiguousarray(np.concatenate(w_parts), dtype=np.float64),
        np.ascontiguousarray(np.concatenate(band_parts), dtype=np.int64),
        labels,
        np.asarray(w_totals, dtype=np.float64),
        chi2_flat,
    )


def sparse_bls(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    q_max: float = 0.15,
    min_points: int = 3,
) -> SBLSResult:
    """Single-band Sparse BLS (compiled). See :func:`reference.sparse_bls_reference`.

    ``power``/``best_power`` are ``Delta chi^2 / chi^2_flat = SR^2 / YY`` (the
    fraction of variance explained, in ``[0, 1]``, maximised at the period).
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    freqs = np.ascontiguousarray(frequencies, dtype=np.float64)
    wx, w_hat, yy = _preprocess_single(y, dy)

    power, best_freq, t0, dur, depth, sr = _sbls.sbls_grid(
        t, wx, w_hat, freqs, float(q_max), int(min_points)
    )
    return SBLSResult(
        frequency=freqs,
        power=variance_explained(power, yy),
        best_frequency=float(best_freq),
        best_period=float(1.0 / best_freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=float(depth),
        best_power=float(variance_explained(np.array([sr]), yy)[0]),
    )


def eebls(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    qmin: float = 0.01,
    qmax: float = 0.10,
    min_points: int = 3,
) -> SBLSResult:
    """Classic binned BLS with edge-effect handling (Kovacs et al. 2002).

    The phase-folded light curve is binned onto ``nbins`` phase bins and the box
    is searched over bins, giving ``O(N)`` per-period cost (linear in the number
    of points for a fixed bin count) versus the unbinned :func:`sparse_bls`'s
    ``O(N^2)``. ``qmin``/``qmax`` bound the fractional transit duration.
    ``power``/``best_power`` are ``Delta chi^2 / chi^2_flat = SR^2 / YY``.

    If ``nbins`` is ``None`` (default) it is chosen automatically as
    ``clip(5 / qmin, 50, 500)`` so that at least 5 bins fall inside the
    narrowest transit searched.
    """
    if nbins is None:
        nbins = auto_nbins(qmin)
    t = np.ascontiguousarray(t, dtype=np.float64)
    freqs = np.ascontiguousarray(frequencies, dtype=np.float64)
    tau_arr = np.zeros(1, dtype=np.float64)  # box only
    wx, w_hat, yy = _preprocess_single(y, dy)

    power, best_freq, t0, dur, depth, sr, _ = _eebls.eebls_grid(
        t, wx, w_hat, freqs, int(nbins), float(qmin), float(qmax),
        int(min_points), tau_arr,
    )
    return SBLSResult(
        frequency=freqs,
        power=variance_explained(power, yy),
        best_frequency=float(best_freq),
        best_period=float(1.0 / best_freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=float(depth),
        best_power=float(variance_explained(np.array([sr]), yy)[0]),
    )


def multiband_sparse_bls(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    q_max: float = 0.15,
    min_points: int = 3,
) -> SBLSResult:
    """Multiband Sparse BLS (compiled). See :func:`reference.multiband_sparse_bls_reference`.

    Bands combine with matched-filter weighting (each weighted by its total
    inverse variance ``W_b``). ``power``/``best_power`` are
    ``Delta chi^2 / chi^2_flat`` -- the fraction of variance explained, in
    ``[0, 1]``, maximised at the best period.
    """
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    freqs = np.ascontiguousarray(frequencies, dtype=np.float64)
    band_w = np.ascontiguousarray(w_totals, dtype=np.float64)

    power, best_freq, t0, dur, depths, sr = _msbls.msbls_grid(
        t_all, wx_all, w_all, band_all, len(labels), freqs,
        float(q_max), int(min_points), band_w,
    )
    return SBLSResult(
        frequency=freqs,
        power=variance_explained(power, chi2_flat),
        best_frequency=float(best_freq),
        best_period=float(1.0 / best_freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=np.asarray(depths),
        best_power=float(variance_explained(np.array([sr]), chi2_flat)[0]),
        bands=labels,
    )


def multiband_eebls(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    qmin: float = 0.01,
    qmax: float = 0.10,
    min_points: int = 3,
) -> SBLSResult:
    """Binned multiband BLS (compiled). See :func:`reference.multiband_eebls_reference`.

    Combines the speed of binning with the per-band-depth multiband model: each
    band is binned onto ``nbins`` shared phase bins; ``O(N)`` per period rather
    than the ``O(N^2)`` of :func:`multiband_sparse_bls`. Matched-filter weighting;
    ``power``/``best_power`` are ``Delta chi^2 / chi^2_flat`` (as above).

    If ``nbins`` is ``None`` (default) it is chosen automatically as
    ``clip(5 / qmin, 50, 500)``.
    """
    if nbins is None:
        nbins = auto_nbins(qmin)
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    freqs = np.ascontiguousarray(frequencies, dtype=np.float64)
    band_w = np.ascontiguousarray(w_totals, dtype=np.float64)

    power, best_freq, t0, dur, depths, sr = _meebls.meebls_grid(
        t_all, wx_all, w_all, band_all, len(labels), freqs, int(nbins),
        float(qmin), float(qmax), int(min_points), band_w,
    )
    return SBLSResult(
        frequency=freqs,
        power=variance_explained(power, chi2_flat),
        best_frequency=float(best_freq),
        best_period=float(1.0 / best_freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=np.asarray(depths),
        best_power=float(variance_explained(np.array([sr]), chi2_flat)[0]),
        bands=labels,
    )
