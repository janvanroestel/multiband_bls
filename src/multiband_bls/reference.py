"""Pure-Python / numpy reference implementation of Sparse BLS (SBLS) and
multiband extensions.

This module is the *readable specification* and the test oracle for the
compiled Cython cores.  It follows Panahi & Zucker (2021), PASP 133, 024502
(arXiv:2103.06193): the light curve is left unbinned and box edges are placed
only at observed phases.

Notation (their Eqs. 1-10), with ``x`` the magnitude (or flux) and ``sigma``
its uncertainty:

    w_j   = sigma_j^-2,   W = sum_j w_j,   w-hat_j = w_j / W
    mu    = sum_j w-hat_j x_j,             x-tilde_j = x_j - mu
    phi_i = (t_i mod P) / P                          (then sort by phi)
    s     = sum_in w-hat_i x-tilde_i,  r = sum_in w-hat_i
    SR    = sqrt( s^2 / (r (1 - r)) )
    M-hat = mu - s/(1-r),              d-hat = s / (r (1-r))

The transit duration is estimated from the phases of the out-of-transit
neighbours of the in-transit block (Eqs. 6-10).

The functions here are intentionally simple (explicit Python loops) so they are
easy to verify; for production use the Cython entry points in
:mod:`multiband_bls.api`.
"""

from __future__ import annotations

__all__ = [
    "preprocess",
    "auto_nbins",
    "variance_explained",
    "build_frequency_grid",
    "sparse_bls_reference",
    "multiband_sparse_bls_reference",
    "eebls_reference",
    "multiband_eebls_reference",
    "coadd_bands",
]

import logging
from collections.abc import Mapping
from typing import cast

import numpy as np

from .periodogram import BLSResult

logger = logging.getLogger(__name__)

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Preprocessing and grids                                                      #
# --------------------------------------------------------------------------- #
def preprocess(y: Array, dy: Array) -> tuple[Array, Array, float]:
    """Return ``(w_hat, x_tilde, mu)`` from Eqs. 1-2.

    ``w_hat`` are the normalised weights (sum to 1) and ``x_tilde`` is the
    weighted-mean-subtracted signal.
    """
    w = 1.0 / np.asarray(dy, dtype=float) ** 2
    W = w.sum()
    w_hat = w / W
    mu = float(np.sum(w_hat * np.asarray(y, dtype=float)))
    x_tilde = np.asarray(y, dtype=float) - mu
    return w_hat, x_tilde, mu


def auto_nbins(qmin: float, n_res: int = 5) -> int:
    """Return the recommended number of phase bins for a given minimum transit fraction.

    Ensures at least ``n_res`` bins span the narrowest transit (``qmin``),
    capped between 50 and 500 to keep compute tractable.
    """
    return int(np.clip(round(n_res / qmin), 50, 500))


def variance_explained(sr2: Array, denom: float) -> Array:
    """Map an SR² periodogram to ``Delta chi^2 / chi^2_flat``.

    All cores accumulate ``SR² = s² / (r(1-r))`` (or the multiband analogue
    ``T3 = sum_b W_b SR_b²``); ``denom`` is the flat-model total chi² (``YY``
    for one band, ``chi2_flat = sum_b W_b YY_b`` for multiband). Result lies
    in ``[0, 1]`` and is maximised at the period.
    """
    if denom <= 0.0:
        return np.zeros_like(np.asarray(sr2, dtype=float))
    return np.asarray(sr2, dtype=float) / denom


def build_frequency_grid(
    t: Array,
    f_min: float | None = None,
    f_max: float | None = None,
    df: float | None = None,
    oversample: float = 3.0,
    p_min: float | None = None,
    p_max: float | None = None,
    q_min: float = 0.01,
) -> Array:
    """Uniform frequency grid (1/day).

    Specify the search range either in frequency (``f_min``, ``f_max``) or in
    period (``p_min``, ``p_max``). Note the inversion: ``p_min → f_max`` and
    ``p_max → f_min``. Mixing both for the same bound raises ``ValueError``.

    If ``df`` is not given it defaults to ``q_min / (oversample * baseline)``,
    where ``q_min`` is the shortest fractional transit duration to resolve.
    The default ``q_min=0.01`` matches :func:`eebls` and :func:`multiband_eebls`.

    Raises
    ------
    ValueError
        If conflicting bounds are given, if ``f_min >= f_max``, or if the time
        array spans zero baseline.
    """
    # Convert period bounds → frequency bounds (note the inversion).
    period_given = p_min is not None or p_max is not None
    freq_given = f_min is not None or f_max is not None
    if period_given and freq_given:
        raise ValueError("Provide either (p_min, p_max) or (f_min, f_max), not both.")
    if p_min is not None:
        f_max = 1.0 / p_min
    if p_max is not None:
        f_min = 1.0 / p_max

    if f_min is None or f_max is None:
        raise ValueError("Specify either (f_min, f_max) or (p_min, p_max).")
    if f_min >= f_max:
        raise ValueError(f"f_min ({f_min}) must be < f_max ({f_max}).")

    baseline = float(np.max(t) - np.min(t))
    if baseline <= 0.0:
        raise ValueError("Time array spans zero baseline.")

    if df is None:
        df = float(q_min) / (oversample * baseline)
    return np.arange(f_min, f_max, df)


# --------------------------------------------------------------------------- #
# Duration helper (Eqs. 6-10)                                                  #
# --------------------------------------------------------------------------- #
def _transit_fraction(
    phi: Array, i1: int, i2: int, n: int
) -> tuple[float, float]:
    """Fractional transit duration and ingress phase for the block ``i1..i2``.

    ``phi`` must be sorted ascending. Returns ``(frac, phi_ingress)`` where
    ``frac`` is the duration as a fraction of the period in ``[0, 1)``.
    """
    i1m = (i1 - 1) % n
    i2p = (i2 + 1) % n
    p_i1, p_i1m = phi[i1], phi[i1m]
    p_i2, p_i2p = phi[i2], phi[i2p]

    if p_i1m < p_i1:  # neighbour does not wrap past phase 0
        phi_ing = 0.5 * (p_i1m + p_i1)
    else:
        phi_ing = 0.5 * (p_i1m + p_i1) - 0.5

    if p_i2 < p_i2p:  # neighbour does not wrap past phase 1
        phi_eg = 0.5 * (p_i2 + p_i2p)
    else:
        phi_eg = 0.5 * (p_i2 + p_i2p) + 0.5

    frac = (phi_eg - phi_ing) % 1.0
    return frac, phi_ing % 1.0


# --------------------------------------------------------------------------- #
# Single-band SBLS                                                             #
# --------------------------------------------------------------------------- #
def _single_period(
    phi_sorted: Array,
    wx_sorted: Array,
    w_sorted: Array,
    period: float,
    q_max: float,
    min_points: int,
) -> tuple[float, float, float, float]:
    """Best ``(SR, t0, duration, depth)`` for one trial period.

    The block of in-transit points is a contiguous run (with wrap-around) in
    phase-sorted order, starting at ``i1`` and grown by adding successive
    points until the duration exceeds ``q_max``.
    """
    n = phi_sorted.shape[0]
    best_sr = 0.0
    best_t0 = 0.0
    best_dur = 0.0
    best_depth = 0.0

    for i1 in range(n):
        s = 0.0
        r = 0.0
        cnt = 0
        for k in range(n - 1):
            i2 = (i1 + k) % n
            s += wx_sorted[i2]
            r += w_sorted[i2]
            cnt += 1
            frac, phi_ing = _transit_fraction(phi_sorted, i1, i2, n)
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
                phi_mid = (phi_ing + 0.5 * frac) % 1.0
                best_t0 = phi_mid * period
    return best_sr, best_t0, best_dur, best_depth


def sparse_bls_reference(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    q_max: float = 0.15,
    min_points: int = 3,
) -> BLSResult:
    """Single-band Sparse BLS (pure-Python reference).

    This is the pure-Python reference implementation. Use :func:`sparse_bls` for
    production; this version is ~100× slower and exists for correctness checks.

    Parameters
    ----------
    t, y, dy:
        Times (days), magnitudes (or fluxes), and 1-sigma uncertainties.
    frequencies:
        Trial frequencies (1/day).
    q_max:
        Maximum transit duration as a fraction of the period.
    min_points:
        Minimum number of in-transit points required (paper uses 3).
    """
    t = np.asarray(t, dtype=float)
    w_hat, x_tilde, _ = preprocess(y, dy)
    wx = w_hat * x_tilde
    yy = float(np.sum(w_hat * x_tilde ** 2))  # weighted variance (flat chi^2)

    freqs = np.asarray(frequencies, dtype=float)
    power = np.zeros(freqs.shape[0])
    best = {"sr": 0.0, "t0": 0.0, "dur": 0.0, "depth": 0.0, "idx": 0}

    for idx, f in enumerate(freqs):
        period = 1.0 / f
        phi = (t % period) / period
        order = np.argsort(phi)
        sr, t0, dur, depth = _single_period(
            phi[order], wx[order], w_hat[order], period, q_max, min_points
        )
        power[idx] = sr
        if sr > best["sr"]:
            best.update(sr=sr, t0=t0, dur=dur, depth=depth, idx=idx)

    bidx = int(best["idx"])
    return BLSResult(
        frequency=freqs,
        power=variance_explained(power, yy),
        best_frequency=float(freqs[bidx]),
        best_period=float(1.0 / freqs[bidx]),
        best_t0=float(best["t0"]),
        best_duration=float(best["dur"]),
        best_depth=float(best["depth"]),
        best_power=float(variance_explained(np.array([best["sr"]]), yy)[0]),
    )


# --------------------------------------------------------------------------- #
# Multiband SBLS                                                               #
# --------------------------------------------------------------------------- #
def _multiband_period(
    phi_sorted: Array,
    wx_sorted: Array,
    w_sorted: Array,
    band_sorted: Array,
    n_bands: int,
    period: float,
    q_max: float,
    min_points: int,
    band_w: Array,
) -> tuple[float, float, float, Array]:
    """Best combined ``(SR, t0, duration, depths[n_bands])`` for one period.

    A single contiguous phase block (shared across bands) is grown over the
    *merged* phase-sorted samples; each band accumulates its own ``s_b, r_b``
    and the combined statistic is ``sqrt(sum_b s_b^2 / (r_b (1 - r_b)))``.

    ``min_points`` is the minimum *total* number of in-transit points in the
    window; any band with at least one in-transit point (and ``0 < r_b < 1``)
    contributes. This keeps deep but sparsely-sampled bands (e.g. ``u``) in
    play, which is the whole point of a multiband search for WD-MD eclipses.
    """
    n = phi_sorted.shape[0]
    best_sr = 0.0
    best_t0 = 0.0
    best_dur = 0.0
    best_depths = np.zeros(n_bands)

    for i1 in range(n):
        s = np.zeros(n_bands)
        r = np.zeros(n_bands)
        cnt = np.zeros(n_bands, dtype=np.int64)
        total = 0
        for k in range(n - 1):
            i2 = (i1 + k) % n
            b = band_sorted[i2]
            s[b] += wx_sorted[i2]
            r[b] += w_sorted[i2]
            cnt[b] += 1
            total += 1
            frac, phi_ing = _transit_fraction(phi_sorted, i1, i2, n)
            if frac >= q_max:
                break
            if total < min_points:
                continue
            sr2 = 0.0
            depths = np.zeros(n_bands)
            ok = False
            for bb in range(n_bands):
                if cnt[bb] < 1:
                    continue
                denom = r[bb] * (1.0 - r[bb])
                if denom <= 0.0:
                    continue
                sr2 += band_w[bb] * s[bb] * s[bb] / denom
                depths[bb] = s[bb] / denom
                ok = True
            if not ok:
                continue
            if sr2 > best_sr:
                best_sr = sr2
                best_depths = depths
                best_dur = frac * period
                phi_mid = (phi_ing + 0.5 * frac) % 1.0
                best_t0 = phi_mid * period
    return best_sr, best_t0, best_dur, best_depths


def multiband_sparse_bls_reference(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    q_max: float = 0.15,
    min_points: int = 3,
) -> BLSResult:
    """Multiband Sparse BLS (pure-Python reference).

    This is the pure-Python reference implementation. Use :func:`multiband_sparse_bls`
    for production; this version is ~100× slower and exists for correctness checks.

    Bands combine with matched-filter weighting (each band weighted by its total
    inverse variance ``W_b``); ``power`` is ``Delta chi^2 / chi^2_flat`` in
    ``[0, 1]``, maximised at the best period.

    Parameters
    ----------
    bands:
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch and duration.
    frequencies, q_max, min_points:
        As in :func:`sparse_bls_reference`.
    """
    labels = tuple(bands.keys())
    n_bands = len(labels)

    t_parts: list[Array] = []
    wx_parts: list[Array] = []
    w_parts: list[Array] = []
    band_parts: list[Array] = []
    w_totals: list[float] = []
    chi2_flat = 0.0
    for bi, label in enumerate(labels):
        t_b, y_b, dy_b = bands[label]
        w_hat, x_tilde, _ = preprocess(y_b, dy_b)
        w_raw = 1.0 / np.asarray(dy_b, dtype=float) ** 2
        w_totals.append(float(w_raw.sum()))
        chi2_flat += float(np.sum(w_raw * x_tilde ** 2))
        t_parts.append(np.asarray(t_b, dtype=float))
        wx_parts.append(w_hat * x_tilde)
        w_parts.append(w_hat)
        band_parts.append(np.full(len(t_b), bi, dtype=np.int64))

    t_all = np.concatenate(t_parts)
    wx_all = np.concatenate(wx_parts)
    w_all = np.concatenate(w_parts)
    band_all = np.concatenate(band_parts)
    band_w = np.asarray(w_totals, dtype=float)

    freqs = np.asarray(frequencies, dtype=float)
    power = np.zeros(freqs.shape[0])
    best = {
        "sr": 0.0,
        "t0": 0.0,
        "dur": 0.0,
        "depths": np.zeros(n_bands),
        "idx": 0,
    }

    for idx, f in enumerate(freqs):
        period = 1.0 / f
        phi = (t_all % period) / period
        order = np.argsort(phi)
        sr, t0, dur, depths = _multiband_period(
            phi[order],
            wx_all[order],
            w_all[order],
            band_all[order],
            n_bands,
            period,
            q_max,
            min_points,
            band_w,
        )
        power[idx] = sr
        if sr > cast(float, best["sr"]):
            best.update(sr=sr, t0=t0, dur=dur, depths=depths, idx=idx)

    bidx = cast(int, best["idx"])
    return BLSResult(
        frequency=freqs,
        power=variance_explained(power, chi2_flat),
        best_frequency=float(freqs[bidx]),
        best_period=float(1.0 / freqs[bidx]),
        best_t0=cast(float, best["t0"]),
        best_duration=cast(float, best["dur"]),
        best_depth=cast(np.ndarray, best["depths"]),
        best_power=float(variance_explained(np.array([cast(float, best["sr"])]), chi2_flat)[0]),
        bands=labels,
    )


def _meebls_period(
    t: Array,
    wx: Array,
    w: Array,
    band: Array,
    n_bands: int,
    period: float,
    nb: int,
    kmi: int,
    kma: int,
    min_points: int,
    band_w: Array,
) -> tuple[float, float, float, Array]:
    """Best combined ``(SR, t0, duration, depths[n_bands])`` for one period.

    Binned multiband search: each band is binned onto the same ``nb`` phase bins,
    and a shared window of bins is grown; the combined statistic is
    ``sqrt(sum_b scale_b * s_b^2 / (r_b (1 - r_b)))``.
    """
    n = t.shape[0]
    ybin = np.zeros((n_bands, nb))
    rbin = np.zeros((n_bands, nb))
    cbin = np.zeros((n_bands, nb), dtype=np.int64)

    phi = (t % period) / period
    ib = np.minimum((nb * phi).astype(np.int64), nb - 1)
    for i in range(n):
        b = band[i]
        ybin[b, ib[i]] += wx[i]
        rbin[b, ib[i]] += w[i]
        cbin[b, ib[i]] += 1

    best_sr = 0.0
    best_t0 = 0.0
    best_dur = 0.0
    best_depths = np.zeros(n_bands)

    for i1 in range(nb):
        s = np.zeros(n_bands)
        r = np.zeros(n_bands)
        cnt = np.zeros(n_bands, dtype=np.int64)
        total = 0
        for width in range(1, kma + 1):
            jj = (i1 + width - 1) % nb
            s += ybin[:, jj]
            r += rbin[:, jj]
            cnt += cbin[:, jj]
            total += int(cbin[:, jj].sum())
            if width < kmi:
                continue
            if total < min_points:
                continue
            sr2 = 0.0
            depths = np.zeros(n_bands)
            ok = False
            for bb in range(n_bands):
                if cnt[bb] < 1:
                    continue
                denom = r[bb] * (1.0 - r[bb])
                if denom <= 0.0:
                    continue
                sr2 += band_w[bb] * s[bb] * s[bb] / denom
                depths[bb] = s[bb] / denom
                ok = True
            if not ok:
                continue
            if sr2 > best_sr:
                best_sr = sr2
                best_depths = depths
                best_dur = width / nb * period
                phi_mid = ((i1 + 0.5 * width) / nb) % 1.0
                best_t0 = phi_mid * period
    return best_sr, best_t0, best_dur, best_depths


def multiband_eebls_reference(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int = 300,
    qmin: float = 0.01,
    qmax: float = 0.10,
    min_points: int = 3,
) -> BLSResult:
    """Binned multiband BLS (pure-Python reference).

    This is the pure-Python reference implementation. Use :func:`multiband_eebls`
    for production; this version is ~100× slower and exists for correctness checks.

    Like :func:`multiband_sparse_bls_reference` (matched-filter weighting, returns
    ``Delta chi^2 / chi^2_flat``) but the phase-folded data of each band is binned
    onto ``nbins`` phase bins and the box is searched over bins.
    """
    labels = tuple(bands.keys())
    n_bands = len(labels)
    kmi = max(1, int(qmin * nbins))
    kma = min(nbins, int(qmax * nbins) + 1)

    t_parts: list[Array] = []
    wx_parts: list[Array] = []
    w_parts: list[Array] = []
    band_parts: list[Array] = []
    w_totals: list[float] = []
    chi2_flat = 0.0
    for bi, label in enumerate(labels):
        t_b, y_b, dy_b = bands[label]
        w_hat, x_tilde, _ = preprocess(y_b, dy_b)
        w_raw = 1.0 / np.asarray(dy_b, dtype=float) ** 2
        w_totals.append(float(w_raw.sum()))
        chi2_flat += float(np.sum(w_raw * x_tilde ** 2))
        t_parts.append(np.asarray(t_b, dtype=float))
        wx_parts.append(w_hat * x_tilde)
        w_parts.append(w_hat)
        band_parts.append(np.full(len(t_b), bi, dtype=np.int64))

    t_all = np.concatenate(t_parts)
    wx_all = np.concatenate(wx_parts)
    w_all = np.concatenate(w_parts)
    band_all = np.concatenate(band_parts)
    band_w = np.asarray(w_totals, dtype=float)

    freqs = np.asarray(frequencies, dtype=float)
    power = np.zeros(freqs.shape[0])
    best = {"sr": 0.0, "t0": 0.0, "dur": 0.0, "depths": np.zeros(n_bands), "idx": 0}

    for idx, f in enumerate(freqs):
        period = 1.0 / f
        sr, t0, dur, depths = _meebls_period(
            t_all, wx_all, w_all, band_all, n_bands, period, nbins, kmi, kma,
            min_points, band_w,
        )
        power[idx] = sr
        if sr > cast(float, best["sr"]):
            best.update(sr=sr, t0=t0, dur=dur, depths=depths, idx=idx)

    bidx = cast(int, best["idx"])
    return BLSResult(
        frequency=freqs,
        power=variance_explained(power, chi2_flat),
        best_frequency=float(freqs[bidx]),
        best_period=float(1.0 / freqs[bidx]),
        best_t0=cast(float, best["t0"]),
        best_duration=cast(float, best["dur"]),
        best_depth=cast(np.ndarray, best["depths"]),
        best_power=float(variance_explained(np.array([cast(float, best["sr"])]), chi2_flat)[0]),
        bands=labels,
    )


def eebls_reference(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    qmin: float = 0.01,
    qmax: float = 0.10,
    min_points: int = 3,
) -> BLSResult:
    """Single-band binned BLS (pure-Python reference).

    .. note::
        This is the pure-Python reference implementation. Use :func:`eebls` for
        production; this version is ~100× slower and exists for correctness checks.

    Parameters
    ----------
    t, y, dy:
        Times (days), magnitudes (or fluxes), and 1-sigma uncertainties.
    frequencies:
        Trial frequencies (1/day).
    nbins:
        Number of phase bins. Chosen automatically via :func:`auto_nbins` if ``None``.
    qmin, qmax:
        Minimum and maximum transit duration as a fraction of the period.
    min_points:
        Minimum number of in-transit points required.
    """
    if nbins is None:
        nbins = auto_nbins(qmin)
    kmi = max(1, int(qmin * nbins))
    kma = min(nbins, int(qmax * nbins) + 1)

    t = np.asarray(t, dtype=float)
    w_hat, x_tilde, _ = preprocess(y, dy)
    yy = float(np.sum(w_hat * x_tilde ** 2))
    wx = w_hat * x_tilde

    freqs = np.asarray(frequencies, dtype=float)
    power = np.zeros(freqs.shape[0])
    best = {"sr": 0.0, "t0": 0.0, "dur": 0.0, "depth": 0.0, "idx": 0}

    for idx, f in enumerate(freqs):
        period = 1.0 / f
        phi = (t % period) / period
        ib = np.minimum((nbins * phi).astype(np.int64), nbins - 1)
        ybin = np.zeros(nbins)
        rbin = np.zeros(nbins)
        cbin = np.zeros(nbins, dtype=np.int64)
        np.add.at(ybin, ib, wx)
        np.add.at(rbin, ib, w_hat)
        np.add.at(cbin, ib, 1)

        best_sr = 0.0
        best_t0 = 0.0
        best_dur = 0.0
        best_depth = 0.0
        for i1 in range(nbins):
            s = 0.0
            r = 0.0
            cnt = 0
            for width in range(1, kma + 1):
                jj = (i1 + width - 1) % nbins
                s += ybin[jj]
                r += rbin[jj]
                cnt += int(cbin[jj])
                if width < kmi:
                    continue
                if cnt < min_points:
                    continue
                denom = r * (1.0 - r)
                if denom <= 0.0:
                    continue
                sr = s * s / denom
                if sr > best_sr:
                    best_sr = sr
                    best_depth = s / denom
                    best_dur = width / nbins * period
                    phi_mid = ((i1 + 0.5 * width) / nbins) % 1.0
                    best_t0 = phi_mid * period

        power[idx] = best_sr
        if best_sr > best["sr"]:
            best.update(sr=best_sr, t0=best_t0, dur=best_dur, depth=best_depth, idx=idx)

    bidx = int(best["idx"])
    return BLSResult(
        frequency=freqs,
        power=variance_explained(power, yy),
        best_frequency=float(freqs[bidx]),
        best_period=float(1.0 / freqs[bidx]),
        best_t0=float(best["t0"]),
        best_duration=float(best["dur"]),
        best_depth=float(best["depth"]),
        best_power=float(variance_explained(np.array([best["sr"]]), yy)[0]),
    )


def coadd_bands(
    bands: Mapping[str, tuple[Array, Array, Array]]
) -> tuple[Array, Array, Array]:
    """Naive co-add baseline: stack all bands after per-band mean subtraction.

    Each band is shifted to zero weighted mean so a single-band SBLS can be run
    on the pooled series. This is the "regular SBLS" straw-man to beat.
    """
    t_parts: list[Array] = []
    y_parts: list[Array] = []
    dy_parts: list[Array] = []
    for t_b, y_b, dy_b in bands.values():
        _, x_tilde, _ = preprocess(y_b, dy_b)
        t_parts.append(np.asarray(t_b, dtype=float))
        y_parts.append(x_tilde)
        dy_parts.append(np.asarray(dy_b, dtype=float))
    return (
        np.concatenate(t_parts),
        np.concatenate(y_parts),
        np.concatenate(dy_parts),
    )

