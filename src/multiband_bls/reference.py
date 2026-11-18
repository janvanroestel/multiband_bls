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
import numpy.typing as npt

from ._prep import (
    _merge_band_arrays,
    _prepare_bands,
    auto_nbins,
    coadd_bands,
    preprocess,
    variance_explained,
    width_bounds,
)
from ._validate import (  # noqa: F401
    _MAX_C_INT,
    _MAX_NBINS,
    _MAX_PHASE_CYCLES,
    _SIGNS,
    _check_fraction,
    _check_int,
    _check_phase_range,
    _check_sign,
    _check_signal,
    _check_weights,
    _validate_light_curve,
    _validate_search_params,
)
from .grids import build_frequency_grid
from .result import BLSResult

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]


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

    # Wrap-around is decided by index (phi is sorted), not by comparing values,
    # so tied phases cannot be mistaken for a wrap.
    if i1 > 0:  # neighbour does not wrap past phase 0
        phi_ing = 0.5 * (p_i1m + p_i1)
    else:
        phi_ing = 0.5 * (p_i1m + p_i1) - 0.5

    if i2p != 0:  # neighbour does not wrap past phase 1
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
    q_min: float,
    q_max: float,
    min_points: int,
    sign: float = 0.0,
) -> tuple[float, float, float, float]:
    """Best ``(SR, t0, duration, depth)`` for one trial period.

    The block of in-transit points is a contiguous run (with wrap-around) in
    phase-sorted order, starting at ``i1`` and grown by adding successive
    points until the duration exceeds ``q_max``. Windows shorter than ``q_min``
    or whose signed flux sum has the wrong ``sign`` (-1/0/+1; 0 keeps both) are
    skipped.
    """
    n = phi_sorted.shape[0]
    best_sr = 0.0
    best_t0 = 0.0
    best_dur = 0.0
    best_depth = 0.0

    for i1 in range(n):
        if phi_sorted[(i1 - 1) % n] == phi_sorted[i1]:
            continue  # ingress edge would split identical phases
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
            if phi_sorted[(i2 + 1) % n] == phi_sorted[i2]:
                continue  # egress edge would split identical phases
            if frac < q_min or cnt < min_points:
                continue
            if sign * s < 0.0:
                continue  # wrong-signed window (sign = 0: keep both)
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
    q_min: float = 0.01,
    q_max: float = 0.15,
    min_points: int = 3,
    sign: str = "both",
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
    q_min, q_max:
        Minimum and maximum transit duration as a fraction of the period. The
        duration of a window is measured between the midpoints of the gaps to its
        neighbouring points. Note the default ``q_max`` (0.15) differs from the
        binned variants' (0.10).
    min_points:
        Minimum number of in-transit points required (paper uses 3).
    sign:
        ``"both"`` (default), ``"positive"`` or ``"negative"``: only windows whose
        signed weighted flux sum has that sign contribute (see :func:`eebls`).
    """
    t, y, dy = _validate_light_curve(t, y, dy)
    sg = _check_sign(sign, "sparse_bls")
    freqs, min_points, _ = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min
    )
    _check_phase_range(t, freqs)
    w_hat, x_tilde, _ = preprocess(y, dy)
    wx = w_hat * x_tilde
    yy = float(np.sum(w_hat * x_tilde ** 2))  # weighted variance (flat chi^2)

    power = np.zeros(freqs.shape[0])
    best = {"sr": 0.0, "t0": 0.0, "dur": 0.0, "depth": 0.0, "idx": 0}

    for idx, f in enumerate(freqs):
        period = 1.0 / f
        phi = (t % period) / period
        order = np.argsort(phi)
        sr, t0, dur, depth = _single_period(
            phi[order], wx[order], w_hat[order], period, q_min, q_max, min_points, sg
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
    q_min: float,
    q_max: float,
    min_points: int,
    band_w: Array,
    band_n: npt.NDArray[np.int64],
    sign: float = 0.0,
) -> tuple[float, float, float, Array, int]:
    """Best combined ``(SR, t0, duration, depths[n_bands], k)`` for one period.

    ``k`` is the number of bands contributing to the best window (0 if no
    window was selected); depths of non-contributing bands are NaN.

    A single contiguous phase block (shared across bands) is grown over the
    *merged* phase-sorted samples; each band accumulates its own ``s_b, r_b``
    and the combined statistic is
    ``sqrt(sum_b band_w[b] * s_b^2 / (r_b (1 - r_b)))``.

    Adding one sample changes only its own band, so each band's contribution is
    cached and the running total updated in O(1) per extension (instead of
    re-summing all bands); full depths are extracted only for new best windows.
    The Cython core performs the same operations in the same order.

    ``min_points`` is the minimum *total* number of in-transit points in the
    window; any band with at least one in-transit point (and ``0 < r_b < 1``)
    contributes. This keeps deep but sparsely-sampled bands (e.g. ``u``) in
    play, which is the whole point of a multiband search for WD-MD eclipses.
    Windows shorter than ``q_min`` are skipped; ``sign`` (-1/0/+1) drops bands
    whose in-window signed flux sum has the wrong sign.
    """
    n = phi_sorted.shape[0]
    best_sr = 0.0
    best_t0 = 0.0
    best_dur = 0.0
    best_depths = np.full(n_bands, np.nan)
    best_k = 0

    for i1 in range(n):
        if phi_sorted[(i1 - 1) % n] == phi_sorted[i1]:
            continue  # ingress edge would split identical phases
        s = np.zeros(n_bands)
        r = np.zeros(n_bands)
        cnt = np.zeros(n_bands, dtype=np.int64)
        contrib = np.full(n_bands, -1.0)  # cached per-band term; < 0 = not contributing
        sr2 = 0.0
        kk = 0
        total = 0
        for k in range(n - 1):
            i2 = (i1 + k) % n
            b = band_sorted[i2]
            s[b] += wx_sorted[i2]
            r[b] += w_sorted[i2]
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
                    contrib[b] = band_w[b] * s[b] * s[b] / denom
                    sr2 += contrib[b]
                    kk += 1
            if kk == 0:
                sr2 = 0.0  # drop accumulated rounding once nothing contributes

            frac, phi_ing = _transit_fraction(phi_sorted, i1, i2, n)
            if frac >= q_max:
                break
            if phi_sorted[(i2 + 1) % n] == phi_sorted[i2]:
                continue  # egress edge would split identical phases
            if frac < q_min or total < min_points:
                continue
            if kk == 0:
                continue
            if sr2 > best_sr:
                best_sr = sr2
                best_k = kk
                best_dur = frac * period
                phi_mid = (phi_ing + 0.5 * frac) % 1.0
                best_t0 = phi_mid * period
                best_depths = np.full(n_bands, np.nan)
                for bb in range(n_bands):
                    if contrib[bb] >= 0.0:
                        best_depths[bb] = s[bb] / (r[bb] * (1.0 - r[bb]))
    return best_sr, best_t0, best_dur, best_depths, best_k


def multiband_sparse_bls_reference(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    q_min: float = 0.01,
    q_max: float = 0.15,
    min_points: int = 3,
    rescale_errors: bool = False,
    sign: str = "both",
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
    frequencies, q_min, q_max, min_points:
        As in :func:`sparse_bls_reference`.
    rescale_errors:
        Rescale each band's ``dy`` so its flat-model chi^2/dof is 1 (see
        :class:`BLSResult.band_chi2_red`).
    sign:
        ``"both"`` (default), ``"positive"`` or ``"negative"``; each band is judged
        separately (a band whose in-window flux sum has the wrong sign adds nothing).
    """
    sg = _check_sign(sign, "multiband_sparse_bls")
    freqs, min_points, _ = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min
    )
    bands, chi2_red = _prepare_bands(bands, rescale_errors)
    t_all, wx_all, w_all, band_all, labels, band_w, chi2_flat = _merge_band_arrays(bands)
    _check_phase_range(t_all, freqs)
    n_bands = len(labels)
    band_n = np.bincount(band_all, minlength=n_bands).astype(np.int64)

    power = np.zeros(freqs.shape[0])
    k_used = np.zeros(freqs.shape[0], dtype=np.int64)
    best = {
        "sr": 0.0,
        "t0": 0.0,
        "dur": 0.0,
        "depths": np.full(n_bands, np.nan),
        "idx": 0,
    }

    for idx, f in enumerate(freqs):
        period = 1.0 / f
        phi = (t_all % period) / period
        order = np.argsort(phi)
        sr, t0, dur, depths, k = _multiband_period(
            phi[order],
            wx_all[order],
            w_all[order],
            band_all[order],
            n_bands,
            period,
            q_min,
            q_max,
            min_points,
            band_w,
            band_n,
            sg,
        )
        power[idx] = sr
        k_used[idx] = k
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
        n_bands_used=k_used,
        band_chi2_red=chi2_red,
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
    band_n: npt.NDArray[np.int64],
    sign: float = 0.0,
) -> tuple[float, float, float, Array, int]:
    """Best combined ``(SR, t0, duration, depths[n_bands], k)`` for one period.

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
    best_depths = np.full(n_bands, np.nan)
    best_k = 0

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
            depths = np.full(n_bands, np.nan)
            kk = 0
            for bb in range(n_bands):
                # identifiable only with points both in and out of the window
                if cnt[bb] < 1 or cnt[bb] >= band_n[bb]:
                    continue
                denom = r[bb] * (1.0 - r[bb])
                if denom <= 0.0 or sign * s[bb] < 0.0:
                    continue
                sr2 += band_w[bb] * s[bb] * s[bb] / denom
                depths[bb] = s[bb] / denom
                kk += 1
            if kk == 0:
                continue
            if sr2 > best_sr:
                best_sr = sr2
                best_depths = depths
                best_k = kk
                best_dur = width / nb * period
                phi_mid = ((i1 + 0.5 * width) / nb) % 1.0
                best_t0 = phi_mid * period
    return best_sr, best_t0, best_dur, best_depths, best_k


def multiband_eebls_reference(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    rescale_errors: bool = False,
    sign: str = "both",
) -> BLSResult:
    """Binned multiband BLS (pure-Python reference).

    This is the pure-Python reference implementation. Use :func:`multiband_eebls`
    for production; this version is ~100× slower and exists for correctness checks.

    Like :func:`multiband_sparse_bls_reference` (matched-filter weighting, returns
    ``Delta chi^2 / chi^2_flat``) but the phase-folded data of each band is binned
    onto ``nbins`` phase bins and the box is searched over bins.

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch, and duration.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically via :func:`auto_nbins` if
        ``None``, matching the Cython/GPU equivalents.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required across all bands.
    rescale_errors :
        Rescale each band's ``dy`` so its flat-model chi^2/dof is 1 (see
        :class:`BLSResult.band_chi2_red`).
        sign :
        ``"both"`` (default), ``"positive"`` or ``"negative"``: only windows whose
        signed weighted flux sum has that sign contribute (see :func:`eebls`).
    """
    sg = _check_sign(sign, "multiband_eebls")
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    kmi, kma = width_bounds(nbins, q_min, q_max)

    bands, chi2_red = _prepare_bands(bands, rescale_errors)
    t_all, wx_all, w_all, band_all, labels, band_w, chi2_flat = _merge_band_arrays(bands)
    _check_phase_range(t_all, freqs)
    n_bands = len(labels)
    band_n = np.bincount(band_all, minlength=n_bands).astype(np.int64)

    power = np.zeros(freqs.shape[0])
    k_used = np.zeros(freqs.shape[0], dtype=np.int64)
    best = {"sr": 0.0, "t0": 0.0, "dur": 0.0, "depths": np.full(n_bands, np.nan), "idx": 0}

    for idx, f in enumerate(freqs):
        period = 1.0 / f
        sr, t0, dur, depths, k = _meebls_period(
            t_all, wx_all, w_all, band_all, n_bands, period, nbins, kmi, kma,
            min_points, band_w, band_n, sg,
        )
        power[idx] = sr
        k_used[idx] = k
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
        n_bands_used=k_used,
        band_chi2_red=chi2_red,
    )


def eebls_reference(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    sign: str = "both",
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
    q_min, q_max:
        Minimum and maximum transit duration as a fraction of the period.
    min_points:
        Minimum number of in-transit points required.
        sign :
        ``"both"`` (default), ``"positive"`` or ``"negative"``: only windows whose
        signed weighted flux sum has that sign contribute (see :func:`eebls`).
    """
    sg = _check_sign(sign, "eebls")
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    kmi, kma = width_bounds(nbins, q_min, q_max)

    t, y, dy = _validate_light_curve(t, y, dy)
    _check_phase_range(t, freqs)
    w_hat, x_tilde, _ = preprocess(y, dy)
    yy = float(np.sum(w_hat * x_tilde ** 2))
    wx = w_hat * x_tilde

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
                if cnt >= t.shape[0]:
                    continue  # no out-of-transit point: depth not identifiable
                denom = r * (1.0 - r)
                if sg * s < 0.0:
                    continue  # wrong-signed window (sign = 0: keep both)
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

