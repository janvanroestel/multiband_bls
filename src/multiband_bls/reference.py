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
import warnings
from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt

from .periodogram import BLSResult

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Preprocessing and grids                                                      #
# --------------------------------------------------------------------------- #
def _check_weights(dy: Array, where: str) -> None:
    """Raise unless the inverse-variance weights ``1/dy**2`` are usable.

    Finite, positive ``dy`` can still over/underflow (``1e-200`` -> ``w = inf``,
    ``1e200`` -> ``w = 0``), which would silently turn the periodogram into NaN.
    """
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        w = 1.0 / dy ** 2
        w_sum = w.sum()
    if not (np.all(np.isfinite(w)) and np.all(w > 0.0)
            and np.isfinite(w_sum) and w_sum > 0.0):
        raise ValueError(
            f"dy{where} spans [{dy.min():.3g}, {dy.max():.3g}], which over/underflows "
            "the inverse-variance weights 1/dy**2; rescale y and dy."
        )


def _check_signal(y: Array, dy: Array, where: str) -> None:
    """Raise unless the preprocessed signal and flat-model chi^2 are finite.

    Finite ``y`` can still overflow once mean-subtracted and squared (e.g.
    ``|y| ~ 1e308``), which would silently turn ``power`` into NaN.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        w_hat, x_tilde, mu = preprocess(y, dy)
        wx = w_hat * x_tilde
        chi2 = float(np.sum((x_tilde / dy) ** 2))
        yy = float(np.sum(w_hat * x_tilde ** 2))
    if not (np.isfinite(mu) and np.all(np.isfinite(x_tilde)) and np.all(np.isfinite(wx))
            and np.isfinite(chi2) and np.isfinite(yy)):
        raise ValueError(
            f"y{where} spans [{y.min():.3g}, {y.max():.3g}], which overflows the "
            "mean-subtracted signal or its chi^2; rescale y and dy."
        )


def _validate_light_curve(
    t: Array, y: Array, dy: Array, label: str | None = None
) -> tuple[Array, Array, Array]:
    """Coerce ``(t, y, dy)`` to 1-D float64 arrays, raising ``ValueError`` on bad input.

    Rejects non-1-D or empty arrays, unequal lengths, non-finite values,
    ``dy <= 0`` and ``dy`` too extreme for ``1/dy**2``. Bad points are reported,
    never silently dropped.
    """
    where = f" in band {label!r}" if label is not None else ""
    arrays = {
        "t": np.asarray(t, dtype=float),
        "y": np.asarray(y, dtype=float),
        "dy": np.asarray(dy, dtype=float),
    }
    for name, a in arrays.items():
        if a.ndim != 1:
            raise ValueError(f"{name}{where} must be 1-D, got shape {a.shape}.")
    lengths = {name: a.shape[0] for name, a in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"t, y and dy{where} must have equal lengths, got {lengths}.")
    if lengths["t"] == 0:
        raise ValueError(f"Empty light curve{where}.")
    for name, a in arrays.items():
        n_bad = int(np.count_nonzero(~np.isfinite(a)))
        if n_bad:
            raise ValueError(
                f"{name}{where} contains {n_bad} non-finite value(s); remove them "
                "(e.g. with np.isfinite) before searching."
            )
    n_nonpos = int(np.count_nonzero(arrays["dy"] <= 0.0))
    if n_nonpos:
        raise ValueError(f"dy{where} contains {n_nonpos} value(s) <= 0.")
    _check_weights(arrays["dy"], where)
    _check_signal(arrays["y"], arrays["dy"], where)
    return arrays["t"], arrays["y"], arrays["dy"]


# Beyond ~1e15 cycles a float64 phase t*f has no fractional precision left
# (resolution ~0.1 cycle), and far beyond it t*f overflows to inf -> NaN phase ->
# an invalid bin index in the compiled/GPU cores (bounds checks are off).
_MAX_PHASE_CYCLES: float = 1e15

# Explicit nbins ceiling: the binned cores allocate O(n_bands * nbins) buffers and
# sweep O(nbins * kma) windows per frequency, and take nbins as a C int. 1e5 bins
# resolves q ~ 5e-5 at 5 bins per transit, far beyond any practical search.
_MAX_NBINS: int = 100_000
# min_points is a C int in the cores. Any value above the number of points
# already rejects every window, so capping it here cannot change a result.
_MAX_C_INT: int = 2**31 - 1


def _check_phase_range(t: Array, frequencies: Array) -> None:
    """Raise unless ``max|t| * max(frequency)`` stays below :data:`_MAX_PHASE_CYCLES`.

    Uses a division so the check itself cannot overflow. ``t`` and
    ``frequencies`` must already be validated (finite, frequencies > 0).
    """
    t_max = float(np.max(np.abs(t)))
    f_max = float(np.max(frequencies))
    if t_max > _MAX_PHASE_CYCLES / f_max:
        raise ValueError(
            f"max|t| ({t_max:.3g}) * max(frequency) ({f_max:.3g}) exceeds "
            f"{_MAX_PHASE_CYCLES:.0e} cycles: phases would lose all precision or "
            "overflow. Subtract a reference epoch from t (e.g. t - t.min())."
        )


def _check_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an ``int`` if it is an integer ``>= minimum``, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return int(value)


def _check_fraction(name: str, value: object) -> float:
    """Return ``value`` as a float in the open interval ``(0, 1)``, else raise."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in (0, 1), got {value!r}.") from None
    if not (np.isfinite(v) and 0.0 < v < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {value!r}.")
    return v


def _validate_search_params(
    frequencies: Array,
    q_max: float,
    min_points: int,
    q_min: float | None = None,
    nbins: int | None = None,
) -> tuple[Array, int, int | None]:
    """Validate the search parameters shared by every entry point.

    Returns ``(frequencies, min_points, nbins)`` as a C-contiguous float64 array
    and validated ints (``nbins`` stays ``None`` if not given). ``min_points`` is
    capped at the C-int maximum (larger values reject every window anyway);
    ``nbins`` above ``_MAX_NBINS`` is rejected before anything is allocated. Must
    run before :func:`auto_nbins` in the binned entry points.
    """
    freqs = np.ascontiguousarray(frequencies, dtype=np.float64)
    if freqs.ndim != 1 or freqs.shape[0] == 0:
        raise ValueError(f"frequencies must be a non-empty 1-D array, got shape {freqs.shape}.")
    if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0.0):
        raise ValueError("frequencies must be finite and > 0.")
    q_max_v = _check_fraction("q_max", q_max)
    if q_min is not None:
        q_min_v = _check_fraction("q_min", q_min)
        if q_min_v > q_max_v:
            raise ValueError(f"q_min ({q_min}) must be <= q_max ({q_max}).")
    min_points_v = min(_check_int("min_points", min_points, 1), _MAX_C_INT)
    nbins_v = None if nbins is None else _check_int("nbins", nbins, 1)
    if nbins_v is not None and nbins_v > _MAX_NBINS:
        raise ValueError(f"nbins must be <= {_MAX_NBINS}, got {nbins_v}.")
    return freqs, min_points_v, nbins_v


def preprocess(y: Array, dy: Array) -> tuple[Array, Array, float]:
    """Return ``(w_hat, x_tilde, mu)`` from Eqs. 1-2.

    ``w_hat`` are the normalised weights (sum to 1) and ``x_tilde`` is the
    weighted-mean-subtracted signal.

    Parameters
    ----------
    y :
        Magnitudes or fluxes, 1-D float array.
    dy :
        1-sigma uncertainties, 1-D float array, same length as ``y``.
    """
    y = np.asarray(y, dtype=float)
    w = 1.0 / np.asarray(dy, dtype=float) ** 2
    W = w.sum()
    w_hat = w / W
    if y.size and np.all(y == y[0]):
        # Exactly constant signal: sum(w_hat * y) = y[0] * (1 +- eps) would leave
        # a ~1e-15 residue in x_tilde and hence a spurious, noise-driven power.
        mu = float(y[0])
    else:
        mu = float(np.sum(w_hat * y))
    x_tilde = y - mu
    return w_hat, x_tilde, mu


def _merge_band_arrays(
    bands: Mapping[str, tuple[Array, Array, Array]]
) -> tuple[Array, Array, Array, Array, tuple[str, ...], Array, float]:
    """Preprocess each band and concatenate into all-band arrays.

    Returns ``(t_all, wx_all, w_all, band_all, labels, band_w, chi2_flat)``
    where ``wx = w_hat * x_tilde`` uses each band's own normalised weights,
    ``band_all`` labels each point with its band index, ``band_w[b] = sum_i
    1/sigma^2`` per band, and ``chi2_flat = sum_b sum_i (x_tilde/sigma)^2`` is
    the total flat-model chi^2. Shared by the multiband reference entry points
    and (wrapped with contiguity/dtype coercion) by :func:`api._merge_bands`.
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
        w_raw = 1.0 / np.asarray(dy_b, dtype=float) ** 2
        w_totals.append(float(w_raw.sum()))
        chi2_flat += float(np.sum(w_raw * x_tilde ** 2))
        t_parts.append(np.asarray(t_b, dtype=float))
        wx_parts.append(w_hat * x_tilde)
        w_parts.append(w_hat)
        band_parts.append(np.full(len(t_b), bi, dtype=np.int64))
    return (
        np.concatenate(t_parts),
        np.concatenate(wx_parts),
        np.concatenate(w_parts),
        np.concatenate(band_parts),
        labels,
        np.asarray(w_totals, dtype=float),
        chi2_flat,
    )


def _prepare_bands(
    bands: Mapping[str, tuple[Array, Array, Array]], rescale_errors: bool = False
) -> tuple[dict[str, tuple[Array, Array, Array]], Array]:
    """Validate every band and optionally rescale its errors.

    The single validation/rescaling path for all multiband entry points. Returns
    ``(bands, chi2_red)``: the validated (and, if ``rescale_errors``, rescaled)
    bands mapping, and the per-band flat-model reduced chi^2
    ``chi2_red[b] = sum_i (x_tilde/sigma)^2 / (N_b - 1)`` computed from the
    *original* ``dy`` (NaN for a single-point band).

    With ``rescale_errors``, each band's ``dy`` is multiplied by
    ``sqrt(chi2_red[b])`` so its flat-model chi^2/dof becomes 1. A band whose
    ``chi2_red`` is non-finite or ``<= 0`` (one point, or all ``y`` identical) keeps
    its original ``dy`` and a ``UserWarning`` names it.
    """
    if len(bands) == 0:
        raise ValueError("bands must contain at least one band.")
    prepared: dict[str, tuple[Array, Array, Array]] = {}
    chi2_red = np.full(len(bands), np.nan)
    skipped: list[str] = []
    for bi, (label, (t_b, y_b, dy_b)) in enumerate(bands.items()):
        t_v, y_v, dy_v = _validate_light_curve(t_b, y_b, dy_b, label=label)
        _, x_tilde, _ = preprocess(y_v, dy_v)
        if t_v.shape[0] > 1:
            chi2_red[bi] = float(np.sum((x_tilde / dy_v) ** 2)) / (t_v.shape[0] - 1)
        if rescale_errors:
            # All-identical y gives chi2_red ~ 1e-28 from rounding, not exactly 0,
            # so test for zero scatter directly as well.
            zero_scatter = bool(np.all(y_v == y_v[0]))
            if not zero_scatter and np.isfinite(chi2_red[bi]) and chi2_red[bi] > 0.0:
                dy_v = dy_v * np.sqrt(chi2_red[bi])
                _check_weights(dy_v, f" in band {label!r} (after rescaling)")
                _check_signal(y_v, dy_v, f" in band {label!r} (after rescaling)")
            else:
                skipped.append(str(label))
        prepared[label] = (t_v, y_v, dy_v)
    if skipped:
        warnings.warn(
            f"rescale_errors: band(s) {skipped} have no usable flat-model chi^2/dof "
            "(single point or zero scatter); their dy were left unchanged.",
            UserWarning,
            stacklevel=3,
        )
    return prepared, chi2_red


def auto_nbins(q_min: float, n_res: int = 5) -> int:
    """Return the recommended number of phase bins for a given minimum transit fraction.

    Ensures at least ``n_res`` bins span the narrowest transit (``q_min``),
    capped between 50 and 500 to keep compute tractable.

    Parameters
    ----------
    q_min :
        Minimum transit duration as a fraction of the period.
    n_res :
        Minimum number of bins inside the narrowest transit.

    Raises
    ------
    ValueError
        If ``q_min`` is not finite and ``> 0``.

    Warns
    -----
    UserWarning
        If the 500-bin cap means fewer than ``n_res`` bins span ``q_min``
        (i.e. ``q_min < n_res / 500``); pass an explicit ``nbins`` to override.
    """
    if not (np.isfinite(q_min) and q_min > 0.0):
        raise ValueError(f"q_min must be finite and > 0, got {q_min!r}.")
    wanted = round(n_res / q_min)
    if wanted > 500:
        warnings.warn(
            f"auto_nbins: q_min={q_min} needs {wanted} bins for {n_res} bins per "
            f"transit, but nbins is capped at 500, giving only {500 * q_min:.2g} "
            "bins across the narrowest transit. Pass an explicit nbins to override.",
            UserWarning,
            stacklevel=2,
        )
    return int(np.clip(wanted, 50, 500))


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

    Parameters
    ----------
    t :
        Observation times (days); used only to compute the baseline.
    f_min, f_max :
        Frequency search bounds (1/day). Mutually exclusive with ``p_min``/``p_max``.
    p_min, p_max :
        Period search bounds (days). Mutually exclusive with ``f_min``/``f_max``.
        Note the inversion: ``p_min → f_max`` and ``p_max → f_min``.
    df :
        Frequency step (1/day). Defaults to ``q_min / (oversample * baseline)``.
    oversample :
        Number of frequency steps per transit peak width when ``df`` is auto-computed.
    q_min :
        Shortest fractional transit duration to resolve; sets the default ``df``.
        Should match the ``q_min`` passed to the search function.

    Raises
    ------
    ValueError
        If conflicting bounds are given, if a bound, ``df``, ``oversample`` or
        ``q_min`` is non-finite or non-positive, if ``f_min >= f_max``, or if the
        time array is not finite or spans zero baseline.
    """
    def _positive(name: str, value: float) -> float:
        v = float(value)
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value!r}.")
        return v

    # Convert period bounds → frequency bounds (note the inversion).
    period_given = p_min is not None or p_max is not None
    freq_given = f_min is not None or f_max is not None
    if period_given and freq_given:
        raise ValueError("Provide either (p_min, p_max) or (f_min, f_max), not both.")
    if p_min is not None:
        f_max = 1.0 / _positive("p_min", p_min)
    if p_max is not None:
        f_min = 1.0 / _positive("p_max", p_max)

    if f_min is None or f_max is None:
        raise ValueError("Specify either (f_min, f_max) or (p_min, p_max).")
    f_min = _positive("f_min", f_min)
    f_max = _positive("f_max", f_max)
    if f_min >= f_max:
        raise ValueError(f"f_min ({f_min}) must be < f_max ({f_max}).")
    # validated even when df is given explicitly, so bad values never pass silently
    q_min = _positive("q_min", q_min)
    oversample = _positive("oversample", oversample)

    t = np.asarray(t, dtype=float)
    if t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("t must be a non-empty array of finite times.")
    baseline = float(np.max(t) - np.min(t))
    if baseline <= 0.0:
        raise ValueError("Time array spans zero baseline.")

    if df is None:
        df = q_min / (oversample * baseline)
    else:
        df = _positive("df", df)
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

    Unlike the binned variants there is no minimum duration (``q_min``):
    ``min_points`` is the only guard against ultra-narrow boxes.
    """
    t, y, dy = _validate_light_curve(t, y, dy)
    freqs, min_points, _ = _validate_search_params(frequencies, q_max, min_points)
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
    band_n: npt.NDArray[np.int64],
) -> tuple[float, float, float, Array, int]:
    """Best combined ``(SR, t0, duration, depths[n_bands], k)`` for one period.

    ``k`` is the number of bands contributing to the best window (0 if no
    window was selected); depths of non-contributing bands are NaN.

    A single contiguous phase block (shared across bands) is grown over the
    *merged* phase-sorted samples; each band accumulates its own ``s_b, r_b``
    and the combined statistic is
    ``sqrt(sum_b band_w[b] * s_b^2 / (r_b (1 - r_b)))``.

    ``min_points`` is the minimum *total* number of in-transit points in the
    window; any band with at least one in-transit point (and ``0 < r_b < 1``)
    contributes. This keeps deep but sparsely-sampled bands (e.g. ``u``) in
    play, which is the whole point of a multiband search for WD-MD eclipses.
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
            if phi_sorted[(i2 + 1) % n] == phi_sorted[i2]:
                continue  # egress edge would split identical phases
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
                if denom <= 0.0:
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
                best_dur = frac * period
                phi_mid = (phi_ing + 0.5 * frac) % 1.0
                best_t0 = phi_mid * period
    return best_sr, best_t0, best_dur, best_depths, best_k


def multiband_sparse_bls_reference(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    q_max: float = 0.15,
    min_points: int = 3,
    rescale_errors: bool = False,
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
    rescale_errors:
        Rescale each band's ``dy`` so its flat-model chi^2/dof is 1 (see
        :class:`BLSResult.band_chi2_red`).
    """
    freqs, min_points, _ = _validate_search_params(frequencies, q_max, min_points)
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
            q_max,
            min_points,
            band_w,
            band_n,
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
                if denom <= 0.0:
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
    """
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    kmi = max(1, int(q_min * nbins))
    kma = min(nbins, int(q_max * nbins) + 1)

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
            min_points, band_w, band_n,
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
    """
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    kmi = max(1, int(q_min * nbins))
    kma = min(nbins, int(q_max * nbins) + 1)

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

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band is mean-subtracted before stacking.
    """
    bands, _ = _prepare_bands(bands)
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

