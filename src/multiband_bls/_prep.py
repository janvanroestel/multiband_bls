"""Preprocessing shared by the search backends: weights, band merge, widths, coadd."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from ._validate import _check_signal, _check_weights, _validate_light_curve
from ._weights import preprocess

Array = npt.NDArray[np.float64]


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


def width_bounds(nbins: int, q_min: float, q_max: float) -> tuple[int, int]:
    """Smallest and largest window width (in bins) with ``q_min <= w/nbins <= q_max``.

    Uses ceil for the lower and floor for the upper bound (with a tiny tolerance for
    float noise such as ``0.07 * 100 = 7.000000000000001``). Raises ``ValueError`` if
    no integer width fits, which means ``nbins`` is too small for the duration range.
    """
    kmi = max(1, math.ceil(q_min * nbins - 1e-9))
    kma = min(nbins, math.floor(q_max * nbins + 1e-9))
    if kma < kmi:
        raise ValueError(
            f"no window width fits q_min={q_min}, q_max={q_max} with nbins={nbins}; "
            "increase nbins or widen the duration range."
        )
    return kmi, kma


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
        If ``q_min`` is not finite and ``> 0``, or ``n_res`` is not an integer ``>= 1``.

    Warns
    -----
    UserWarning
        If the 500-bin cap means fewer than ``n_res`` bins span ``q_min``
        (i.e. ``q_min < n_res / 500``); pass an explicit ``nbins`` to override.
    """
    if not (np.isfinite(q_min) and q_min > 0.0):
        raise ValueError(f"q_min must be finite and > 0, got {q_min!r}.")
    if isinstance(n_res, (bool, np.bool_)) or not isinstance(n_res, (int, np.integer)) or n_res < 1:
        raise ValueError(f"n_res must be an integer >= 1, got {n_res!r}.")
    wanted = n_res / q_min  # float: a tiny q_min must not overflow an int conversion
    if wanted > 500:
        warnings.warn(
            f"auto_nbins: q_min={q_min} needs {wanted:.0f} bins for {n_res} bins per "
            f"transit, but nbins is capped at 500, giving only {500 * q_min:.2g} "
            "bins across the narrowest transit. Pass an explicit nbins to override.",
            UserWarning,
            stacklevel=2,
        )
    return int(np.clip(round(wanted), 50, 500)) if wanted <= 500 else 500


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


