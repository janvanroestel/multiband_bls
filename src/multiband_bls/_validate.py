"""Input validation shared by every search entry point (no search logic)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._weights import preprocess

Array = npt.NDArray[np.float64]


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
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{name} must be a number in (0, 1), got {value!r}.")
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



_SIGNS = {"both": 0.0, "positive": 1.0, "negative": -1.0}


def _check_sign(sign: str, name: str) -> float:
    """Map ``sign`` ('both' | 'positive' | 'negative') to the kernels' -1/0/+1."""
    if sign not in _SIGNS:
        raise ValueError(f"{name}: sign must be one of {sorted(_SIGNS)}, got {sign!r}.")
    return _SIGNS[sign]

