"""Frequency-grid construction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


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
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise ValueError(f"{name} must be a number > 0, got {value!r}.")
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number > 0, got {value!r}.") from None
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
    return np.arange(f_min, f_max, df, dtype=np.float64)
