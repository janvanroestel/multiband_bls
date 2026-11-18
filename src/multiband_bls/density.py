"""Stellar-density-based search: period-dependent duration range and frequency grid.

A transit of a small companion across a star of mean density ``rho`` lasts
``T = (P / pi) * arcsin(R / a)`` with ``a / R = (G rho P^2 / 3 pi)^(1/3)``, i.e. a
duty cycle ``q = T / P`` proportional to ``P^(-2/3)`` for ``a >> R``. A search with a
fixed ``(q_min, q_max)`` therefore tests many widths no real transit can have, and its
frequency step has to be small enough for the *narrowest* box at *every* period.

The physics helpers here back :class:`multiband_bls.DensityPrior`, which splits a frequency
grid into segments with their own ``(q_min, q_max, nbins)``; :func:`density_frequency_grid`
builds the matching grid, whose step follows the (period-dependent) minimum duration, so it
is uniform in ``f^(1/3)`` (Ofir 2014) rather than in ``f``.
"""

from __future__ import annotations

__all__ = [
    "RHO_SUN",
    "central_duration_fraction",
    "density_duration_bounds",
    "density_frequency_grid",
]

import logging
import math

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]

#: Mean solar density (g/cm^3); ``density_range`` is expressed in units of this.
RHO_SUN = 1.41

# a/R = _A_OVER_R * P_day^(2/3) * (rho / RHO_SUN)^(1/3)  (Kepler's third law)
_A_OVER_R = 4.2063


def central_duration_fraction(frequency: Array | float, density: float) -> Array:
    """Duty cycle ``T/P`` of a central transit of a point-like companion.

    ``frequency`` is in 1/day and ``density`` the mean stellar density in solar
    units. Saturates at 0.5 when the orbit is inside the star (``a / R <= 1``).
    """
    f = np.asarray(frequency, dtype=np.float64)
    a_over_r = _A_OVER_R * density ** (1.0 / 3.0) * f ** (-2.0 / 3.0)
    return np.arcsin(np.minimum(1.0, 1.0 / a_over_r)) / math.pi


def _check_ranges(
    density_range: tuple[float, float],
    duration_range: tuple[float, float],
    q_cap: float,
) -> tuple[float, float, float, float, float]:
    rho_lo, rho_hi = (float(x) for x in density_range)
    eta_lo, eta_hi = (float(x) for x in duration_range)
    if not (np.isfinite(rho_lo) and np.isfinite(rho_hi) and 0.0 < rho_lo <= rho_hi):
        raise ValueError(f"density_range must satisfy 0 < lo <= hi, got {density_range!r}.")
    if not (np.isfinite(eta_lo) and np.isfinite(eta_hi) and 0.0 < eta_lo <= eta_hi):
        raise ValueError(f"duration_range must satisfy 0 < lo <= hi, got {duration_range!r}.")
    q_cap = float(q_cap)
    if not 0.0 < q_cap < 1.0:
        raise ValueError(f"q_cap must be in (0, 1), got {q_cap!r}.")
    return rho_lo, rho_hi, eta_lo, eta_hi, q_cap


def density_duration_bounds(
    frequency: Array | float,
    density_range: tuple[float, float],
    duration_range: tuple[float, float] = (0.1, 1.5),
    q_cap: float = 0.25,
) -> tuple[Array, Array]:
    """Allowed duty-cycle range ``(q_lo, q_hi)`` at each ``frequency`` (1/day).

    Parameters
    ----------
    density_range :
        ``(rho_min, rho_max)`` of the host stars, in solar densities (1.41 g/cm^3).
        The shortest durations come from the densest stars, the longest from the
        least dense.
    duration_range :
        ``(eta_lo, eta_hi)``: allowed duration as a multiple of the central-transit
        duration. Values below 1 cover grazing and high-impact-parameter transits;
        above 1 allows for the companion radius and eccentricity.
    q_cap :
        Upper limit on ``q_hi`` (and ``q_lo``), which keeps the box widths -- and the
        cost of the search -- bounded at very short periods.
    """
    rho_lo, rho_hi, eta_lo, eta_hi, q_cap = _check_ranges(density_range, duration_range, q_cap)
    q_lo = np.minimum(eta_lo * central_duration_fraction(frequency, rho_hi), q_cap)
    q_hi = np.minimum(eta_hi * central_duration_fraction(frequency, rho_lo), q_cap)
    return q_lo, q_hi


def density_frequency_grid(
    t: Array,
    p_min: float,
    p_max: float,
    density_range: tuple[float, float],
    duration_range: tuple[float, float] = (0.1, 1.5),
    oversample: float = 3.0,
    q_cap: float = 0.25,
) -> Array:
    """Frequency grid (1/day) whose step follows the minimum duration at each period.

    The step is ``q_lo(f) / (oversample * baseline)`` with ``q_lo`` from
    :func:`density_duration_bounds`, so the phase drift across the baseline stays a
    fixed fraction of the narrowest allowed box. ``q_lo`` grows as ``f^(2/3)``, which
    makes the grid sparse at short periods; for wide period ranges it is orders of
    magnitude smaller than a uniform grid sized for ``q_lo`` at the longest period.
    """
    if not (0.0 < p_min < p_max and np.isfinite(p_max)):
        raise ValueError(f"Need 0 < p_min < p_max, got p_min={p_min!r}, p_max={p_max!r}.")
    if not oversample > 0.0:
        raise ValueError(f"oversample must be > 0, got {oversample!r}.")
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("t must be a non-empty array of finite times.")
    baseline = float(t.max() - t.min())
    if baseline <= 0.0:
        raise ValueError("Time array spans zero baseline.")
    f_min, f_max = 1.0 / p_max, 1.0 / p_min
    # Number of grid points below f: N(f) = oversample * baseline * integral df / q_lo.
    log_f = np.linspace(math.log(f_min), math.log(f_max), 8192)
    f_nodes = np.exp(log_f)
    q_lo, _ = density_duration_bounds(f_nodes, density_range, duration_range, q_cap)
    integrand = f_nodes / q_lo  # d(log f) measure
    cum = np.concatenate(
        ([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(log_f)))
    )
    cum *= oversample * baseline
    n_total = int(cum[-1]) + 1
    return np.interp(np.arange(n_total, dtype=np.float64), cum, f_nodes)
