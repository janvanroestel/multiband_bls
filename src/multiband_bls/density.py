"""Stellar-density-based search: period-dependent duration range and frequency grid.

A transit of a small companion across a star of mean density ``rho`` lasts
``T = (P / pi) * arcsin(R / a)`` with ``a / R = (G rho P^2 / 3 pi)^(1/3)``, i.e. a
duty cycle ``q = T / P`` proportional to ``P^(-2/3)`` for ``a >> R``. A search with a
fixed ``(q_min, q_max)`` therefore tests many widths no real transit can have, and its
frequency step has to be small enough for the *narrowest* box at *every* period.

:func:`density_bls` instead splits the frequency grid into segments, gives each its own
``(q_min, q_max, nbins)`` from an allowed stellar-density range, and runs any of the
existing search functions on each segment. :func:`density_frequency_grid` builds the
matching grid, whose step follows the (period-dependent) minimum duration, so it is
uniform in ``f^(1/3)`` (Ofir 2014) rather than in ``f``.
"""

from __future__ import annotations

__all__ = [
    "RHO_SUN",
    "central_duration_fraction",
    "density_duration_bounds",
    "density_frequency_grid",
    "density_bls",
]

import inspect
import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from ._prep import auto_nbins
from .result import BLSResult

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]

_MAX_AUTO_NBINS = 500  # upper clip of auto_nbins

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


def density_bls(
    search: Callable[..., BLSResult],
    *args: Any,
    frequencies: Array,
    density_range: tuple[float, float],
    duration_range: tuple[float, float] = (0.1, 1.5),
    q_cap: float = 0.25,
    q_step: float = 1.25,
    **kwargs: Any,
) -> BLSResult:
    """Run ``search`` with a period-dependent duration range from a density range.

    ``search`` is any of the package's search functions (``eebls``,
    ``multiband_eebls_fast``, ``eebls_gpu_fast``, ``sparse_bls``, ...); ``*args``
    are its leading data arguments (``t, y, dy`` or ``bands``) and ``**kwargs`` its
    remaining options (``min_points``, ``sign``, ``n_threads``, ...). Do not pass
    ``q_min``/``q_max`` -- they are derived per segment.

    The sorted frequencies are cut into segments over which the allowed duty cycle
    changes by at most a factor ``q_step``; each segment is searched with
    ``q_min`` = the lower bound at its lowest frequency and ``q_max`` = the upper
    bound at its highest, and ``nbins = auto_nbins(q_min)`` unless ``nbins`` is given.

    Returns a single :class:`~multiband_bls.BLSResult` in the order of
    ``frequencies``; the ``best_*`` fields come from the segment with the highest
    power. ``power`` is maximised over a different width range in each segment, so
    its noise floor drifts slowly with frequency.

    See :func:`density_duration_bounds` for ``density_range``, ``duration_range`` and
    ``q_cap``.
    """
    for forbidden in ("q_min", "q_max"):
        if forbidden in kwargs:
            raise ValueError(f"Do not pass {forbidden}: it is derived from density_range.")
    if not q_step > 1.0:
        raise ValueError(f"q_step must be > 1, got {q_step!r}.")
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size == 0 or not np.all(np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError("frequencies must be a non-empty 1-D array of positive finite values.")
    _check_ranges(density_range, duration_range, q_cap)

    order = np.argsort(freqs, kind="stable")
    f_sorted = freqs[order]
    q_lo, q_hi = density_duration_bounds(f_sorted, density_range, duration_range, q_cap)

    # Segment edges: a new segment starts once q_hi *or* q_lo has grown by q_step since
    # the segment's first frequency (both are non-decreasing in f). Checking q_lo too
    # matters where q_hi is pinned at q_cap while q_lo still climbs.
    starts = [0]
    while True:
        s0 = starts[-1]
        nxt = min(
            int(np.searchsorted(q_hi, q_hi[s0] * q_step, side="right")),
            int(np.searchsorted(q_lo, q_lo[s0] * q_step, side="right")),
        )
        if nxt >= f_sorted.size:
            break
        starts.append(nxt)
    ends = [*starts[1:], f_sorted.size]

    has_nbins = "nbins" in inspect.signature(search).parameters
    results: list[tuple[int, int, BLSResult]] = []
    capped_logged = False
    for s, e in zip(starts, ends, strict=True):
        seg_kwargs = dict(kwargs)
        q_min_seg, q_max_seg = float(q_lo[s]), float(q_hi[e - 1])
        if has_nbins and seg_kwargs.get("nbins") is None:
            # auto_nbins would warn once per segment when 500 bins cannot resolve the
            # narrowest allowed box; the cap is expected here, so note it once instead.
            wanted = round(5.0 / q_min_seg)
            seg_kwargs["nbins"] = auto_nbins(max(q_min_seg, 5.0 / _MAX_AUTO_NBINS))
            if wanted > _MAX_AUTO_NBINS and not capped_logged:
                logger.info(
                    "density_bls: narrowest allowed duty cycle (%.2g) needs %d bins for 5 "
                    "bins per transit; using %d. Pass nbins to override.",
                    q_min_seg, wanted, _MAX_AUTO_NBINS)
                capped_logged = True
        # The width grid is integer bins: if no whole number of bins lies in
        # [q_min, q_max] (a narrow range, e.g. rho_min == rho_max with eta_lo == eta_hi),
        # widen q_max to the nearest representable width so every segment is searched
        # the same way on every backend rather than raising or returning zero power.
        nb_seg = seg_kwargs.get("nbins")
        if nb_seg is not None:
            kmi = max(1, math.ceil(q_min_seg * nb_seg - 1e-9))  # as reference.width_bounds
            if kmi > math.floor(q_max_seg * nb_seg + 1e-9):
                q_max_seg = min(kmi / nb_seg, 0.999)
        logger.debug("density segment f=[%g, %g] q=[%g, %g]", f_sorted[s], f_sorted[e - 1],
                     q_min_seg, q_max_seg)
        res = search(
            *args, frequencies=np.ascontiguousarray(f_sorted[s:e]),
            q_min=q_min_seg, q_max=q_max_seg, **seg_kwargs,
        )
        results.append((s, e, res))

    power = np.empty(f_sorted.size, dtype=np.float64)
    first = results[0][2]
    # n_bands_used is all-or-nothing: some backends drop it (e.g. coarse_factor > 1 on
    # the GPU), possibly for only some segments.
    have_counts = all(r.n_bands_used is not None for _, _, r in results)
    n_used = np.zeros(f_sorted.size, dtype=np.int64) if have_counts else None
    for s, e, res in results:
        power[s:e] = res.power
        if n_used is not None and res.n_bands_used is not None:
            n_used[s:e] = res.n_bands_used
    # undo the sort so the output follows the caller's frequency order
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    best = max((r for _, _, r in results), key=lambda r: r.best_power)
    best_frequency, best_period = best.best_frequency, best.best_period
    if best.best_power <= 0.0:
        # Nothing passed the cuts anywhere: BLSResult documents the best_* fields as
        # referring to the *first trial frequency*, i.e. the caller's, not the lowest.
        pos = int(inverse[0])
        best = next(r for s, e, r in results if s <= pos < e)
        best_frequency, best_period = float(freqs[0]), float(1.0 / freqs[0])
    return BLSResult(
        frequency=freqs,
        power=power[inverse],
        best_frequency=best_frequency,
        best_period=best_period,
        best_t0=best.best_t0,
        best_duration=best.best_duration,
        best_depth=best.best_depth,
        best_power=best.best_power,
        bands=first.bands,
        n_bands_used=None if n_used is None else n_used[inverse],
        band_chi2_red=first.band_chi2_red,
    )
