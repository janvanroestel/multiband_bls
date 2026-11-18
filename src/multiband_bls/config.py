"""Search configuration: :class:`SearchConfig` and the stellar-density duration prior."""

from __future__ import annotations

__all__ = ["DensityPrior", "SearchConfig", "Method", "Backend", "Sign"]

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._prep import auto_nbins
from ._validate import _check_fraction
from .density import density_duration_bounds, density_frequency_grid
from .engine import SegmentSpec

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]

Method = Literal["sparse", "binned", "fast"]
Backend = Literal["cpu", "gpu", "auto"]
#: Which signal to keep, as the in-window deviation from the out-of-window baseline of the
#: input values: ``"positive"`` = above baseline (magnitudes: dimming/eclipses; fluxes:
#: brightening), ``"negative"`` = below baseline (fluxes: transits/eclipses).
Sign = Literal["both", "positive", "negative"]

_MAX_AUTO_NBINS = 500  # upper clip of auto_nbins


@dataclass(frozen=True)
class DensityPrior:
    """Allowed transit-duration range as a function of period, from a stellar-density range.

    A transit around a star of mean density ``rho`` has duty cycle ``q = T/P`` that scales
    as ``P^(-2/3)``. The prior turns an allowed density range into a period-dependent
    ``(q_lo, q_hi)``; a search then uses a separate width range and bin count for each
    stretch of the frequency grid over which that range changes by at most ``q_step``.

    Parameters
    ----------
    density_range :
        ``(rho_min, rho_max)`` of the host stars in solar densities (1.41 g/cm^3).
    duration_range :
        Allowed duration as a multiple of the central-transit duration: below 1 covers
        grazing / high-impact-parameter transits, above 1 the companion radius and
        eccentricity.
    q_cap :
        Upper limit on the duty cycles (bounds the cost at very short periods).
    q_floor :
        Optional lower limit on the duty cycles (``None``: none). Without it ``q_lo`` keeps
        falling as ``P^(-2/3)``; a floor stops the search resolving transits narrower than
        you can detect, and coarsens the frequency grid at long periods accordingly.
    q_step :
        A segment ends once ``q_lo`` or ``q_hi`` has grown by this factor.
    """

    density_range: tuple[float, float]
    duration_range: tuple[float, float] = (0.1, 1.5)
    q_cap: float = 0.25
    q_step: float = 1.05
    q_floor: float | None = None

    def __post_init__(self) -> None:
        density_duration_bounds(np.array([1.0]), self.density_range, self.duration_range,
                                self.q_cap, self.q_floor)  # validates the ranges
        if not self.q_step > 1.0:
            raise ValueError(f"q_step must be > 1, got {self.q_step!r}.")

    def bounds(self, frequency: Array | float) -> tuple[Array, Array]:
        """``(q_lo, q_hi)`` at ``frequency`` (1/day)."""
        return density_duration_bounds(frequency, self.density_range, self.duration_range,
                                       self.q_cap, self.q_floor)

    def grid(self, t: Array, p_min: float, p_max: float, oversample: float = 3.0) -> Array:
        """Frequency grid whose step follows the minimum duration (uniform in f^(1/3))."""
        return density_frequency_grid(t, p_min, p_max, self.density_range,
                                      self.duration_range, oversample, self.q_cap,
                                      self.q_floor)

    def segments(
        self, freqs: Array, *, binned: bool, nbins: int | None = None
    ) -> list[tuple[npt.NDArray[np.intp], SegmentSpec]]:
        """Cut ``freqs`` (any order) into segments with their own widths and bins.

        Returns ``(indices into freqs, SegmentSpec)`` pairs. ``q_min``/``q_max`` are the
        lower bound at the segment's lowest frequency and the upper bound at its highest.
        For binned methods ``nbins`` defaults to ``auto_nbins`` per segment (clipped to 500,
        noted once in the log); if no whole number of bins then fits ``[q_min, q_max]``,
        ``q_max`` is widened to the nearest representable width so that every backend
        searches the segment the same way. Widening never crosses ``q_cap``: at the cap
        ``q_min`` is lowered to the widest whole bin below it instead (only an ``nbins``
        too small for any width ``<= q_cap`` exceeds it). A collapsed range for the sparse
        search (``q_min == q_max``, e.g. ``q_floor == q_cap``) gets ``q_min = q_max /
        q_step``. Either adjustment sets ``SegmentSpec.widened``.
        """
        order = np.argsort(freqs, kind="stable")
        f_sorted = freqs[order]
        q_lo, q_hi = self.bounds(f_sorted)
        # a segment ends once q_hi *or* q_lo has grown by q_step since its first frequency
        # (both are non-decreasing in f); q_lo matters where q_hi is pinned at q_cap
        starts = [0]
        while True:
            s0 = starts[-1]
            nxt = min(
                int(np.searchsorted(q_hi, q_hi[s0] * self.q_step, side="right")),
                int(np.searchsorted(q_lo, q_lo[s0] * self.q_step, side="right")),
            )
            if nxt >= f_sorted.size:
                break
            starts.append(nxt)
        ends = [*starts[1:], f_sorted.size]

        out: list[tuple[npt.NDArray[np.intp], SegmentSpec]] = []
        capped_logged = False
        for s, e in zip(starts, ends, strict=True):
            q_min, q_max = float(q_lo[s]), float(q_hi[e - 1])
            nb = nbins
            if binned and nb is None:
                wanted = round(5.0 / q_min)
                nb = auto_nbins(max(q_min, 5.0 / _MAX_AUTO_NBINS))
                if wanted > _MAX_AUTO_NBINS and not capped_logged:
                    logger.info(
                        "density prior: narrowest allowed duty cycle (%.2g) needs %d bins "
                        "for 5 bins per transit; using %d. Pass nbins to override.",
                        q_min, wanted, _MAX_AUTO_NBINS)
                    capped_logged = True
            widened = False
            if nb is not None:
                kmi = max(1, math.ceil(q_min * nb - 1e-9))  # as reference.width_bounds
                kma = math.floor(q_max * nb + 1e-9)
                if kmi > kma:
                    widened = True
                    if kmi / nb <= self.q_cap + 1e-12:
                        q_max = kmi / nb          # nearest whole bin above, within q_cap
                    elif kma >= 1:
                        q_min = kma / nb          # at the cap: widest whole bin below it
                    else:                         # nbins too small for any width <= q_cap
                        q_max = min(kmi / nb, 0.999)
            elif q_min >= q_max:
                # sparse windows are half-open [q_min, q_max): a collapsed range (q_floor
                # == q_cap, or both bounds pinned at q_cap) would admit nothing
                q_min, widened = q_max / self.q_step, True
            out.append((order[s:e], SegmentSpec(q_min, q_max, nb, widened)))
        return out


@dataclass(frozen=True)
class SearchConfig:
    """Search settings shared between grid construction and the search.

    ``q_min``/``q_max`` left as ``None`` resolve per method (``q_min=0.01``; ``q_max=0.15``
    for ``"sparse"``, ``0.10`` otherwise). A ``density_prior`` replaces them and is
    exclusive with explicit values.
    """

    q_min: float | None = None
    q_max: float | None = None
    nbins: int | None = None
    min_points: int = 3
    sign: Sign = "both"
    density_prior: DensityPrior | None = None

    def __post_init__(self) -> None:
        if self.density_prior is not None and (self.q_min is not None
                                               or self.q_max is not None):
            raise ValueError("density_prior replaces q_min/q_max: pass one or the other.")

    def resolve_q(self, method: str) -> tuple[float, float]:
        """``(q_min, q_max)`` with the per-method defaults filled in."""
        # explicit bounds go through the same numeric-type check as the legacy functions
        q_min = 0.01 if self.q_min is None else _check_fraction("q_min", self.q_min)
        q_max = (0.15 if method == "sparse" else 0.10) if self.q_max is None \
            else _check_fraction("q_max", self.q_max)
        return q_min, q_max

