"""Result containers for the (multiband) Sparse BLS periodograms."""

from __future__ import annotations

__all__ = ["BLSResult"]

from dataclasses import dataclass

import numpy as np


@dataclass
class BLSResult:
    """Outcome of a Sparse BLS search.

    Attributes
    ----------
    frequency:
        Trial frequencies that were scanned (1/day).
    power:
        ``Delta chi^2 / chi^2_flat`` at each trial frequency -- the fraction of
        variance explained by the best box, in ``[0, 1]``, maximised at the best
        period (the cuvarbase BLS convention). For the multiband search the bands
        are combined with matched-filter weighting before the ratio is taken.
    best_frequency, best_period:
        Frequency/period of the global ``power`` maximum.
    best_t0:
        Epoch of mid-transit (days), in ``[0, best_period)``.
    best_duration:
        Transit/eclipse duration (days) at the best solution.
    best_depth:
        Fitted depth. A scalar for single-band searches; one value per band for
        the multiband search (same order as :attr:`bands`).
    best_power:
        ``power`` at the best solution.
    bands:
        Band labels for a multiband result, else ``None``.
    """

    frequency: np.ndarray
    power: np.ndarray
    best_frequency: float
    best_period: float
    best_t0: float
    best_duration: float
    best_depth: float | np.ndarray
    best_power: float
    bands: tuple[str, ...] | None = None

    @property
    def period(self) -> np.ndarray:
        """Trial periods (days), i.e. ``1 / frequency``."""
        return 1.0 / self.frequency
