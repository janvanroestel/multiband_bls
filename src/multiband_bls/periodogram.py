"""Result containers for the (multiband) Sparse BLS periodograms."""

from __future__ import annotations

__all__ = ["BLSResult"]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
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
        the multiband search (same order as :attr:`bands`). NaN for a band with
        no in-transit (or no out-of-transit) data at the best solution, i.e. a
        band whose depth was not measured.
    best_power:
        ``power`` at the best solution. 0 when no box passed the cuts at any
        frequency; the ``best_*`` fields then refer to the first trial
        frequency and should not be used.
    bands:
        Band labels for a multiband result, else ``None``.
    n_bands_used:
        Multiband results (else ``None``; also ``None`` from
        :func:`multiband_eebls_gpu_fast` with ``compute_n_bands_used=False``): for each trial
        frequency, the number of bands ``k`` contributing to that frequency's
        best window, or 0 if no window was selected there. For a fixed window
        and correct uncertainties each contributing band adds ~1 to
        ``Delta chi^2`` under pure noise, so the noise level of ``power`` grows
        with ``k`` and values at frequencies with different ``k`` are not
        directly comparable. (``power`` is maximised over windows, so this does
        not give a significance.)
    band_chi2_red:
        Multiband results only (else ``None``): per-band flat-model reduced
        chi^2, ``sum_i (x_tilde/sigma)^2 / (N_b - 1)``, from the *input* ``dy``
        (NaN for a single-point band). Values far from 1, or very different
        between bands, can mean the reported uncertainties are mis-scaled and the
        matched-filter band weights are off; see the ``rescale_errors`` option.
        But the flat-model chi^2 includes any real variability, so bands with
        deep eclipses can read far above 1 even with correct uncertainties.
    """

    frequency: Array
    power: Array
    best_frequency: float
    best_period: float
    best_t0: float
    best_duration: float
    best_depth: float | Array
    best_power: float
    bands: tuple[str, ...] | None = None
    n_bands_used: npt.NDArray[np.int64] | None = None
    band_chi2_red: Array | None = None

    @property
    def period(self) -> Array:
        """Trial periods (days), i.e. ``1 / frequency``."""
        return 1.0 / self.frequency
