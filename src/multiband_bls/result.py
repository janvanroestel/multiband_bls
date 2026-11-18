"""Result containers for the (multiband) BLS periodograms."""

from __future__ import annotations

__all__ = ["BLSResult", "SearchOptions", "SegmentRecord"]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SegmentRecord:
    """One searched frequency segment: its range, size and duration/bin settings.

    ``q_min``/``q_max`` are the duty-cycle limits the segment was *searched* with. For a
    density search ``q_max`` may have been widened (``widened=True``) so that a whole
    number of bins fits. ``nbins`` is ``None`` for the unbinned sparse search.
    """

    f_min: float
    f_max: float
    n_freqs: int
    q_min: float
    q_max: float
    nbins: int | None = None
    widened: bool = False


@dataclass(frozen=True)
class SearchOptions:
    """The resolved options a result was computed with (for reproducing the search)."""

    sign: str = "both"
    min_points: int = 3
    dlogq: float | None = None  # "fast" method only
    coarse_factor: int = 1
    n_refine: int = 20
    rescale_errors: bool = False
    requested_backend: str | None = None  # what the caller asked for ("auto", "gpu", ...)
    fallback: str | None = None  # why "auto" used the CPU instead of the GPU, if it did


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
    method, backend:
        Which search produced the result (e.g. ``"binned"`` / ``"cpu"``), or ``None`` for
        results built by hand.
    options:
        The resolved :class:`SearchOptions` (sign, ``min_points``, fast/coarse settings,
        ``rescale_errors``, requested backend and any automatic CPU fallback).
    evaluated:
        Boolean array over :attr:`frequency`, or ``None`` when every frequency was
        evaluated. For a coarse-to-fine search the False entries hold interpolated power.
    segments:
        One :class:`SegmentRecord` per searched frequency segment (a single record for a
        fixed-``q`` search), or ``None``. This is a compact record of the widths actually
        searched; per-frequency bounds can be derived from it.
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
    method: str | None = None
    backend: str | None = None
    segments: tuple[SegmentRecord, ...] | None = None
    options: SearchOptions | None = None
    evaluated: npt.NDArray[np.bool_] | None = None

    def q_bounds(self, frequency: Array | float) -> tuple[Array, Array]:
        """``(q_min, q_max)`` the search used at ``frequency`` (from :attr:`segments`).

        Frequencies in no segment's range (or a result without segments) give NaN.
        """
        f = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
        lo = np.full(f.shape, np.nan)
        hi = np.full(f.shape, np.nan)
        for seg in self.segments or ():
            inside = (f >= seg.f_min) & (f <= seg.f_max)
            lo[inside], hi[inside] = seg.q_min, seg.q_max
        return lo, hi

    @property
    def has_solution(self) -> bool:
        """True if a window with positive power was found (``best_power > 0``).

        This says nothing about statistical significance. When it is False the ``best_*``
        fields refer to the first trial frequency and should not be used.
        """
        return bool(self.best_power > 0.0)

    @property
    def period(self) -> Array:
        """Trial periods (days), i.e. ``1 / frequency``."""
        return 1.0 / self.frequency
