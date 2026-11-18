"""Fast entry points backed by the compiled Cython cores.

- :func:`sparse_bls` / :func:`multiband_sparse_bls` — unbinned, O(N²/freq), precise
- :func:`eebls` / :func:`multiband_eebls` — phase-binned, O(N/freq), fast for large N
- :func:`eebls_fast` / :func:`multiband_eebls_fast` — phase-binned with prefix sums and
  log-spaced widths (the CPU counterparts of the ``*_gpu_fast`` entry points), optionally
  multi-threaded; for large frequency grids on machines without a GPU

All functions return :class:`~multiband_bls.BLSResult` with
``power = Δχ²/χ²_flat ∈ [0, 1]`` (fraction of variance explained, maximised at
the best period). Drop-in replacements for the reference implementations in
:mod:`multiband_bls.reference`.
"""

from __future__ import annotations

__all__ = [
    "sparse_bls",
    "eebls",
    "multiband_sparse_bls",
    "multiband_eebls",
    "eebls_fast",
    "multiband_eebls_fast",
]

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from ._common import _check_coarse, _check_dlogq, _check_n_threads
from ._prep import auto_nbins
from ._validate import _check_sign, _validate_search_params
from .backends import get_implementation
from .engine import RunOptions, SegmentSpec, execute
from .result import BLSResult

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]






def _search(
    method: str,
    multiband: bool,
    data: Any,
    freqs: Array,
    spec: SegmentSpec,
    opts: RunOptions,
    rescale_errors: bool = False,
    backend: str = "cpu",
) -> BLSResult:
    """Prepare ``data`` once and run a single-segment search (the legacy entry points)."""
    impl = get_implementation(method, backend, multiband)
    prep = impl.prepare(data, rescale_errors)  # type: ignore[attr-defined]
    return execute(impl, prep, freqs, [(slice(None), spec)], opts)



def sparse_bls(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    q_min: float = 0.01,
    q_max: float = 0.15,
    min_points: int = 3,
    sign: str = "both",
) -> BLSResult:
    """Single-band Sparse BLS (compiled). See :func:`reference.sparse_bls_reference`.

    ``power``/``best_power`` are ``Delta chi^2 / chi^2_flat = SR^2 / YY`` (the
    fraction of variance explained, in ``[0, 1]``, maximised at the period).

    Parameters
    ----------
    t :
        Observation times (days), 1-D float array.
    y :
        Magnitudes or fluxes, 1-D float array, same length as ``t``.
    dy :
        1-sigma uncertainties, 1-D float array, same length as ``t``.
    frequencies :
        Trial frequencies (1/day).
    q_min, q_max :
        Minimum and maximum transit duration as a fraction of the period. The
        duration of a window is measured between the midpoints of the gaps to
        its neighbouring points. Note the default ``q_max`` (0.15) differs from
        the binned variants' (0.10).
    min_points :
        Minimum number of in-transit points required.
    sign :
        Signal sign to search for: ``"both"`` (default), ``"positive"`` or
        ``"negative"``. Positive keeps windows whose weighted mean of ``y`` lies above
        the out-of-window baseline (magnitudes: dimming/eclipse; fluxes: brightening),
        negative the opposite.

    Raises
    ------
    ValueError
        On invalid input: non-finite values, ``dy <= 0``, unequal lengths,
        non-positive frequencies, ``q_max`` outside ``(0, 1)``, or a
        non-integer / ``< 1`` ``min_points``.
    """
    sg = _check_sign(sign, "sparse_bls")
    freqs, min_points, _ = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min
    )
    return _search("sparse", False, (t, y, dy), freqs, SegmentSpec(float(q_min), float(q_max)),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign))


def eebls(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    sign: str = "both",
) -> BLSResult:
    """Classic binned BLS with edge-effect handling (Kovacs et al. 2002).

    The phase-folded light curve is binned onto ``nbins`` phase bins and the box
    is searched over bins, giving ``O(N)`` per-period cost versus the unbinned
    :func:`sparse_bls`'s ``O(N^2)``.
    ``power``/``best_power`` are ``Delta chi^2 / chi^2_flat = SR^2 / YY``.

    Parameters
    ----------
    t :
        Observation times (days), 1-D float array.
    y :
        Magnitudes or fluxes, 1-D float array, same length as ``t``.
    dy :
        1-sigma uncertainties, 1-D float array, same length as ``t``.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``, ensuring at least 5 bins inside the narrowest transit.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required.
    sign :
        Signal sign to search for: ``"both"`` (default), ``"positive"`` or
        ``"negative"``. Positive keeps windows whose weighted mean of ``y`` lies above
        the out-of-window baseline (magnitudes: dimming/eclipse; fluxes: brightening),
        negative the opposite. In multiband searches the cut is per band: a band whose
        window has the wrong sign contributes nothing. Costs one multiply-compare per
        window.

    Raises
    ------
    ValueError
        On invalid input (see :func:`sparse_bls`), ``q_min`` outside
        ``(0, q_max]``, or a non-integer / ``< 1`` ``nbins``.
    """
    sg = _check_sign(sign, "eebls")
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    return _search("binned", False, (t, y, dy), freqs,
                   SegmentSpec(float(q_min), float(q_max), nbins),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign))


def multiband_sparse_bls(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    q_min: float = 0.01,
    q_max: float = 0.15,
    min_points: int = 3,
    rescale_errors: bool = False,
    sign: str = "both",
) -> BLSResult:
    """Multiband Sparse BLS (compiled). See :func:`reference.multiband_sparse_bls_reference`.

    Bands combine with matched-filter weighting (each weighted by its total
    inverse variance ``W_b``). ``power``/``best_power`` are
    ``Delta chi^2 / chi^2_flat`` -- the fraction of variance explained, in
    ``[0, 1]``, maximised at the best period.

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch, and duration.
    frequencies :
        Trial frequencies (1/day).
    q_min, q_max :
        Minimum and maximum transit duration as a fraction of the period (see
        :func:`sparse_bls`).
    min_points :
        Minimum number of in-transit points required across all bands.
    rescale_errors :
        If True, multiply each band's ``dy`` by ``sqrt(band_chi2_red)`` so its
        flat-model chi^2/dof is 1 before weighting (bands with a single point or
        zero scatter are left unchanged, with a warning). Off by default.
    sign :
        ``"both"`` (default), ``"positive"`` or ``"negative"``; each band is
        judged separately (a band whose in-window flux sum has the wrong sign
        adds nothing).

    Returns
    -------
    BLSResult
        Includes ``n_bands_used`` (contributing bands per frequency) and
        ``band_chi2_red`` (per-band reduced chi^2 of the input ``dy``).
    """
    sg = _check_sign(sign, "multiband_sparse_bls")
    freqs, min_points, _ = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min
    )
    return _search("sparse", True, bands, freqs, SegmentSpec(float(q_min), float(q_max)),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign),
                   rescale_errors=rescale_errors)


def multiband_eebls(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    rescale_errors: bool = False,
    sign: str = "both",
) -> BLSResult:
    """Binned multiband BLS (compiled). See :func:`reference.multiband_eebls_reference`.

    Combines the speed of binning with the per-band-depth multiband model: each
    band is binned onto ``nbins`` shared phase bins; ``O(N)`` per period rather
    than the ``O(N^2)`` of :func:`multiband_sparse_bls`. Matched-filter weighting;
    ``power``/``best_power`` are ``Delta chi^2 / chi^2_flat``.

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch, and duration.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required across all bands.
    rescale_errors :
        As in :func:`multiband_sparse_bls`.
    sign :
        Signal sign to search for: ``"both"`` (default), ``"positive"`` or
        ``"negative"``. Positive keeps windows whose weighted mean of ``y`` lies above
        the out-of-window baseline (magnitudes: dimming/eclipse; fluxes: brightening),
        negative the opposite. In multiband searches the cut is per band: a band whose
        window has the wrong sign contributes nothing. Costs one multiply-compare per
        window.
    """
    sg = _check_sign(sign, "multiband_eebls")
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    return _search("binned", True, bands, freqs,
                   SegmentSpec(float(q_min), float(q_max), nbins),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign),
                   rescale_errors=rescale_errors)


# --------------------------------------------------------------------------- #
# Prefix-sum / log-width helpers, shared with the fast GPU entry points        #
# --------------------------------------------------------------------------- #



























def eebls_fast(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    dlogq: float = 0.3,
    n_threads: int = 1,
    coarse_factor: int = 1,
    n_refine: int = 20,
    sign: str = "both",
) -> BLSResult:
    """Fast CPU binned BLS using prefix-sum bins and log-spaced transit widths.

    The CPU counterpart of :func:`multiband_bls.eebls_gpu_fast`: the phase bins are
    turned into cumulative sums, so each window sum is an O(1) difference, and only
    a geometrically-spaced subset of widths between ``q_min`` and ``q_max`` is
    evaluated. Per frequency this costs ``O(N + nbins * n_widths)`` instead of the
    ``O(N + nbins * kma)`` of :func:`eebls`, which dominates for large ``nbins`` and
    wide ``q`` ranges.

    Parameters
    ----------
    t, y, dy, frequencies, nbins, q_min, q_max, min_points :
        As in :func:`eebls`.
    dlogq :
        Logarithmic width spacing, as in :func:`multiband_bls.eebls_gpu_fast`.
        ``dlogq=0`` evaluates every integer width (same windows as :func:`eebls`).
    n_threads :
        Number of threads the frequency grid is split over (default 1).
    coarse_factor, n_refine :
        Opt-in coarse-to-fine search (default ``coarse_factor=1``: the full grid).
        With ``coarse_factor = k > 1`` only every k-th frequency is searched, then
        the full grid is searched around the ``n_refine`` highest separated coarse
        peaks. This cuts the work by roughly ``k`` but can miss a peak whose coarse
        sample falls outside the top ``n_refine`` (narrow signals, weak peaks among
        many aliases); keep ``k`` at or below the grid's oversampling and validate
        recovery against the full grid before using it. ``frequency``/``power`` stay
        full length, with unsearched frequencies holding the interpolated coarse
        power.
    sign :
        Signal sign to search for: ``"both"`` (default), ``"positive"`` or
        ``"negative"``. Positive keeps windows whose weighted mean of ``y`` lies above
        the out-of-window baseline (magnitudes: dimming/eclipse; fluxes: brightening),
        negative the opposite. In multiband searches the cut is per band: a band whose
        window has the wrong sign contributes nothing. Costs one multiply-compare per
        window.

    Notes
    -----
    Times are median-centred before folding (like the fast GPU kernels), so the
    phase bins start at a different phase than in :func:`eebls`; with ``dlogq=0``
    the periodogram equals :func:`eebls` run on median-centred times. All sums are
    double precision. ``best_t0``/``best_duration``/``best_depth`` are recomputed
    exactly with :func:`eebls` at the best frequency (on the original times), while
    ``power``/``best_power`` are this search's values.
    """
    sg = _check_sign(sign, "eebls_fast")
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    dlogq = _check_dlogq(dlogq, "eebls_fast")
    n_threads = _check_n_threads(n_threads, "eebls_fast")
    coarse_factor, n_refine = _check_coarse(coarse_factor, n_refine, "eebls_fast")
    return _search("fast", False, (t, y, dy), freqs,
                   SegmentSpec(float(q_min), float(q_max), nbins),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign, dlogq=dlogq,
                              n_threads=n_threads, coarse_factor=coarse_factor,
                              n_refine=n_refine))


def multiband_eebls_fast(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    dlogq: float = 0.3,
    rescale_errors: bool = False,
    n_threads: int = 1,
    coarse_factor: int = 1,
    n_refine: int = 20,
    sign: str = "both",
) -> BLSResult:
    """Fast CPU binned multiband BLS using prefix-sum bins and log-spaced widths.

    The CPU counterpart of :func:`multiband_bls.multiband_eebls_gpu_fast`; see
    :func:`eebls_fast` for the design. Each band keeps its own baseline and depth;
    all bands share the trial period, epoch and duration.

    Parameters
    ----------
    bands, frequencies, nbins, q_min, q_max, min_points, rescale_errors :
        As in :func:`multiband_eebls`.
    dlogq :
        Logarithmic width spacing, as in :func:`multiband_bls.multiband_eebls_gpu_fast`.
    n_threads :
        Number of threads the frequency grid is split over (default 1).
    coarse_factor, n_refine :
        Opt-in coarse-to-fine search, as in :func:`eebls_fast`.
    sign :
        Signal sign to search for: ``"both"`` (default), ``"positive"`` or
        ``"negative"``. Positive keeps windows whose weighted mean of ``y`` lies above
        the out-of-window baseline (magnitudes: dimming/eclipse; fluxes: brightening),
        negative the opposite. In multiband searches the cut is per band: a band whose
        window has the wrong sign contributes nothing. Costs one multiply-compare per
        window.

    Notes
    -----
    As in :func:`eebls_fast`: median-centred times, double-precision sums, and the
    best period's ``t0``/duration/depths recomputed exactly with
    :func:`multiband_eebls`. ``min_points`` is an exact total in-transit count over
    all bands, and a band contributes only with >= 1 point in and out of the window
    (same semantics as :func:`multiband_eebls`). ``n_bands_used`` is not computed
    (``None``).
    """
    sg = _check_sign(sign, "multiband_eebls_fast")
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    dlogq = _check_dlogq(dlogq, "multiband_eebls_fast")
    n_threads = _check_n_threads(n_threads, "multiband_eebls_fast")
    coarse_factor, n_refine = _check_coarse(coarse_factor, n_refine, "multiband_eebls_fast")
    return _search("fast", True, bands, freqs,
                   SegmentSpec(float(q_min), float(q_max), nbins),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign, dlogq=dlogq,
                              n_threads=n_threads, coarse_factor=coarse_factor,
                              n_refine=n_refine),
                   rescale_errors=rescale_errors)
