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
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from . import _eebls, _fast, _meebls, _msbls, _sbls  # type: ignore[attr-defined]
from .periodogram import BLSResult
from .reference import (
    _check_phase_range,
    _merge_band_arrays,
    _prepare_bands,
    _validate_light_curve,
    _validate_search_params,
    auto_nbins,
    preprocess,
    variance_explained,
)

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]


def _preprocess_single(
    y: Array, dy: Array
) -> tuple[Array, Array, float]:
    """Preprocess a single-band light curve for the Cython cores.

    Returns ``(wx, w_hat, yy)`` where ``wx = w_hat * x_tilde``,
    ``w_hat`` are the normalised weights (sum to 1), and ``yy`` is the
    flat-model weighted variance (``Δχ²`` denominator).
    """
    w_hat, x_tilde, _ = preprocess(y, dy)
    yy = float(np.sum(w_hat * x_tilde ** 2))
    wx = np.ascontiguousarray(w_hat * x_tilde, dtype=np.float64)
    w_hat = np.ascontiguousarray(w_hat, dtype=np.float64)
    return wx, w_hat, yy


def _merge_bands(
    bands: Mapping[str, tuple[Array, Array, Array]]
) -> tuple[Array, Array, Array, Array, tuple[str, ...], Array, float]:
    """Preprocess each band and merge into contiguous all-band arrays.

    Returns ``(t, wx, w_hat, band_idx, labels, w_totals, chi2_flat)`` where
    ``wx = w_hat * x_tilde`` uses each band's own normalised weights,
    ``band_idx`` labels each point with its band index, ``w_totals[b] =
    sum_i 1/sigma^2`` per band, and ``chi2_flat`` is the total flat-model chi^2
    (``sum_b sum_i (x_tilde/sigma)^2``). Shared by the multiband entry points.

    Thin wrapper over :func:`reference._merge_band_arrays` that makes the arrays
    C-contiguous with the dtypes the Cython cores require.
    """
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_band_arrays(bands)
    return (
        np.ascontiguousarray(t_all, dtype=np.float64),
        np.ascontiguousarray(wx_all, dtype=np.float64),
        np.ascontiguousarray(w_all, dtype=np.float64),
        np.ascontiguousarray(band_all, dtype=np.int64),
        labels,
        np.ascontiguousarray(w_totals, dtype=np.float64),
        chi2_flat,
    )


def sparse_bls(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    q_max: float = 0.15,
    min_points: int = 3,
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
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required. Unlike the binned
        variants there is no minimum duration (``q_min``), so this is the only
        guard against ultra-narrow boxes. Note the default ``q_max`` (0.15)
        differs from the binned variants' (0.10).

    Raises
    ------
    ValueError
        On invalid input: non-finite values, ``dy <= 0``, unequal lengths,
        non-positive frequencies, ``q_max`` outside ``(0, 1)``, or a
        non-integer / ``< 1`` ``min_points``.
    """
    freqs, min_points, _ = _validate_search_params(frequencies, q_max, min_points)
    t, y, dy = _validate_light_curve(t, y, dy)
    _check_phase_range(t, freqs)
    t = np.ascontiguousarray(t, dtype=np.float64)
    wx, w_hat, yy = _preprocess_single(y, dy)

    power, best_freq, t0, dur, depth, sr = _sbls.sbls_grid(
        t, wx, w_hat, freqs, float(q_max), min_points
    )
    return BLSResult(
        frequency=freqs,
        power=variance_explained(power, yy),
        best_frequency=float(best_freq),
        best_period=float(1.0 / best_freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=float(depth),
        best_power=float(variance_explained(np.array([sr]), yy)[0]),
    )


def eebls(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
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

    Raises
    ------
    ValueError
        On invalid input (see :func:`sparse_bls`), ``q_min`` outside
        ``(0, q_max]``, or a non-integer / ``< 1`` ``nbins``.
    """
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    t, y, dy = _validate_light_curve(t, y, dy)
    _check_phase_range(t, freqs)
    t = np.ascontiguousarray(t, dtype=np.float64)
    tau_arr = np.zeros(1, dtype=np.float64)  # box only
    wx, w_hat, yy = _preprocess_single(y, dy)

    power, best_freq, t0, dur, depth, sr, _ = _eebls.eebls_grid(
        t, wx, w_hat, freqs, nbins, float(q_min), float(q_max),
        min_points, tau_arr,
    )
    return BLSResult(
        frequency=freqs,
        power=variance_explained(power, yy),
        best_frequency=float(best_freq),
        best_period=float(1.0 / best_freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=float(depth),
        best_power=float(variance_explained(np.array([sr]), yy)[0]),
    )


def multiband_sparse_bls(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    q_max: float = 0.15,
    min_points: int = 3,
    rescale_errors: bool = False,
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
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required across all bands. As in
        :func:`sparse_bls`, there is no minimum duration (``q_min``).
    rescale_errors :
        If True, multiply each band's ``dy`` by ``sqrt(band_chi2_red)`` so its
        flat-model chi^2/dof is 1 before weighting (bands with a single point or
        zero scatter are left unchanged, with a warning). Off by default.

    Returns
    -------
    BLSResult
        Includes ``n_bands_used`` (contributing bands per frequency) and
        ``band_chi2_red`` (per-band reduced chi^2 of the input ``dy``).
    """
    freqs, min_points, _ = _validate_search_params(frequencies, q_max, min_points)
    bands, chi2_red = _prepare_bands(bands, rescale_errors)
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    _check_phase_range(t_all, freqs)
    band_w = np.ascontiguousarray(w_totals, dtype=np.float64)

    power, best_freq, t0, dur, depths, sr, k_used = _msbls.msbls_grid(
        t_all, wx_all, w_all, band_all, len(labels), freqs,
        float(q_max), min_points, band_w,
    )
    return BLSResult(
        frequency=freqs,
        power=variance_explained(power, chi2_flat),
        best_frequency=float(best_freq),
        best_period=float(1.0 / best_freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=np.asarray(depths),
        best_power=float(variance_explained(np.array([sr]), chi2_flat)[0]),
        bands=labels,
        n_bands_used=np.asarray(k_used),
        band_chi2_red=chi2_red,
    )


def multiband_eebls(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    rescale_errors: bool = False,
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
    """
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    bands, chi2_red = _prepare_bands(bands, rescale_errors)
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    _check_phase_range(t_all, freqs)
    band_w = np.ascontiguousarray(w_totals, dtype=np.float64)

    power, best_freq, t0, dur, depths, sr, k_used = _meebls.meebls_grid(
        t_all, wx_all, w_all, band_all, len(labels), freqs, nbins,
        float(q_min), float(q_max), min_points, band_w,
    )
    return BLSResult(
        frequency=freqs,
        power=variance_explained(power, chi2_flat),
        best_frequency=float(best_freq),
        best_period=float(1.0 / best_freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=np.asarray(depths),
        best_power=float(variance_explained(np.array([sr]), chi2_flat)[0]),
        bands=labels,
        n_bands_used=np.asarray(k_used),
        band_chi2_red=chi2_red,
    )


# --------------------------------------------------------------------------- #
# Prefix-sum / log-width helpers, shared with the fast GPU entry points        #
# --------------------------------------------------------------------------- #
def _check_dlogq(dlogq: float, name: str) -> float:
    """Return ``dlogq`` as a float; raise unless it is a finite real number >= 0."""
    if isinstance(dlogq, bool) or not isinstance(dlogq, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name}: dlogq must be a real number, got {dlogq!r}.")
    dlogq = float(dlogq)
    if not (np.isfinite(dlogq) and dlogq >= 0.0):
        raise ValueError(f"{name}: dlogq must be finite and >= 0, got {dlogq!r}.")
    return dlogq



def _log_widths(kmi: int, kma: int, dlogq: float = 0.3) -> np.ndarray:
    """Geometrically-spaced integer window widths between kmi and kma (inclusive).

    Reduces window evaluations from O(kma - kmi) to O(log(kma/kmi) / log(1+dlogq)).
    Smaller dlogq = more widths = higher fidelity but slower.
    """
    from math import ceil

    widths: set[int] = set()
    w = kmi
    while w <= kma:
        widths.add(w)
        w = max(w + 1, ceil(w * (1.0 + dlogq)))
    widths.add(kma)
    return np.array(sorted(widths), dtype=np.int32)


def _grid_params(nbins: int, q_min: float, q_max: float) -> tuple[int, int]:
    kmi = max(1, int(q_min * nbins))
    kma = min(nbins, int(q_max * nbins) + 1)
    # Load-bearing invariant: every kernel wraps the circular window with a single
    # subtraction (`if (jj >= nb) jj -= nb`), which is only correct when the widest
    # window (kma) does not exceed nbins. Violating this would silently read/write
    # out of bounds in shared memory. Raise (not assert) so `python -O` keeps it.
    if kma > nbins:
        raise ValueError(f"kma ({kma}) must not exceed nbins ({nbins})")
    return kmi, kma


def _result(
    power_raw: np.ndarray,
    chi2_flat: float,
    frequencies: np.ndarray,
    refine_fn: Callable[[float], BLSResult],
    bands: tuple[str, ...] | None = None,
    band_chi2_red: Array | None = None,
    n_bands_used: npt.NDArray[np.int64] | None = None,
) -> BLSResult:
    power = variance_explained(power_raw, chi2_flat)
    best_idx = int(np.argmax(power))
    best_freq = float(frequencies[best_idx])
    ref = refine_fn(best_freq)
    return BLSResult(
        frequency=frequencies,
        power=power,
        best_frequency=best_freq,
        best_period=1.0 / best_freq,
        best_t0=ref.best_t0,
        best_duration=ref.best_duration,
        best_depth=ref.best_depth,
        best_power=float(power[best_idx]),
        bands=bands,
        n_bands_used=n_bands_used,
        band_chi2_red=band_chi2_red,
    )


def _check_n_threads(n_threads: int, name: str) -> int:
    """Return ``n_threads`` as an int; raise unless it is an integer >= 1."""
    if isinstance(n_threads, bool) or not isinstance(n_threads, (int, np.integer)) or n_threads < 1:
        raise ValueError(f"{name}: n_threads must be an integer >= 1, got {n_threads!r}.")
    return int(n_threads)


def _threaded_grid(kernel: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
                   freqs: npt.NDArray[np.float64], n_threads: int) -> npt.NDArray[np.float64]:
    """Run ``kernel(freq_chunk) -> power_chunk`` over ``freqs``, split over threads.

    The ``_fast`` Cython cores release the GIL for their whole frequency loop, so
    the chunks run in parallel. Four chunks per thread even out uneven chunk costs.
    """
    if n_threads == 1 or freqs.size < 2 * n_threads:
        return kernel(freqs)
    chunks = np.array_split(freqs, 4 * n_threads)
    with ThreadPoolExecutor(n_threads) as pool:
        return np.concatenate(list(pool.map(kernel, chunks)))


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

    Notes
    -----
    Times are median-centred before folding (like the fast GPU kernels), so the
    phase bins start at a different phase than in :func:`eebls`; with ``dlogq=0``
    the periodogram equals :func:`eebls` run on median-centred times. All sums are
    double precision. ``best_t0``/``best_duration``/``best_depth`` are recomputed
    exactly with :func:`eebls` at the best frequency (on the original times), while
    ``power``/``best_power`` are this search's values.
    """
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    t, y, dy = _validate_light_curve(t, y, dy)
    _check_phase_range(t, freqs)
    dlogq = _check_dlogq(dlogq, "eebls_fast")
    n_threads = _check_n_threads(n_threads, "eebls_fast")
    wx, w_hat, yy = _preprocess_single(y, dy)
    tc = np.ascontiguousarray(t - np.median(t), dtype=np.float64)
    kmi, kma = _grid_params(nbins, q_min, q_max)
    widths = np.ascontiguousarray(_log_widths(kmi, kma, dlogq), dtype=np.int32)
    mp = min(min_points, t.shape[0] + 1)

    def kernel(f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _fast.eebls_fast_grid(tc, wx, w_hat, np.ascontiguousarray(f), nbins, widths, mp)

    power = _threaded_grid(kernel, freqs, n_threads)
    return _result(power, yy, freqs,
                   lambda bf: eebls(t, y, dy, np.array([bf]), nbins=nbins,
                                    q_min=q_min, q_max=q_max, min_points=min_points))


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

    Notes
    -----
    As in :func:`eebls_fast`: median-centred times, double-precision sums, and the
    best period's ``t0``/duration/depths recomputed exactly with
    :func:`multiband_eebls`. ``min_points`` is an exact total in-transit count over
    all bands, and a band contributes only with >= 1 point in and out of the window
    (same semantics as :func:`multiband_eebls`). ``n_bands_used`` is not computed
    (``None``).
    """
    freqs, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    bands, chi2_red = _prepare_bands(bands, rescale_errors)
    dlogq = _check_dlogq(dlogq, "multiband_eebls_fast")
    n_threads = _check_n_threads(n_threads, "multiband_eebls_fast")
    t_all, wx_all, w_all, band_all, labels, w_totals, chi2_flat = _merge_bands(bands)
    _check_phase_range(t_all, freqs)
    tc = np.ascontiguousarray(t_all - np.median(t_all), dtype=np.float64)
    band_w = np.ascontiguousarray(w_totals, dtype=np.float64)
    kmi, kma = _grid_params(nbins, q_min, q_max)
    widths = np.ascontiguousarray(_log_widths(kmi, kma, dlogq), dtype=np.int32)
    mp = min(min_points, t_all.shape[0] + 1)

    def kernel(f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _fast.meebls_fast_grid(tc, wx_all, w_all, band_all, len(labels),
                                      np.ascontiguousarray(f), nbins, widths, mp, band_w)

    power = _threaded_grid(kernel, freqs, n_threads)
    # bands were already rescaled above: refine on them without rescaling again
    return _result(power, chi2_flat, freqs,
                   lambda bf: multiband_eebls(bands, np.array([bf]), nbins=nbins,
                                              q_min=q_min, q_max=q_max,
                                              min_points=min_points),
                   bands=labels, band_chi2_red=chi2_red)
