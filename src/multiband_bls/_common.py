"""Helpers shared by the CPU and GPU backends (time/width grids, coarse search, bands)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from ._prep import _merge_band_arrays, preprocess, width_bounds
from ._validate import _check_phase_range  # noqa: F401

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
    kmi, kma = width_bounds(nbins, q_min, q_max)
    # Load-bearing invariant: every kernel wraps the circular window with a single
    # subtraction (`if (jj >= nb) jj -= nb`), which is only correct when the widest
    # window (kma) does not exceed nbins. Violating this would silently read/write
    # out of bounds in shared memory. Raise (not assert) so `python -O` keeps it.
    if kma > nbins:
        raise ValueError(f"kma ({kma}) must not exceed nbins ({nbins})")
    return kmi, kma


def _shift_bands(
    bands: Mapping[str, tuple[Array, Array, Array]], offset: float
) -> dict[str, tuple[Array, Array, Array]]:
    """Return ``bands`` with ``offset`` subtracted from every band's times."""
    return {k: (np.asarray(t, dtype=np.float64) - offset, y, dy) for k, (t, y, dy) in bands.items()}


def _flux_scale(dys: list[Array]) -> float:
    """Median uncertainty over all bands: the unit the float32 GPU kernels work in."""
    return float(np.median(np.concatenate([np.asarray(d, dtype=np.float64) for d in dys])))


def _scale_bands(
    bands: Mapping[str, tuple[Array, Array, Array]], scale: float
) -> dict[str, tuple[Array, Array, Array]]:
    """Return ``bands`` with ``y`` and ``dy`` divided by ``scale`` (power is invariant)."""
    return {k: (t, np.asarray(y, dtype=np.float64) / scale,
                np.asarray(dy, dtype=np.float64) / scale) for k, (t, y, dy) in bands.items()}


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


def _check_coarse(coarse_factor: int, n_refine: int, name: str) -> tuple[int, int]:
    """Validate the coarse-to-fine parameters (integers >= 1)."""
    for label, v in (("coarse_factor", coarse_factor), ("n_refine", n_refine)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{name}: {label} must be an integer >= 1, got {v!r}.")
    return int(coarse_factor), int(n_refine)


def _coarse_active(n_freqs: int, coarse_factor: int) -> bool:
    """Whether :func:`_coarse_to_fine` skips frequencies (else it evaluates them all)."""
    return coarse_factor > 1 and n_freqs >= 4 * coarse_factor


def _coarse_to_fine(
    evaluate: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    freqs: npt.NDArray[np.float64],
    coarse_factor: int,
    n_refine: int,
) -> npt.NDArray[np.float64]:
    """Raw power over ``freqs`` from a coarse pass plus local refinement.

    ``evaluate(f)`` returns the raw (unnormalised) power at the frequencies ``f``.
    With ``coarse_factor == 1`` this is just ``evaluate(freqs)``. Otherwise every
    ``coarse_factor``-th frequency (plus the last) is evaluated first; the
    ``n_refine`` highest coarse samples that are at least two coarse steps apart
    are then refined by evaluating every grid frequency within one coarse step of
    them. Frequencies that were not evaluated hold the linear interpolation of the
    coarse values (never above the larger neighbouring coarse sample), so the
    maximum of the result is always an exactly evaluated frequency.
    """
    nf = freqs.size
    if not _coarse_active(nf, coarse_factor):
        return evaluate(freqs)
    if np.any(np.diff(freqs) <= 0.0):  # interpolation and the refine bands need ascending order
        # np.unique sorts and drops duplicates (which would otherwise recurse forever,
        # since a sorted array with repeats still fails the strict-ascent test).
        uniq, inverse = np.unique(freqs, return_inverse=True)
        return _coarse_to_fine(evaluate, uniq, coarse_factor, n_refine)[inverse.ravel()]
    idx_c: npt.NDArray[np.intp] = np.arange(0, nf, coarse_factor, dtype=np.intp)
    if idx_c[-1] != nf - 1:
        idx_c = np.append(idx_c, np.intp(nf - 1))
    pc = evaluate(freqs[idx_c])
    picked: list[int] = []
    for c in np.argsort(pc)[::-1]:
        if all(abs(int(c) - q) >= 2 for q in picked):
            picked.append(int(c))
            if len(picked) == n_refine:
                break
    want = np.zeros(nf, dtype=bool)
    for c in picked:
        want[idx_c[max(c - 1, 0)]: idx_c[min(c + 1, idx_c.size - 1)] + 1] = True
    want[idx_c] = False
    todo = np.flatnonzero(want)
    power = np.interp(freqs, freqs[idx_c], pc)
    if todo.size:
        power[todo] = evaluate(freqs[todo])
    power[idx_c] = pc
    return power
