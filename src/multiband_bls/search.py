"""The single front door: :func:`bls`."""

from __future__ import annotations

__all__ = ["bls", "bls_prepare", "PreparedData"]

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ._common import _check_coarse, _check_dlogq, _check_n_threads
from ._prep import auto_nbins
from ._validate import _check_sign, _validate_search_params
from .backends import get_implementation
from .config import Backend, Method, SearchConfig
from .engine import CapabilityError, RunOptions, SegmentSpec, execute
from .grids import build_frequency_grid
from .result import BLSResult, SearchOptions

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]

_METHODS = ("sparse", "binned", "fast")
_BACKENDS = ("cpu", "gpu", "auto")


@dataclass(frozen=True)
class PreparedData:
    """Validated, weighted (and, on the GPU, uploaded) data for repeated searches.

    Built by :func:`bls_prepare` from a *copy* of the caller's arrays, so later in-place
    edits of the originals cannot change it. Pass it to :func:`bls` in place of the data.
    """

    method: str
    backend: str
    multiband: bool
    rescale_errors: bool
    times: Array  # a read-only copy
    _state: Any  # backend state (private: arrays the cores read; never handed out)


def bls_prepare(
    data: tuple[Array, Array, Array] | Mapping[str, tuple[Array, Array, Array]],
    *,
    method: Method = "binned",
    backend: Literal["cpu", "gpu"] = "cpu",
    rescale_errors: bool = False,
) -> PreparedData:
    """Prepare ``data`` once for several :func:`bls` calls with the same method/backend.

    ``bls(data, ...)`` re-validates and re-preprocesses the caller's current arrays on
    every call; use this to skip that (and the GPU upload) when searching one light curve
    repeatedly, e.g. with several configurations.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}.")
    multiband = _is_multiband(data)
    if rescale_errors and not multiband:
        raise ValueError("rescale_errors applies to multiband data (a mapping of bands) only.")
    if multiband:
        owned: Any = {k: tuple(np.array(a, dtype=np.float64) for a in v)
                      for k, v in data.items()}  # type: ignore[union-attr]
    else:
        owned = tuple(np.array(a, dtype=np.float64) for a in data)  # type: ignore[union-attr]
    impl = get_implementation(method, backend, multiband)
    prep = impl.prepare(owned, rescale_errors)  # type: ignore[attr-defined]
    times = _all_times(owned, multiband).copy()  # not the backend's own array
    times.flags.writeable = False
    return PreparedData(method, backend, multiband, rescale_errors, times, prep)


def _is_multiband(data: Any) -> bool:
    if isinstance(data, Mapping):
        return True
    if isinstance(data, (tuple, list)) and len(data) == 3:
        return False
    raise ValueError("data must be (t, y, dy) for one band or a mapping label -> (t, y, dy).")


def _all_times(data: Any, multiband: bool) -> Array:
    if multiband:
        return np.concatenate([np.asarray(b[0], dtype=np.float64) for b in data.values()])
    return np.asarray(data[0], dtype=np.float64)


def bls(
    data: tuple[Array, Array, Array] | Mapping[str, tuple[Array, Array, Array]] | PreparedData,
    frequencies: Array | None = None,
    *,
    p_min: float | None = None,
    p_max: float | None = None,
    method: Method | None = None,
    backend: Backend | None = None,
    config: SearchConfig | None = None,
    rescale_errors: bool = False,
    oversample: float = 3.0,
    dlogq: float = 0.3,
    n_threads: int = 1,
    coarse_factor: int = 1,
    n_refine: int = 20,
    compute_n_bands_used: bool = True,
) -> BLSResult:
    """Box-fitting least-squares search over one band or several.

    Parameters
    ----------
    data :
        ``(t, y, dy)`` for one band, or a mapping ``label -> (t, y, dy)`` for a multiband
        search (shared period, epoch and duration; one depth per band).
    frequencies :
        Trial frequencies (1/day), or ``None`` to build the grid from ``p_min``/``p_max``
        (days) and ``config``: uniform with step ``q_min / (oversample * baseline)``, or,
        with a density prior, a grid whose step follows the minimum duration.
    data :
        Or a :class:`PreparedData` from :func:`bls_prepare`, which fixes ``method`` and
        ``backend``.
    method :
        ``"binned"`` (the default; exhaustive over integer-bin widths), ``"fast"`` (prefix sums
        and log-spaced widths, approximate in width resolution, much faster) or ``"sparse"``
        (unbinned, CPU only, ``O(N^2)`` per period).
    backend :
        ``"cpu"``, ``"gpu"`` (CuPy; ``"sparse"`` has no GPU version) or ``"auto"``: the GPU
        if available and the problem fits its limits, else the CPU -- the *same method*
        either way. The backend used is recorded in ``result.backend``.
    config :
        :class:`SearchConfig` (durations or a density prior, ``nbins``, ``min_points``,
        ``sign``).
    rescale_errors :
        Multiband only: scale each band's ``dy`` so its flat-model chi^2/dof is 1.
    oversample :
        Frequency steps per peak width when the grid is built here.
    dlogq, coarse_factor, n_refine :
        ``"fast"`` only: log-width spacing and the coarse-to-fine frequency search.
    n_threads :
        ``"fast"`` on the CPU only.
    compute_n_bands_used :
        Multiband ``"fast"`` on the GPU: set False to skip the per-frequency band counts.

    Returns
    -------
    BLSResult
        With ``method``, ``backend`` and the searched ``segments`` filled in.
    """
    cfg = SearchConfig() if config is None else config
    snap = data if isinstance(data, PreparedData) else None
    if snap is not None:
        for name, given, fixed in (("method", method, snap.method),
                                   ("backend", backend, snap.backend)):
            if given not in (None, fixed, "auto"):
                raise ValueError(f"{name}={given!r} conflicts with the prepared data's "
                                 f"{fixed!r}; leave it unset.")
        if rescale_errors and not snap.rescale_errors:
            raise ValueError("rescale_errors was not applied in bls_prepare().")
        method, backend, rescale_errors = (
            snap.method, snap.backend, snap.rescale_errors)  # type: ignore[assignment]
    method = "binned" if method is None else method
    backend = "auto" if backend is None else backend
    multiband = snap.multiband if snap is not None else _is_multiband(data)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}.")
    if method == "sparse" and backend == "gpu":
        raise ValueError("method='sparse' has no GPU implementation; use backend='cpu'.")
    if rescale_errors and not multiband:
        raise ValueError("rescale_errors applies to multiband data (a mapping of bands) only.")
    if method == "sparse" and cfg.nbins is not None:
        raise ValueError("nbins does not apply to method='sparse'.")
    sg = _check_sign(cfg.sign, "bls")
    prior = cfg.density_prior

    q_min, q_max = cfg.resolve_q(method)
    # frequencies
    if frequencies is None:
        if p_min is None or p_max is None:
            raise ValueError("Give frequencies, or both p_min and p_max.")
        times = snap.times if snap is not None else _all_times(data, multiband)
        if prior is not None:
            freqs_in = prior.grid(times, p_min, p_max, oversample)
        else:
            freqs_in = build_frequency_grid(times, p_min=p_min, p_max=p_max,
                                            oversample=oversample, q_min=q_min)
    else:
        if p_min is not None or p_max is not None:
            raise ValueError("Give either frequencies or p_min/p_max, not both.")
        freqs_in = np.asarray(frequencies)
    freqs, min_points, nbins = _validate_search_params(
        freqs_in, prior.q_cap if prior is not None else q_max, cfg.min_points,
        q_min=None if prior is not None else q_min, nbins=cfg.nbins)
    binned = method != "sparse"
    dlogq_v, (coarse_v, n_refine_v), threads_v = 0.3, (1, 20), 1
    if method == "fast":
        dlogq_v = _check_dlogq(dlogq, "bls")
        coarse_v, n_refine_v = _check_coarse(coarse_factor, n_refine, "bls")
        threads_v = _check_n_threads(n_threads, "bls")  # also checked when the GPU ignores it
    opts = RunOptions(min_points=min_points, sign=sg, sign_name=cfg.sign, dlogq=dlogq_v,
                      n_threads=threads_v, coarse_factor=coarse_v, n_refine=n_refine_v,
                      compute_n_bands_used=compute_n_bands_used)

    if prior is not None:
        segments: list[Any] = prior.segments(freqs, binned=binned, nbins=nbins)
    else:
        nb = (auto_nbins(q_min) if nbins is None else nbins) if binned else None
        segments = [(slice(None), SegmentSpec(q_min, q_max, nb))]

    def run(which: str, fallback: str | None = None) -> BLSResult:
        impl = get_implementation(method, which, multiband)
        prep = snap._state if snap is not None else impl.prepare(  # type: ignore[attr-defined]
            data, rescale_errors)
        info = SearchOptions(rescale_errors=rescale_errors, requested_backend=backend,
                             fallback=fallback)
        return execute(impl, prep, freqs, segments, opts, info)

    if backend != "auto":
        return run(backend)
    reason: str | None = None
    if method != "sparse" and _gpu_ready():
        try:
            return run("gpu")
        except CapabilityError as exc:
            reason = f"GPU limit: {exc}"
            logger.info("bls(backend='auto'): GPU cannot run this search (%s); using the CPU.",
                        exc)
    return run("cpu", reason)


def _gpu_ready() -> bool:
    try:
        from .gpu import gpu_available
    except ImportError:  # pragma: no cover
        return False
    return gpu_available()
