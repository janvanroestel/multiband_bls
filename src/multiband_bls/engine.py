"""Search-engine contract shared by every method/backend, and the segment executor.

A search is ``prepare`` (validate, weight, merge, upload -- once) followed by ``run`` over
one or more *segments* of the frequency grid, each with its own duration range and bin
count. A fixed-q search is a single segment; a density-prior search has many. Backends
whose kernels only produce a periodogram (the fast CPU kernels and every GPU kernel) get
their best-fit epoch/duration/depth from ``refine`` -- run **once** at the global peak
against the already prepared host data -- while the exact CPU cores report it from ``run``.
"""

from __future__ import annotations

__all__ = [
    "CapabilityError",
    "BestFit",
    "Implementation",
    "Prepared",
    "RunOptions",
    "RunOutput",
    "SegmentSpec",
    "SegmentRecord",
    "execute",
]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._prep import variance_explained
from .result import BLSResult, SegmentRecord

Array = npt.NDArray[np.float64]


class CapabilityError(ValueError):
    """The requested search exceeds a backend limit (not invalid input).

    ``bls(backend="auto")`` falls back to the CPU on this; an explicit ``backend="gpu"``
    surfaces it. It subclasses ``ValueError`` so existing handlers keep working.
    """


@dataclass(frozen=True)
class SegmentSpec:
    """Duration range (fractions of the period) and bin count used for one segment."""

    q_min: float
    q_max: float
    nbins: int | None = None


@dataclass(frozen=True)
class RunOptions:
    """Execution options common to all backends (validated by the entry points)."""

    min_points: int = 3
    sign: float = 0.0  # kernels' -1/0/+1
    sign_name: str = "both"
    dlogq: float = 0.3
    n_threads: int = 1
    coarse_factor: int = 1
    n_refine: int = 20
    compute_n_bands_used: bool = True


@dataclass(frozen=True)
class BestFit:
    """Best window of a run: frequency, epoch, duration, depth(s) and raw statistic."""

    frequency: float
    t0: float
    duration: float
    depth: float | Array
    sr: float  # raw (unnormalised) statistic; 0 when no window passed the cuts


@dataclass
class RunOutput:
    """What one ``run`` returns. Never stored on the shared :class:`Prepared` state."""

    power: Array  # raw, unnormalised power at the segment's frequencies
    best: BestFit | None = None  # exact cores report their best window directly
    n_bands_used: npt.NDArray[np.int64] | None = None


class Prepared:
    """Data prepared once per search execution; backends subclass it with their arrays."""

    def __init__(self, denom: float, labels: tuple[str, ...] | None = None,
                 band_chi2_red: Array | None = None) -> None:
        self.denom = denom  # flat-model chi^2 normalising the raw power
        self.labels = labels
        self.band_chi2_red = band_chi2_red


class Implementation(ABC):
    """One (method, backend, single/multiband) search."""

    method: str
    backend: str
    multiband: bool
    exact: bool  # True: run() returns the best window; False: refine() is needed

    @abstractmethod
    def check_frequencies(self, prep: Prepared, freqs: Array, opts: RunOptions) -> None:
        """Frequency-dependent checks over the whole grid (phase range, int32 sizes)."""

    @abstractmethod
    def run(self, prep: Prepared, freqs: Array, spec: SegmentSpec,
            opts: RunOptions) -> RunOutput:
        """Search ``freqs`` with the segment's duration range."""

    def refine(self, prep: Prepared, freq: float, spec: SegmentSpec,
               opts: RunOptions) -> BestFit:  # pragma: no cover - exact backends
        raise NotImplementedError(f"{type(self).__name__} reports its best fit from run()")


# A segment is the indices (or ``slice(None)`` for "all") of the frequencies it covers.
Segment = tuple["np.ndarray[Any, Any] | slice", SegmentSpec]


def execute(
    impl: Implementation,
    prep: Prepared,
    freqs: Array,
    segments: Sequence[Segment],
    opts: RunOptions,
) -> BLSResult:
    """Run ``impl`` over ``segments`` of ``freqs`` and assemble one :class:`BLSResult`.

    ``power`` follows the caller's frequency order. The best window is the highest-power
    one over all segments; if nothing passed the cuts anywhere, ``best_*`` refer to the
    caller's first frequency (the :class:`BLSResult` contract).
    """
    impl.check_frequencies(prep, freqs, opts)
    nf = freqs.shape[0]
    single = len(segments) == 1 and isinstance(segments[0][0], slice)
    power = np.empty(nf, dtype=np.float64)
    n_used: npt.NDArray[np.int64] | None = None
    have_counts = True
    outs: list[RunOutput] = []
    for sel, spec in segments:
        f_seg = freqs if single else np.ascontiguousarray(freqs[sel])
        out = impl.run(prep, f_seg, spec, opts)
        outs.append(out)
        if out.n_bands_used is None:
            have_counts = False
        if single:
            power = variance_explained(out.power, prep.denom)
        else:
            power[sel] = variance_explained(out.power, prep.denom)
    if have_counts and impl.multiband:
        if single:
            n_used = np.asarray(outs[0].n_bands_used)
        else:
            n_used = np.zeros(nf, dtype=np.int64)
            for (sel, _), out in zip(segments, outs, strict=True):
                n_used[sel] = np.asarray(out.n_bands_used)

    records = tuple(
        SegmentRecord(
            f_min=float(np.min(freqs[sel])), f_max=float(np.max(freqs[sel])),
            n_freqs=int(np.size(freqs[sel])),
            q_min=spec.q_min, q_max=spec.q_max, nbins=spec.nbins)
        for sel, spec in segments
    )

    if impl.exact:
        bests = [o.best for o in outs if o.best is not None]
        best = max(bests, key=lambda b: b.sr)
        if best.sr <= 0.0 and not single:
            # nothing passed anywhere: the first *caller* frequency, with the fields of
            # the segment containing it
            first = _segment_of(segments, 0)
            best = BestFit(float(freqs[0]), bests[first].t0, bests[first].duration,
                           bests[first].depth, 0.0)
        best_power = float(variance_explained(np.array([best.sr]), prep.denom)[0])
        freq, depth = best.frequency, best.depth
        t0, dur = best.t0, best.duration
    else:
        idx = int(np.argmax(power))
        freq = float(freqs[idx])
        spec = segments[_segment_of(segments, idx)][1]
        fit = impl.refine(prep, freq, spec, opts)
        best_power = float(power[idx])
        t0, dur, depth = fit.t0, fit.duration, fit.depth

    return BLSResult(
        frequency=freqs,
        power=power,
        best_frequency=float(freq),
        best_period=float(1.0 / freq),
        best_t0=float(t0),
        best_duration=float(dur),
        best_depth=depth,
        best_power=best_power,
        bands=prep.labels,
        n_bands_used=n_used,
        band_chi2_red=prep.band_chi2_red,
        method=impl.method,
        backend=impl.backend,
        segments=records,
    )


def _segment_of(segments: Sequence[Segment], index: int) -> int:
    """Position in ``segments`` of the segment that contains frequency ``index``."""
    for i, (sel, _) in enumerate(segments):
        if isinstance(sel, slice):
            return i
        if np.any(sel == index):
            return i
    raise ValueError("frequency index is not covered by any segment")  # pragma: no cover
