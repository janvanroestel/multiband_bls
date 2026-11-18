"""Seeded cases that pin the numerical behaviour of every public search entry point.

``python -m tests.golden_cases`` (re)writes ``tests/data/golden_v0.npz`` from the current
code; ``tests/test_golden.py`` compares later code against that file. Regenerate only
when a numerical change is intended and explained in the commit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import multiband_bls as mb
from multiband_bls import BLSResult, DensityPrior, SearchConfig, bls

GOLDEN = Path(__file__).parent / "data" / "golden_v0.npz"
PERIOD = 0.37
_DENSITY = SearchConfig(density_prior=DensityPrior((0.2, 5.0)))


def _bands() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(1234)
    t0, q = 0.11, 0.08
    out = {}
    for b, (n, depth, sig) in {"g": (70, 0.20, 0.03), "r": (90, 0.12, 0.02),
                               "i": (60, 0.06, 0.04)}.items():
        t = np.sort(rng.uniform(0.0, 120.0, n))
        phase = ((t - t0) / PERIOD + 0.5) % 1.0 - 0.5
        y = 15.0 + depth * (np.abs(phase) < 0.5 * q) + rng.normal(0.0, sig, n)
        out[b] = (t, y, np.full(n, sig * rng.uniform(0.8, 1.2)))
    return out


def _freqs() -> np.ndarray:
    f0 = 1.0 / PERIOD
    return np.linspace(f0 - 0.05, f0 + 0.05, 240)


def _single() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _bands()["r"]


def cases() -> dict[str, Callable[[], BLSResult]]:
    b, f, s = _bands(), _freqs(), _single()
    c: dict[str, Callable[[], BLSResult]] = {
        "sparse": lambda: mb.sparse_bls(*s, f),
        "sparse_neg": lambda: mb.sparse_bls(*s, f, sign="negative", q_min=0.02),
        "eebls": lambda: mb.eebls(*s, f),
        "eebls_pos": lambda: mb.eebls(*s, f, sign="positive", nbins=120),
        "eebls_fast": lambda: mb.eebls_fast(*s, f),
        "eebls_fast_coarse": lambda: mb.eebls_fast(*s, f, coarse_factor=3, n_refine=5),
        "msparse": lambda: mb.multiband_sparse_bls(b, f),
        "msparse_rescale": lambda: mb.multiband_sparse_bls(b, f, rescale_errors=True),
        "meebls": lambda: mb.multiband_eebls(b, f),
        "meebls_rescale": lambda: mb.multiband_eebls(b, f, rescale_errors=True, sign="negative"),
        "meebls_fast": lambda: mb.multiband_eebls_fast(b, f),
        "meebls_fast_rescale": lambda: mb.multiband_eebls_fast(b, f, rescale_errors=True),
        # recorded from the pre-engine density_bls wrapper: the new bls() path must match
        "density_eebls": lambda: bls(s, f, method="binned", backend="cpu", config=_DENSITY),
        "density_meebls_fast": lambda: bls(b, f, method="fast", backend="cpu",
                                           config=_DENSITY),
    }
    return c


def gpu_cases() -> dict[str, Callable[[], BLSResult]]:
    b, f, s = _bands(), _freqs(), _single()
    return {
        "gpu_eebls": lambda: mb.eebls_gpu(*s, f),
        "gpu_eebls_fast": lambda: mb.eebls_gpu_fast(*s, f),
        "gpu_meebls": lambda: mb.multiband_eebls_gpu(b, f),
        "gpu_meebls_fast": lambda: mb.multiband_eebls_gpu_fast(b, f),
        "gpu_meebls_fast_rescale": lambda: mb.multiband_eebls_gpu_fast(b, f, rescale_errors=True),
    }


def flatten(res: BLSResult) -> dict[str, Any]:
    """Every public numeric field of a result as plain arrays (absent fields omitted)."""
    out: dict[str, Any] = {
        "frequency": res.frequency, "power": res.power,
        "best": np.array([res.best_frequency, res.best_period, res.best_t0,
                          res.best_duration, res.best_power]),
        "best_depth": np.atleast_1d(np.asarray(res.best_depth, dtype=float)),
    }
    if res.n_bands_used is not None:
        out["n_bands_used"] = np.asarray(res.n_bands_used)
    if res.band_chi2_red is not None:
        out["band_chi2_red"] = np.asarray(res.band_chi2_red)
    if res.bands is not None:
        out["bands"] = np.array(res.bands)
    return out


def main() -> None:
    data: dict[str, Any] = {}
    allcases = dict(cases())
    if mb.gpu_available():
        allcases.update(gpu_cases())
    for name, fn in allcases.items():
        for key, val in flatten(fn()).items():
            data[f"{name}/{key}"] = val
    GOLDEN.parent.mkdir(exist_ok=True)
    np.savez_compressed(GOLDEN, **data)
    print(f"wrote {GOLDEN} ({len(allcases)} cases)")


if __name__ == "__main__":
    main()
