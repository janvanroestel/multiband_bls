"""Golden-output parity: every public entry point must reproduce recorded results."""

from __future__ import annotations

import numpy as np
import pytest

import multiband_bls as mb

from .golden_cases import GOLDEN, cases, flatten, gpu_cases

_GOLDEN = np.load(GOLDEN)
_CPU = cases()
_GPU = gpu_cases()


def _expected(name: str) -> dict[str, np.ndarray]:
    prefix = f"{name}/"
    return {k[len(prefix):]: _GOLDEN[k] for k in _GOLDEN.files if k.startswith(prefix)}


@pytest.mark.parametrize("name", sorted(_CPU))
def test_cpu_matches_golden(name):
    got, want = flatten(_CPU[name]()), _expected(name)
    assert set(got) == set(want)
    for key, w in want.items():
        if np.issubdtype(np.asarray(w).dtype, np.floating):
            # last-bit differences between compilers/NumPy builds are not regressions
            np.testing.assert_allclose(got[key], w, rtol=1e-12, atol=1e-15,
                                       err_msg=f"{name}/{key}")
        else:
            np.testing.assert_array_equal(got[key], w, err_msg=f"{name}/{key}")


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(_GPU))
def test_gpu_matches_golden(name):
    if not mb.gpu_available():
        pytest.skip("no GPU")
    want = _expected(name)
    if not want:
        pytest.skip("golden file recorded without a GPU")
    got = flatten(_GPU[name]())
    assert set(got) == set(want)
    # shared-memory atomics make summation order (and the last bits) nondeterministic
    np.testing.assert_allclose(got["power"], want["power"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(got["best"], want["best"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(got["best_depth"], want["best_depth"], rtol=1e-3, atol=1e-6)
    for key in ("n_bands_used", "band_chi2_red", "bands", "frequency"):
        if key in want:
            np.testing.assert_array_equal(got[key], want[key], err_msg=f"{name}/{key}")
