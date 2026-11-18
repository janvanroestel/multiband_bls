"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from multiband_bls import (
    build_frequency_grid,
)

from .helpers import _make_lightcurve


@pytest.fixture
def lightcurve():
    rng = np.random.default_rng(2024)
    return _make_lightcurve(period=0.17, duration_frac=0.06, rng=rng)


@pytest.fixture
def freqs(lightcurve):
    _, truth = lightcurve
    f_true = 1.0 / truth.period
    return np.arange(f_true - 0.02, f_true + 0.02, 2e-4)


@pytest.fixture
def oversampled_freqs(lightcurve):
    """A properly oversampled grid (3 samples per peak width) around the signal: the
    coarse-to-fine search is only meaningful when skipping samples does not step over
    a peak (the ``freqs`` fixture's step exceeds the peak width)."""
    bands, truth = lightcurve
    t_all = np.concatenate([b[0] for b in bands.values()])
    return build_frequency_grid(t_all, p_min=0.99 * truth.period, p_max=1.01 * truth.period,
                                oversample=3.0, q_min=0.03)
