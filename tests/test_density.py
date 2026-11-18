"""Tests for the density-based search (period-dependent duration range)."""

from __future__ import annotations

import numpy as np
import pytest

from multiband_bls import (
    BLSResult,
    density_bls,
    density_duration_bounds,
    density_frequency_grid,
    eebls,
    eebls_fast,
    multiband_eebls,
    multiband_eebls_fast,
    multiband_sparse_bls,
)
from multiband_bls.density import RHO_SUN, central_duration_fraction

RHO = (0.2, 5.0)


def test_central_duration_matches_sun_earth():
    # Earth transit: ~13 h over 365.25 d -> q ~ 1.5e-3 (circular, central)
    q = float(central_duration_fraction(1.0 / 365.25, 1.0))
    assert q == pytest.approx(13.0 / 24.0 / 365.25, rel=0.05)
    assert float(central_duration_fraction(100.0, 1.0)) == pytest.approx(0.5)  # saturates


def test_bounds_scale_with_period_and_density():
    f = np.array([0.1, 1.0, 10.0])
    lo, hi = density_duration_bounds(f, RHO, (0.1, 1.5), q_cap=0.5)
    assert np.all(np.diff(lo) > 0) and np.all(np.diff(hi) > 0)  # shorter P -> larger q
    assert np.all(lo < hi)
    # P^(-2/3) law in the small-q regime
    assert hi[1] / hi[0] == pytest.approx(10 ** (2 / 3), rel=0.05)
    capped = density_duration_bounds(np.array([50.0]), RHO, q_cap=0.2)
    assert capped[1][0] == pytest.approx(0.2)


def test_bad_ranges_rejected():
    f = np.array([1.0])
    for bad in [(0.0, 1.0), (2.0, 1.0), (np.nan, 1.0)]:
        with pytest.raises(ValueError):
            density_duration_bounds(f, bad)
    with pytest.raises(ValueError):
        density_duration_bounds(f, RHO, (0.0, 1.0))
    with pytest.raises(ValueError):
        density_duration_bounds(f, RHO, q_cap=1.5)


def test_grid_is_sparser_than_uniform_and_resolves_narrowest_box():
    t = np.array([0.0, 365.25 * 3])
    grid = density_frequency_grid(t, 0.05, 5.0, RHO)
    assert grid[0] == pytest.approx(0.2) and grid[-1] <= 20.0
    assert np.all(np.diff(grid) > 0)
    q_lo, _ = density_duration_bounds(grid[:-1], RHO)
    step_limit = q_lo / (3.0 * (t[1] - t[0]))
    assert np.all(np.diff(grid) <= step_limit * 1.01)
    # a uniform grid sized for the smallest q_lo over the range needs far more points
    uniform_n = (20.0 - 0.2) / (q_lo.min() / (3.0 * (t[1] - t[0])))
    assert grid.size < uniform_n / 3


class _Recorder:
    """Stand-in search that records the arguments of each call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, t, frequencies, q_min, q_max, nbins=None, min_points=3):
        self.calls.append({"f": frequencies.copy(), "q_min": q_min, "q_max": q_max,
                           "nbins": nbins})
        p = frequencies.copy()
        return BLSResult(frequency=frequencies, power=p, best_frequency=float(p.argmax()),
                         best_period=1.0, best_t0=0.0, best_duration=0.1,
                         best_depth=0.0, best_power=float(p.max()))


def test_segments_cover_grid_and_bound_physical_q():
    rec = _Recorder()
    rng = np.random.default_rng(0)
    freqs = rng.permutation(np.geomspace(0.1, 20.0, 400))  # unsorted input
    res = density_bls(rec, 0.0, frequencies=freqs, density_range=RHO, q_step=1.3)
    assert len(rec.calls) > 3
    assert np.array_equal(np.sort(np.concatenate([c["f"] for c in rec.calls])), np.sort(freqs))
    np.testing.assert_array_equal(res.frequency, freqs)
    np.testing.assert_array_equal(res.power, freqs)  # output follows caller's order
    for c in rec.calls:
        lo, hi = density_duration_bounds(c["f"], RHO)
        assert c["q_min"] <= lo.min() * (1 + 1e-12) and c["q_max"] >= hi.max() * (1 - 1e-12)
        assert hi.max() / hi.min() <= 1.3 * 1.0001 or c["q_max"] == pytest.approx(0.25)
    # long periods get narrower minimum width and so more bins
    nb = [c["nbins"] for c in sorted(rec.calls, key=lambda c: c["f"][0])]
    assert nb == sorted(nb, reverse=True)


def test_rejects_explicit_q():
    with pytest.raises(ValueError, match="q_min"):
        density_bls(_Recorder(), 0.0, frequencies=np.array([1.0]), density_range=RHO, q_min=0.01)


def _transit(period, rho, rng, n=400, sigma=0.01, depth=0.05, baseline=60.0):
    t = np.sort(rng.uniform(0.0, baseline, n))
    q = float(central_duration_fraction(1.0 / period, rho))
    phase = ((t - 0.3) / period + 0.5) % 1.0 - 0.5
    y = 1.0 - depth * (np.abs(phase) < 0.5 * q) + rng.normal(0.0, sigma, n)
    return t, y, np.full(n, sigma), q


@pytest.mark.parametrize("period", [0.7, 1.9, 3.3])
def test_recovers_transit_with_density_grid(period):
    rng = np.random.default_rng(int(period * 10))
    t, y, dy, q = _transit(period, 1.0, rng)
    grid = density_frequency_grid(t, 0.5, 5.0, RHO)
    res = density_bls(eebls_fast, t, y, dy, frequencies=grid, density_range=RHO)
    assert res.best_period == pytest.approx(period, rel=2e-3)
    assert res.best_duration == pytest.approx(q * period, rel=0.5)
    assert res.power.shape == grid.shape


def test_multiband_variants_agree_on_recovery():
    rng = np.random.default_rng(3)
    bands = {}
    for b, depth in {"g": 0.05, "r": 0.03}.items():
        t, y, dy, _ = _transit(2.1, 1.0, rng, n=300, depth=depth)
        bands[b] = (t, y, dy)
    grid = density_frequency_grid(bands["g"][0], 1.0, 4.0, RHO)
    for fn in (multiband_eebls, multiband_eebls_fast, multiband_sparse_bls):
        sub = grid[:: 16 if fn is multiband_sparse_bls else 1]
        res = density_bls(fn, bands, frequencies=sub, density_range=RHO)
        assert res.best_period == pytest.approx(2.1, rel=5e-3), fn.__name__
        assert res.bands == ("g", "r")
        assert np.asarray(res.best_depth).shape == (2,)


def test_rho_sun_constant():
    assert pytest.approx(1.41) == RHO_SUN


def _flat_lc(n=200, baseline=60.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, baseline, n))
    return t, np.ones(n) + rng.normal(0.0, 0.01, n) * 0.0, np.full(n, 0.01)


def test_segment_ends_when_lower_bound_grows_even_if_upper_is_capped():
    # q_hi is pinned at q_cap for both, but q_lo differs ~7x: must not share a segment
    rec = _Recorder()
    f = np.array([2.0, 20.0])
    density_bls(rec, 0.0, frequencies=f, density_range=(0.2, 5.0), q_step=1.25)
    assert len(rec.calls) == 2
    for c in rec.calls:
        lo, hi = density_duration_bounds(c["f"], (0.2, 5.0))
        assert c["q_min"] == pytest.approx(lo.min()) and c["q_max"] >= hi.max() * (1 - 1e-12)


def test_unrepresentable_width_range_is_widened_not_dropped():
    # rho_min == rho_max and eta_lo == eta_hi at 1/day: q ~ 0.0764, 65 bins -> no integer
    # width fits. Each backend must still search, consistently.
    rng = np.random.default_rng(5)
    t, y, dy, _ = _transit(1.0, 1.0, rng, n=300, baseline=40.0)
    f = np.array([1.0])
    res = {}
    for fn in (eebls, eebls_fast):
        res[fn.__name__] = density_bls(fn, t, y, dy, frequencies=f, density_range=(1.0, 1.0),
                                       duration_range=(1.0, 1.0))
    assert res["eebls"].power[0] > 0.0
    assert res["eebls_fast"].power[0] > 0.0


def test_n_bands_used_none_if_any_segment_lacks_counts():
    class Mixed:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, frequencies, q_min, q_max):
            self.n += 1
            counts = np.full(frequencies.size, 3, dtype=np.int64) if self.n == 1 else None
            return BLSResult(frequency=frequencies, power=np.ones(frequencies.size),
                             best_frequency=float(frequencies[0]), best_period=1.0,
                             best_t0=0.0, best_duration=0.1, best_depth=0.0, best_power=1.0,
                             n_bands_used=counts)

    res = density_bls(Mixed(), frequencies=np.array([0.1, 20.0]), density_range=RHO)
    assert res.n_bands_used is None

    class AllCounts(Mixed):
        def __call__(self, frequencies, q_min, q_max):
            r = super().__call__(frequencies, q_min, q_max)
            return BLSResult(**{**r.__dict__, "n_bands_used": np.full(frequencies.size, 2)})

    res2 = density_bls(AllCounts(), frequencies=np.array([20.0, 0.1]), density_range=RHO)
    np.testing.assert_array_equal(res2.n_bands_used, [2, 2])


def test_no_detection_reports_callers_first_frequency():
    t, y, dy = _flat_lc()
    res = density_bls(eebls, t, y, dy, frequencies=np.array([20.0, 2.0]), density_range=RHO)
    assert np.all(res.power == 0.0)
    assert res.best_power == 0.0
    assert res.best_frequency == 20.0
    assert res.best_period == pytest.approx(1.0 / 20.0)
