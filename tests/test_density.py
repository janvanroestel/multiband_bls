"""Tests for the density-based search (period-dependent duration range)."""

from __future__ import annotations

import numpy as np
import pytest

from multiband_bls import (
    DensityPrior,
    SearchConfig,
    bls,
    density_duration_bounds,
    density_frequency_grid,
)
from multiband_bls.density import RHO_SUN, central_duration_fraction
from multiband_bls.engine import (
    BestFit,
    Implementation,
    RunOutput,
    SegmentSpec,
    execute,
)

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


def _prior(**kw):
    return DensityPrior(RHO, **kw)


def test_segments_cover_grid_and_bound_physical_q():
    rng = np.random.default_rng(0)
    freqs = rng.permutation(np.geomspace(0.1, 20.0, 400))  # unsorted input
    segs = _prior(q_step=1.3).segments(freqs, binned=True)
    assert len(segs) > 3
    idx = np.concatenate([i for i, _ in segs])
    assert np.array_equal(np.sort(idx), np.arange(freqs.size))  # every frequency, once
    for sel, spec in segs:
        lo, hi = density_duration_bounds(freqs[sel], RHO)
        assert spec.q_min <= lo.min() * (1 + 1e-12) and spec.q_max >= hi.max() * (1 - 1e-12)
        assert hi.max() / hi.min() <= 1.3 * 1.0001 or spec.q_max == pytest.approx(0.25)
    # long periods get a narrower minimum width and so more bins
    nb = [spec.nbins for _, spec in sorted(segs, key=lambda x: freqs[x[0]][0])]
    assert nb == sorted(nb, reverse=True)
    assert all(spec.nbins is None for _, spec in _prior().segments(freqs, binned=False))


def test_segment_ends_when_lower_bound_grows_even_if_upper_is_capped():
    # q_hi is pinned at q_cap for both, but q_lo differs ~7x: must not share a segment
    segs = _prior().segments(np.array([2.0, 20.0]), binned=True)
    assert len(segs) == 2
    for sel, spec in segs:
        lo, hi = density_duration_bounds(np.array([2.0, 20.0])[sel], RHO)
        assert spec.q_min == pytest.approx(lo.min()) and spec.q_max >= hi.max() * (1 - 1e-12)


def test_unrepresentable_width_range_is_widened():
    # q ~ 0.0764 at 1/day with 65 bins: no integer width fits, so q_max moves up to one
    prior = DensityPrior((1.0, 1.0), (1.0, 1.0))
    ((_, spec),) = prior.segments(np.array([1.0]), binned=True)
    assert spec.q_max > spec.q_min
    assert np.ceil(spec.q_min * spec.nbins - 1e-9) <= np.floor(spec.q_max * spec.nbins + 1e-9)


def test_config_exclusivity_and_defaults():
    with pytest.raises(ValueError, match="density_prior"):
        SearchConfig(q_min=0.01, density_prior=_prior())
    with pytest.raises(ValueError, match="density_prior"):
        SearchConfig(q_max=0.1, density_prior=_prior())
    assert SearchConfig().resolve_q("sparse") == (0.01, 0.15)
    assert SearchConfig().resolve_q("binned") == (0.01, 0.10)
    assert SearchConfig(q_min=0.02, q_max=0.2).resolve_q("fast") == (0.02, 0.2)
    with pytest.raises(ValueError, match="q_step"):
        DensityPrior(RHO, q_step=1.0)


class _Fake(Implementation):
    """Stand-in backend recording what the executor asks of it."""

    method, backend, multiband, exact = "binned", "cpu", True, True

    def __init__(self, counts: list[bool], power=None) -> None:
        self.counts, self.calls, self.n = counts, [], 0
        self.power = power

    def check_frequencies(self, prep, freqs, opts):
        pass

    def run(self, prep, freqs, spec, opts):
        has = self.counts[min(self.n, len(self.counts) - 1)]
        self.n += 1
        self.calls.append((freqs.copy(), spec))
        p = freqs.copy() if self.power is None else self.power(freqs)
        k = np.full(freqs.size, 3, dtype=np.int64) if has else None
        best = float(p.max())
        return RunOutput(p, BestFit(float(freqs[int(np.argmax(p))]), 0.1, 0.01,
                                    np.array([0.1]), best), k)


def _prep(denom=1.0):
    from multiband_bls.engine import Prepared
    return Prepared(denom)


def test_executor_n_bands_used_is_all_or_nothing():
    f = np.array([20.0, 0.1])
    segs = _prior().segments(f, binned=True)
    assert len(segs) == 2
    from multiband_bls.engine import RunOptions
    res = execute(_Fake([True, False]), _prep(), f, segs, RunOptions())
    assert res.n_bands_used is None
    res = execute(_Fake([True]), _prep(), f, segs, RunOptions())
    np.testing.assert_array_equal(res.n_bands_used, [3, 3])
    assert res.method == "binned" and res.backend == "cpu"
    assert res.segments is not None and len(res.segments) == 2


def test_executor_no_detection_reports_callers_first_frequency():
    from multiband_bls.engine import RunOptions
    f = np.array([20.0, 2.0])
    segs = _prior().segments(f, binned=True)
    res = execute(_Fake([False], power=np.zeros_like), _prep(), f, segs, RunOptions())
    assert np.all(res.power == 0.0) and not res.has_solution
    assert res.best_frequency == 20.0 and res.best_period == pytest.approx(0.05)


def test_executor_orders_power_like_the_caller_and_keeps_best():
    from multiband_bls.engine import RunOptions
    f = np.array([5.0, 0.2, 1.0, 12.0])
    segs = _prior().segments(f, binned=True)
    res = execute(_Fake([False]), _prep(), f, segs, RunOptions())
    np.testing.assert_array_equal(res.power, f)  # power[i] belongs to frequency[i]
    assert res.best_frequency == 12.0 and res.has_solution


def test_single_segment_executor_passes_frequencies_through_untouched():
    from multiband_bls.engine import RunOptions
    fake = _Fake([False])
    f = np.array([3.0, 1.0, 2.0])  # unsorted, one fixed-q segment
    execute(fake, _prep(), f, [(slice(None), SegmentSpec(0.01, 0.1, 100))], RunOptions())
    assert fake.calls[0][0] is f or np.array_equal(fake.calls[0][0], f)


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
    cfg = SearchConfig(density_prior=_prior())
    res = bls((t, y, dy), p_min=0.5, p_max=5.0, method="fast", backend="cpu", config=cfg)
    assert res.best_period == pytest.approx(period, rel=2e-3)
    assert res.best_duration == pytest.approx(q * period, rel=0.5)
    assert res.power.shape == res.frequency.shape
    assert res.segments is not None and len(res.segments) > 1


def test_multiband_methods_recover_with_density_prior():
    rng = np.random.default_rng(3)
    bands = {}
    for b, depth in {"g": 0.05, "r": 0.03}.items():
        t, y, dy, _ = _transit(2.1, 1.0, rng, n=300, depth=depth)
        bands[b] = (t, y, dy)
    grid = density_frequency_grid(bands["g"][0], 1.0, 4.0, RHO)
    cfg = SearchConfig(density_prior=_prior())
    for method, stride in (("binned", 1), ("fast", 1), ("sparse", 16)):
        res = bls(bands, grid[::stride], method=method, backend="cpu", config=cfg)
        assert res.best_period == pytest.approx(2.1, rel=5e-3), method
        assert res.bands == ("g", "r")
        assert np.asarray(res.best_depth).shape == (2,)


def test_rho_sun_constant():
    assert pytest.approx(1.41) == RHO_SUN


def test_q_floor_clamps_lower_bound_and_coarsens_grid():
    f = np.array([0.01, 0.1, 1.0, 10.0])
    free_lo, _ = density_duration_bounds(f, RHO)
    lo, hi = density_duration_bounds(f, RHO, q_floor=0.01)
    assert np.all(lo >= 0.01) and np.all(hi >= lo)
    assert lo[0] == pytest.approx(0.01) and free_lo[0] < 0.01  # floor binds at long period
    assert np.allclose(lo[-1], free_lo[-1])                      # and not at short period
    prior = DensityPrior(RHO, q_floor=0.01)
    assert np.all(prior.bounds(f)[0] >= 0.01)
    t = np.linspace(0.0, 30.0, 200)
    assert prior.grid(t, 0.5, 100.0).size < DensityPrior(RHO).grid(t, 0.5, 100.0).size
    for seg in prior.segments(1.0 / np.geomspace(0.5, 100.0, 500), binned=False)[0:]:
        assert seg[1].q_min >= 0.01
    for bad in (0.0, -1.0, 0.3):  # <= 0 or above q_cap (0.25)
        with pytest.raises(ValueError, match="q_floor"):
            DensityPrior(RHO, q_floor=bad)
        with pytest.raises(ValueError, match="q_floor"):
            density_duration_bounds(f, RHO, q_floor=bad)
    assert DensityPrior(RHO).q_floor is None


@pytest.mark.parametrize("method", ["binned", "fast", "sparse"])
def test_collapsed_range_at_cap_is_searchable_and_respects_cap(method):
    """q_floor == q_cap leaves a single allowed width: widening must stay <= q_cap and the
    sparse search (half-open [q_min, q_max)) must still admit windows."""
    prior = DensityPrior((0.2, 5.0), q_floor=0.25, q_cap=0.25)
    freqs = np.array([0.5, 1.0, 2.0])
    for _, spec in prior.segments(freqs, binned=method != "sparse"):
        assert spec.q_max <= 0.25 + 1e-12 and spec.q_min < spec.q_max and spec.widened
    # quarter-period eclipse at P = 1 d, one band and two
    t = np.arange(0.0, 20.0, 0.01)
    y = np.where((t % 1.0) < 0.25, 1.0, 0.0)
    dy = np.ones_like(t)
    for data in ((t, y, dy), {"a": (t, y, dy), "b": (t + 0.005, y, dy)}):
        res = bls(data, freqs, method=method, backend="cpu",
                  config=SearchConfig(density_prior=prior, min_points=1))
        assert res.power[1] > 0.5 and res.best_period == pytest.approx(1.0)
        for rec in res.segments:
            assert rec.q_max <= 0.25 + 1e-12
