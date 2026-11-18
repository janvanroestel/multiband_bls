"""CPU fast search (prefix sums, log widths, threads) and the coarse-to-fine frequency search."""

from __future__ import annotations

import numpy as np
import pytest

from multiband_bls import (
    coadd_bands,
    eebls,
    eebls_fast,
    multiband_eebls,
    multiband_eebls_fast,
)

from .helpers import _centered


# --------------------------------------------------------------------------- #
# CPU prefix-sum / log-width search (eebls_fast, multiband_eebls_fast)         #
# --------------------------------------------------------------------------- #
def test_cpu_fast_dense_widths_match_exact(lightcurve, freqs):
    """With dlogq=0 (every integer width) the CPU fast search reproduces the exact
    binned search on median-centred times to double precision."""
    bands, _ = lightcurve
    bands = _centered(bands)
    t, y, dy = coadd_bands(bands)
    t = t - np.median(t)
    exact = eebls(t, y, dy, freqs, nbins=200, q_max=0.12, min_points=3)
    fast = eebls_fast(t, y, dy, freqs, nbins=200, q_max=0.12, min_points=3, dlogq=0)
    np.testing.assert_allclose(fast.power, exact.power, rtol=1e-10, atol=1e-13)
    assert fast.best_frequency == exact.best_frequency

    mexact = multiband_eebls(bands, freqs, nbins=200, q_max=0.12, min_points=3)
    mfast = multiband_eebls_fast(bands, freqs, nbins=200, q_max=0.12, min_points=3, dlogq=0)
    np.testing.assert_allclose(mfast.power, mexact.power, rtol=1e-10, atol=1e-13)
    assert mfast.best_frequency == mexact.best_frequency
    assert mfast.best_t0 == pytest.approx(mexact.best_t0)
    np.testing.assert_allclose(mfast.best_depth, mexact.best_depth)


def test_cpu_fast_peak_close_to_exact(lightcurve, freqs):
    """Default log-spaced widths lose at most a few % of the exact peak power and
    never exceed it."""
    bands, _ = lightcurve
    bands = _centered(bands)
    t, y, dy = coadd_bands(bands)
    t = t - np.median(t)
    exact = eebls(t, y, dy, freqs, nbins=300, q_max=0.12)
    fast = eebls_fast(t, y, dy, freqs, nbins=300, q_max=0.12)
    assert exact.best_power * 0.90 <= fast.best_power <= exact.best_power * (1 + 1e-12)

    mexact = multiband_eebls(bands, freqs, nbins=300, q_max=0.12)
    mfast = multiband_eebls_fast(bands, freqs, nbins=300, q_max=0.12)
    assert mexact.best_power * 0.90 <= mfast.best_power <= mexact.best_power * (1 + 1e-12)


def test_cpu_fast_recovers_period(lightcurve, freqs):
    bands, truth = lightcurve
    t, y, dy = coadd_bands(bands)
    res = eebls_fast(t, y, dy, freqs)
    assert abs(res.best_period - truth.period) / truth.period < 0.01
    assert 0.0 <= res.best_power <= 1.0
    mres = multiband_eebls_fast(bands, freqs)
    assert abs(mres.best_period - truth.period) / truth.period < 0.01
    assert mres.bands == tuple(bands)
    assert mres.n_bands_used is None


@pytest.mark.parametrize("n_threads", [2, 3, 8])
def test_cpu_fast_threads_identical(lightcurve, freqs, n_threads):
    """Splitting the frequency grid over threads gives the identical periodogram."""
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    one = eebls_fast(t, y, dy, freqs, nbins=200)
    many = eebls_fast(t, y, dy, freqs, nbins=200, n_threads=n_threads)
    np.testing.assert_array_equal(many.power, one.power)
    mone = multiband_eebls_fast(bands, freqs, nbins=200)
    mmany = multiband_eebls_fast(bands, freqs, nbins=200, n_threads=n_threads)
    np.testing.assert_array_equal(mmany.power, mone.power)


@pytest.mark.parametrize("min_points", [3, 8])
def test_cpu_fast_count_gate_heteroscedastic(min_points):
    """min_points is an exact in-transit point count (not a weight sum), as in the
    exact cores: the zero/non-zero pattern of the periodogram is the same."""
    rng = np.random.default_rng(23)
    bands = {}
    for b, n, sigma in (("u", 15, 0.003), ("g", 100, 0.03), ("r", 100, 0.03)):
        t = np.sort(rng.uniform(-50.0, 50.0, n))
        bands[b] = (t, rng.normal(20.0, sigma, n), np.full(n, sigma))
    bands = _centered(bands)
    f = np.linspace(0.5, 2.0, 150)
    exact = multiband_eebls(bands, f, nbins=100, q_max=0.12, min_points=min_points)
    fast = multiband_eebls_fast(bands, f, nbins=100, q_max=0.12, min_points=min_points, dlogq=0)
    np.testing.assert_array_equal(fast.power == 0.0, exact.power == 0.0)
    np.testing.assert_allclose(fast.power, exact.power, rtol=1e-10, atol=1e-13)


def test_cpu_fast_rescale_errors(lightcurve, freqs):
    bands, _ = lightcurve
    exact = multiband_eebls(bands, freqs, nbins=200, q_max=0.12, rescale_errors=True)
    fast = multiband_eebls_fast(bands, freqs, nbins=200, q_max=0.12, rescale_errors=True)
    np.testing.assert_allclose(fast.band_chi2_red, exact.band_chi2_red)
    assert fast.best_period == pytest.approx(exact.best_period, rel=1e-3)


# --------------------------------------------------------------------------- #
# CPU fast cores: wrap-around / gate edge cases of the C window sweep          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nbins,q_max", [(20, 0.99), (7, 0.99), (50, 0.5)])
@pytest.mark.parametrize("min_points", [1, 3])
def test_cpu_fast_full_width_and_wrap_match_exact(nbins, q_max, min_points):
    """Widths up to nbins (windows covering the whole circle, every start wrapping)
    on a sparse light curve, where most windows are empty or fail the count gate."""
    rng = np.random.default_rng(5)
    n = 25
    t = np.sort(rng.uniform(-30.0, 30.0, n))
    y = 20.0 + rng.normal(0.0, 0.05, n) + 0.2 * (np.abs(((t / 0.8) % 1.0) - 0.5) < 0.1)
    dy = np.full(n, 0.05)
    f = np.linspace(0.3, 3.0, 120)
    t = t - np.median(t)
    exact = eebls(t, y, dy, f, nbins=nbins, q_min=1.0 / nbins, q_max=q_max,
                  min_points=min_points)
    fast = eebls_fast(t, y, dy, f, nbins=nbins, q_min=1.0 / nbins, q_max=q_max,
                      min_points=min_points, dlogq=0)
    np.testing.assert_allclose(fast.power, exact.power, rtol=1e-10, atol=1e-13)

    bands = {"a": (t, y, dy), "b": (t + 0.013, y + 0.01, dy),
             "c": (t[::2], y[::2], dy[::2])}
    mexact = multiband_eebls(bands, f, nbins=nbins, q_min=1.0 / nbins, q_max=q_max,
                             min_points=min_points)
    mfast = multiband_eebls_fast(bands, f, nbins=nbins, q_min=1.0 / nbins, q_max=q_max,
                                 min_points=min_points, dlogq=0)
    np.testing.assert_allclose(mfast.power, mexact.power, rtol=1e-10, atol=1e-13)


# --------------------------------------------------------------------------- #
# Opt-in coarse-to-fine frequency search                                       #
# --------------------------------------------------------------------------- #
def test_coarse_to_fine_helper_unit():
    from multiband_bls._common import _coarse_to_fine

    f = np.linspace(1.0, 2.0, 1001)
    calls: list[int] = []

    def evaluate(x):
        calls.append(len(x))
        peak = lambda x0: np.exp(-0.5 * ((x - x0) / 0.004) ** 2)  # noqa: E731
        return peak(1.37) + 0.1 * peak(1.8)

    full = evaluate(f)
    calls.clear()
    out = _coarse_to_fine(evaluate, f, 4, 3)
    assert out.shape == f.shape
    assert sum(calls) < 0.5 * f.size
    # the maximum is an exactly evaluated point and equals the full-grid maximum
    assert out.max() == full.max() and np.argmax(out) == np.argmax(full)
    assert np.all(out <= full.max() + 1e-15)
    # coarse_factor=1 and tiny grids fall back to a single full evaluation
    np.testing.assert_array_equal(_coarse_to_fine(evaluate, f, 1, 5), full)
    np.testing.assert_array_equal(_coarse_to_fine(evaluate, f[:10], 4, 5), evaluate(f[:10]))


def test_cpu_fast_coarse_to_fine_recovers_full_peak(lightcurve, oversampled_freqs, monkeypatch):
    """coarse_factor=3 finds the same best frequency/power as the full grid while
    evaluating far fewer frequencies; power stays full length."""
    freqs = oversampled_freqs
    from multiband_bls import _fast

    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    full = eebls_fast(t, y, dy, freqs, nbins=200, q_max=0.12)
    mfull = multiband_eebls_fast(bands, freqs, nbins=200, q_max=0.12)

    evaluated = {"n": 0}
    real_s, real_m = _fast.eebls_fast_grid, _fast.meebls_fast_grid

    def count_s(t_, wx, w, f, *a):
        evaluated["n"] += len(f)
        return real_s(t_, wx, w, f, *a)

    def count_m(t_, wx, w, band, nbd, f, *a):
        evaluated["n"] += len(f)
        return real_m(t_, wx, w, band, nbd, f, *a)

    monkeypatch.setattr(_fast, "eebls_fast_grid", count_s)
    monkeypatch.setattr(_fast, "meebls_fast_grid", count_m)
    coarse = eebls_fast(t, y, dy, freqs, nbins=200, q_max=0.12, coarse_factor=3, n_refine=5)
    assert evaluated["n"] < 0.6 * freqs.size
    evaluated["n"] = 0
    mcoarse = multiband_eebls_fast(bands, freqs, nbins=200, q_max=0.12,
                                   coarse_factor=3, n_refine=5, n_threads=2)
    assert evaluated["n"] < 0.6 * freqs.size

    for c, f in ((coarse, full), (mcoarse, mfull)):
        assert c.power.shape == f.power.shape
        assert c.best_frequency == f.best_frequency
        assert c.best_power == f.best_power
        assert c.best_period == pytest.approx(f.best_period)
        assert np.all(c.power <= f.best_power + 1e-15)
    assert mcoarse.bands == tuple(bands)


def test_coarse_to_fine_descending_grid_matches_ascending(lightcurve, oversampled_freqs):
    """An unsorted/descending grid gives the same periodogram as the ascending one."""
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    kw = {"nbins": 200, "q_max": 0.12, "coarse_factor": 3, "n_refine": 5}
    asc = eebls_fast(t, y, dy, oversampled_freqs, **kw)
    desc = eebls_fast(t, y, dy, oversampled_freqs[::-1], **kw)
    np.testing.assert_array_equal(desc.power, asc.power[::-1])
    assert desc.best_frequency == asc.best_frequency


@pytest.mark.parametrize("offset", [0.0, 2_400_000.0])
def test_fast_refinement_matches_searched_window(lightcurve, freqs, offset):
    """The reported epoch/depth come from the window the (median-centred) search
    scored: non-zero depth and an epoch near the truth, also for JD-like times."""
    bands, truth = lightcurve
    shifted = {k: (t + offset, y, dy) for k, (t, y, dy) in bands.items()}
    t, y, dy = coadd_bands(shifted)
    for res in (eebls_fast(t, y, dy, freqs, q_max=0.12),
                multiband_eebls_fast(shifted, freqs, q_max=0.12)):
        assert res.best_duration > 0.0
        assert np.all(np.asarray(res.best_depth) != 0.0)
        dt = (res.best_t0 - (truth.t0 + offset)) % res.best_period
        assert min(dt, res.best_period - dt) < 0.1 * res.best_period


def test_fast_entry_points_permutation_offset_scale_invariant(lightcurve, freqs):
    """The CPU fast searches depend neither on point/band order, per-band baselines,
    nor a common (y, dy) scale."""
    bands, _ = lightcurve
    rng = np.random.default_rng(8)
    shuffled = {}
    for b in reversed(list(bands)):
        t, y, dy = bands[b]
        o = rng.permutation(len(t))
        shuffled[b] = (t[o], y[o], dy[o])
    shifted = {b: (t, y + 7.0 * i, dy) for i, (b, (t, y, dy)) in enumerate(bands.items())}
    scaled = {b: (t, 3.0 * y, 3.0 * dy) for b, (t, y, dy) in bands.items()}
    kw = {"nbins": 200, "q_max": 0.12}
    ref = multiband_eebls_fast(bands, freqs, **kw).power
    for variant in (shuffled, shifted, scaled):
        np.testing.assert_allclose(multiband_eebls_fast(variant, freqs, **kw).power, ref,
                                   rtol=1e-8, atol=1e-12)
    t, y, dy = coadd_bands(bands)
    sref = eebls_fast(t, y, dy, freqs, **kw).power
    o = rng.permutation(len(t))
    for args in ((t[o], y[o], dy[o]), (t, y + 5.0, dy), (t, 3.0 * y, 3.0 * dy)):
        np.testing.assert_allclose(eebls_fast(*args, freqs, **kw).power, sref,
                                   rtol=1e-8, atol=1e-12)


def test_coarse_to_fine_handles_duplicate_frequencies(lightcurve, freqs):
    from multiband_bls._common import _coarse_to_fine

    calls: list[int] = []

    def evaluate(f: np.ndarray) -> np.ndarray:
        calls.append(f.size)
        return np.sin(f * 7.0) ** 2

    f = np.concatenate([freqs, freqs[::-1], freqs[:10]])  # duplicated and unsorted
    out = _coarse_to_fine(evaluate, f, coarse_factor=4, n_refine=2)
    assert out.shape == f.shape
    np.testing.assert_allclose(out[: freqs.size], out[freqs.size: 2 * freqs.size][::-1])
    np.testing.assert_allclose(out[-10:], out[:10])
