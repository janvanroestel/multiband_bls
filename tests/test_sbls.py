"""Tests for the (multiband) Sparse BLS implementation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

import multiband_bls.gpu as gpu_mod
from multiband_bls import (
    build_frequency_grid,
    coadd_bands,
    eebls,
    eebls_fast,
    eebls_gpu,
    eebls_gpu_fast,
    eebls_reference,
    gpu_available,
    multiband_eebls,
    multiband_eebls_fast,
    multiband_eebls_gpu,
    multiband_eebls_gpu_fast,
    multiband_eebls_reference,
    multiband_sparse_bls,
    multiband_sparse_bls_reference,
    sparse_bls,
    sparse_bls_reference,
)
from multiband_bls.gpu import _log_widths
from multiband_bls.reference import _prepare_bands, auto_nbins, preprocess


@dataclass
class _Truth:
    period: float
    t0: float
    duration_frac: float
    depths: dict
    mean_mags: dict


def _make_lightcurve(period: float, duration_frac: float, rng: np.random.Generator,
                     base_mag: float = 20.5, sigma: float = 0.03,
                     baseline: float = 3 * 365.25) -> tuple:
    """Synthetic 6-band box-eclipse light curve with chromatic depths."""
    t0 = float(rng.uniform(0.0, period))
    depths = {"u": 0.45, "g": 0.30, "r": 0.15, "i": 0.08, "z": 0.04, "y": 0.02}
    mean_mags = {"u": base_mag - 0.4, "g": base_mag - 0.2, "r": base_mag,
                 "i": base_mag + 0.2, "z": base_mag + 0.4, "y": base_mag + 0.6}
    nvisits = {"u": 28, "g": 40, "r": 92, "i": 92, "z": 80, "y": 80}
    bands = {}
    for b, n in nvisits.items():
        t = np.sort(rng.uniform(0.0, baseline, size=n))
        phase = ((t - t0) / period + 0.5) % 1.0 - 0.5
        mag = mean_mags[b] + depths[b] * (np.abs(phase) < 0.5 * duration_frac)
        mag = mag + rng.normal(0.0, sigma, size=n)
        bands[b] = (t, mag, np.full(n, sigma))
    return bands, _Truth(period, t0, duration_frac, depths, mean_mags)


@pytest.fixture
def lightcurve():
    rng = np.random.default_rng(2024)
    return _make_lightcurve(period=0.17, duration_frac=0.06, rng=rng)


@pytest.fixture
def freqs(lightcurve):
    _, truth = lightcurve
    f_true = 1.0 / truth.period
    return np.arange(f_true - 0.02, f_true + 0.02, 2e-4)


def test_cython_matches_reference_single(lightcurve, freqs):
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    rc = sparse_bls(t, y, dy, freqs, q_max=0.12)
    rr = sparse_bls_reference(t, y, dy, freqs, q_max=0.12)
    np.testing.assert_allclose(rc.power, rr.power, rtol=0, atol=1e-9)
    assert rc.best_period == pytest.approx(rr.best_period)


def test_cython_matches_reference_multiband(lightcurve, freqs):
    bands, _ = lightcurve
    mc = multiband_sparse_bls(bands, freqs, q_max=0.12)
    mr = multiband_sparse_bls_reference(bands, freqs, q_max=0.12)
    np.testing.assert_allclose(mc.power, mr.power, rtol=0, atol=1e-9)
    assert mc.best_period == pytest.approx(mr.best_period)
    np.testing.assert_allclose(
        np.asarray(mc.best_depth), np.asarray(mr.best_depth), atol=1e-9
    )
    np.testing.assert_array_equal(mc.n_bands_used, mr.n_bands_used)
    np.testing.assert_array_equal(mc.band_chi2_red, mr.band_chi2_red)


def test_recovers_injected_period(lightcurve, freqs):
    bands, truth = lightcurve
    mc = multiband_sparse_bls(bands, freqs, q_max=0.12)
    # within one frequency bin of the truth
    assert abs(mc.best_frequency - 1.0 / truth.period) < 2e-4
    # per-band depths in the right order of magnitude (u deepest, y shallowest)
    depths = dict(zip(mc.bands, mc.best_depth))
    assert depths["u"] > depths["y"]


def test_multiband_recovers_per_band_depths(lightcurve, freqs):
    bands, truth = lightcurve
    mc = multiband_sparse_bls(bands, freqs, q_max=0.12)
    for band, depth in zip(mc.bands, mc.best_depth):
        # generous tolerance: noisy, sparse photometry
        assert depth == pytest.approx(truth.depths[band], abs=0.15 + 0.3 * truth.depths[band])


def test_eebls_recovers_injected_period(lightcurve):
    """The binned eeBLS recovers the period on a well-sampled single band."""
    bands, truth = lightcurve
    t, y, dy = bands["r"]
    f_true = 1.0 / truth.period
    freqs = np.arange(f_true - 0.02, f_true + 0.02, 1e-4)  # grid spans f_true
    res = eebls(t, y, dy, freqs, nbins=300, q_max=0.15)
    assert abs(res.best_frequency - f_true) < 2e-3
    assert res.best_depth > 0  # dimming (positive depth in magnitudes)


def test_eebls_agrees_with_sparse_bls():
    """Binned eeBLS and unbinned SBLS land on the same period (bright band)."""
    rng = np.random.default_rng(7)
    bands, truth = _make_lightcurve(period=0.19, duration_frac=0.06,
                                    rng=rng, base_mag=18.5, sigma=0.01)
    t, y, dy = bands["r"]  # well-sampled, high-SNR single band
    f_true = 1.0 / truth.period
    freqs = np.arange(f_true - 0.02, f_true + 0.02, 5e-5)
    e = eebls(t, y, dy, freqs, nbins=400, q_max=0.12)
    s = sparse_bls(t, y, dy, freqs, q_max=0.12)
    assert abs(e.best_frequency - f_true) < 2e-3
    assert abs(e.best_frequency - s.best_frequency) < 2e-3


def test_multiband_eebls_matches_reference(lightcurve, freqs):
    """Compiled multiband eeBLS reproduces its pure-Python oracle bit-for-bit."""
    bands, _ = lightcurve
    mc = multiband_eebls(bands, freqs, nbins=200, q_max=0.12)
    mr = multiband_eebls_reference(bands, freqs, nbins=200, q_max=0.12)
    np.testing.assert_allclose(mc.power, mr.power, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(mc.best_depth), np.asarray(mr.best_depth), atol=1e-9
    )
    np.testing.assert_array_equal(mc.n_bands_used, mr.n_bands_used)


def test_multiband_eebls_recovers_period(lightcurve, freqs):
    """Binned multiband eeBLS recovers the period and the chromatic depths."""
    bands, truth = lightcurve
    mc = multiband_eebls(bands, freqs, nbins=200, q_max=0.12)
    assert abs(mc.best_frequency - 1.0 / truth.period) < 2e-4
    depths = dict(zip(mc.bands, mc.best_depth))
    assert depths["u"] > depths["y"]  # bluest eclipse deepest


def test_multiband_eebls_agrees_with_sparse(lightcurve, freqs):
    """Binned and unbinned multiband searches land on the same period."""
    bands, _ = lightcurve
    ee = multiband_eebls(bands, freqs, nbins=400, q_max=0.12)
    sp = multiband_sparse_bls(bands, freqs, q_max=0.12)
    assert abs(ee.best_frequency - sp.best_frequency) < 1e-3


def test_multiband_eebls_reduces_to_single_band(lightcurve, freqs):
    """With one band (matched gates), multiband eeBLS == single-band eeBLS."""
    bands, _ = lightcurve
    one = {"r": bands["r"]}
    m = multiband_eebls(one, freqs, nbins=300, q_max=0.12, min_points=1)
    e = eebls(*bands["r"], freqs, nbins=300, q_max=0.12, min_points=1)
    np.testing.assert_allclose(m.power, e.power, rtol=0, atol=1e-9)


def test_power_is_variance_explained(lightcurve, freqs):
    """Every method returns Delta chi2 / chi2_flat in [0, 1], maxed at the period.

    (Frequency recovery is covered by the dedicated recovery tests; the weak
    co-add baseline is not expected to pin the period.)
    """
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    results = {
        "sparse": sparse_bls(t, y, dy, freqs, q_max=0.12),
        "eebls": eebls(t, y, dy, freqs, nbins=300, q_max=0.12),
        "multi": multiband_sparse_bls(bands, freqs, q_max=0.12),
        "multi_ee": multiband_eebls(bands, freqs, nbins=200, q_max=0.12),
    }
    for name, res in results.items():
        assert res.power.min() >= 0.0, name
        assert res.power.max() <= 1.0 + 1e-9, name
        assert res.best_power == pytest.approx(res.power.max()), name


@pytest.mark.skipif(not gpu_available(), reason="no usable CUDA GPU / CuPy")
def test_gpu_matches_cpu(lightcurve, freqs):
    """GPU RawKernel binned BLS matches the CPU cores (single + multiband)."""
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    sc = eebls(t, y, dy, freqs, nbins=200, q_max=0.12, min_points=3)
    sg = eebls_gpu(t, y, dy, freqs, nbins=200, q_max=0.12, min_points=3)
    np.testing.assert_allclose(sg.power, sc.power, rtol=1e-6, atol=1e-9)
    mc = multiband_eebls(bands, freqs, nbins=200, q_max=0.12)
    mg = multiband_eebls_gpu(bands, freqs, nbins=200, q_max=0.12)
    np.testing.assert_allclose(mg.power, mc.power, rtol=1e-6, atol=1e-9)
    assert mg.best_frequency == pytest.approx(mc.best_frequency)


@pytest.mark.skipif(not gpu_available(), reason="no usable CUDA GPU / CuPy")
def test_gpu_rejects_oversized_nbins(lightcurve, freqs):
    """nbins so large the phase bins exceed device shared memory raises ValueError."""
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    huge = 10 ** 5  # the nbins ceiling; 3 * huge * 8 B = 2.4 MB >> any device's smem
    with pytest.raises(ValueError, match="shared memory"):
        eebls_gpu(t, y, dy, freqs, nbins=huge)
    with pytest.raises(ValueError, match="shared memory"):
        multiband_eebls_gpu(bands, freqs, nbins=huge)


@pytest.mark.skipif(not gpu_available(), reason="no usable CUDA GPU / CuPy")
def test_gpu_multiband_rejects_too_many_bands(freqs):
    """More than the kernel's fixed band cap raises a clear ValueError."""
    rng = np.random.default_rng(1)
    bands = {
        str(i): (np.sort(rng.uniform(0, 100, 20)),
                 rng.normal(20, 0.03, 20), np.full(20, 0.03))
        for i in range(9)  # _MAX_BANDS is 8
    }
    with pytest.raises(ValueError, match="bands"):
        multiband_eebls_gpu(bands, freqs)


def test_matches_astropy_box_least_squares(lightcurve):
    """Peak frequency agrees with astropy's independent BLS on the same data."""
    astropy_bls = pytest.importorskip("astropy.timeseries").BoxLeastSquares
    bands, truth = lightcurve
    t, y, dy = bands["r"]  # single, well-sampled band
    f_true = 1.0 / truth.period
    freqs = np.arange(f_true - 0.02, f_true + 0.02, 1e-4)

    res = sparse_bls(t, y, dy, freqs, q_max=0.15)

    periods = 1.0 / freqs
    durations = np.array([0.02, 0.04, 0.06]) * truth.period
    ap = astropy_bls(t, y, dy).power(periods, durations)
    f_astropy = 1.0 / periods[np.argmax(ap.power)]

    # Sparse BLS pins the true frequency; astropy (coarse duration/phase grid)
    # lands in the same neighbourhood. Both validated against the truth.
    assert abs(res.best_frequency - f_true) < 1e-3
    assert abs(f_astropy - f_true) < 2e-2


def test_eebls_reference_matches_cython(lightcurve, freqs):
    """Pure-Python eebls_reference agrees with compiled eebls to floating-point precision."""
    bands, _ = lightcurve
    t, y, dy = bands["r"]
    ref = eebls_reference(t, y, dy, freqs, nbins=200, q_max=0.12)
    cyt = eebls(t, y, dy, freqs, nbins=200, q_max=0.12)
    np.testing.assert_allclose(ref.power, cyt.power, rtol=0, atol=1e-9)
    assert ref.best_period == pytest.approx(cyt.best_period)


def _centered(bands):
    """Shift all bands' times by the median of the pooled times.

    The fast GPU entry points median-subtract ``t`` internally (float32 range)
    while the exact ones fold raw ``t``; the phase-bin anchor therefore differs
    and binned BLS is sensitive to it. Centering up front makes the fast
    kernels' internal offset zero so exact and fast bin identically.
    """
    off = np.median(np.concatenate([b[0] for b in bands.values()]))
    return {k: (t - off, y, dy) for k, (t, y, dy) in bands.items()}


@pytest.mark.skipif(not gpu_available(), reason="no usable CUDA GPU / CuPy")
def test_gpu_fast_dense_widths_match_exact(lightcurve, freqs):
    """With dlogq -> 0 (all integer widths) the fast kernels reproduce the
    exact GPU periodogram to float32 precision, single band and multiband,
    including the (exact-count) min_points gate."""
    bands, _ = lightcurve
    bands = _centered(bands)
    t, y, dy = coadd_bands(bands)
    exact = eebls_gpu(t, y, dy, freqs, nbins=200, q_max=0.12, min_points=3)
    fast = eebls_gpu_fast(t, y, dy, freqs, nbins=200, q_max=0.12,
                          min_points=3, dlogq=1e-4)
    np.testing.assert_allclose(fast.power, exact.power, rtol=5e-3, atol=5e-5)
    assert fast.best_frequency == pytest.approx(exact.best_frequency)

    mexact = multiband_eebls_gpu(bands, freqs, nbins=200, q_max=0.12,
                                 min_points=3)
    mfast = multiband_eebls_gpu_fast(bands, freqs, nbins=200, q_max=0.12,
                                     min_points=3, dlogq=1e-4)
    np.testing.assert_allclose(mfast.power, mexact.power, rtol=5e-3, atol=5e-5)
    assert mfast.best_frequency == pytest.approx(mexact.best_frequency)


@pytest.mark.skipif(not gpu_available(), reason="no usable CUDA GPU / CuPy")
def test_gpu_fast_peak_close_to_exact(lightcurve, freqs):
    """Default log-spaced widths lose at most a few % of the exact peak power
    (and cannot exceed it beyond float32 noise)."""
    bands, _ = lightcurve
    bands = _centered(bands)
    t, y, dy = coadd_bands(bands)
    exact = eebls_gpu(t, y, dy, freqs, nbins=300, q_max=0.12)
    fast = eebls_gpu_fast(t, y, dy, freqs, nbins=300, q_max=0.12)
    assert fast.best_power <= exact.best_power * (1 + 1e-3)
    assert fast.best_power >= exact.best_power * 0.90

    mexact = multiband_eebls_gpu(bands, freqs, nbins=300, q_max=0.12)
    mfast = multiband_eebls_gpu_fast(bands, freqs, nbins=300, q_max=0.12)
    assert mfast.best_power <= mexact.best_power * (1 + 1e-3)
    assert mfast.best_power >= mexact.best_power * 0.90


@pytest.mark.skipif(not gpu_available(), reason="no usable CUDA GPU / CuPy")
def test_gpu_fast_min_points_gate(lightcurve, freqs):
    """An absurdly high min_points threshold suppresses every window."""
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    res = eebls_gpu_fast(t, y, dy, freqs, nbins=200, q_max=0.12,
                         min_points=10**7)
    assert np.all(res.power == 0.0)
    mres = multiband_eebls_gpu_fast(bands, freqs, nbins=200, q_max=0.12,
                                    min_points=10**7)
    assert np.all(mres.power == 0.0)


@pytest.mark.skipif(not gpu_available(), reason="no usable CUDA GPU / CuPy")
def test_gpu_fast_variants(lightcurve, freqs):
    """eebls_gpu_fast and multiband_eebls_gpu_fast recover the period and return valid power."""
    bands, truth = lightcurve
    t, y, dy = coadd_bands(bands)

    res_fast = eebls_gpu_fast(t, y, dy, freqs)
    assert abs(res_fast.best_period - truth.period) / truth.period < 0.01
    assert 0.0 <= res_fast.best_power <= 1.0

    res_mfast = multiband_eebls_gpu_fast(bands, freqs)
    assert abs(res_mfast.best_period - truth.period) / truth.period < 0.01
    assert 0.0 <= res_mfast.best_power <= 1.0


def test_build_frequency_grid():
    """build_frequency_grid returns a monotone array spanning [f_min, f_max)."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 365.25, 100))
    f_min, f_max = 1.0, 10.0
    freqs = build_frequency_grid(t, f_min=f_min, f_max=f_max)
    assert freqs.ndim == 1
    assert len(freqs) > 0
    assert freqs[0] >= f_min
    assert freqs[-1] < f_max
    assert np.all(np.diff(freqs) > 0)

    # Period interface: p_min=0.1 d → f_max=10, p_max=1.0 d → f_min=1
    freqs_p = build_frequency_grid(t, p_min=0.1, p_max=1.0)
    np.testing.assert_allclose(freqs_p[0], freqs[0], rtol=1e-10)
    np.testing.assert_allclose(freqs_p[-1], freqs[-1], rtol=1e-10)

    # Mixing period and frequency raises
    with pytest.raises(ValueError, match="not both"):
        build_frequency_grid(t, p_min=0.1, p_max=1.0, f_min=1.0, f_max=10.0)
    with pytest.raises(ValueError, match="not both"):
        build_frequency_grid(t, f_max=10.0, p_min=0.1)
    with pytest.raises(ValueError, match="not both"):
        build_frequency_grid(t, f_min=1.0, p_max=1.0)

    # Missing bounds raises
    with pytest.raises(ValueError, match="f_min, f_max"):
        build_frequency_grid(t, f_min=1.0)

    # Inverted range raises
    with pytest.raises(ValueError, match="f_min"):
        build_frequency_grid(t, f_min=10.0, f_max=1.0)


def test_coadd_bands():
    """coadd_bands concatenates bands and mean-subtracts each band's magnitudes."""
    rng = np.random.default_rng(0)
    t1 = np.sort(rng.uniform(0, 100, 10))
    t2 = np.sort(rng.uniform(0, 100, 8))
    y1, dy1 = rng.normal(20.0, 0.01, 10), np.full(10, 0.01)
    y2, dy2 = rng.normal(21.0, 0.02, 8),  np.full(8,  0.02)
    t, y, dy = coadd_bands({"a": (t1, y1, dy1), "b": (t2, y2, dy2)})
    assert len(t) == 18 and len(y) == 18 and len(dy) == 18
    np.testing.assert_array_equal(t[:10], t1)
    np.testing.assert_array_equal(t[10:], t2)
    np.testing.assert_array_equal(dy[:10], dy1)
    np.testing.assert_array_equal(dy[10:], dy2)
    # each band is mean-subtracted: the weighted mean of each segment should be ~0
    assert abs(np.average(y[:10], weights=1 / dy1 ** 2)) < 1e-10
    assert abs(np.average(y[10:], weights=1 / dy2 ** 2)) < 1e-10


def test_auto_nbins_clips():
    """auto_nbins = clip(round(n_res / q_min), 50, 500)."""
    assert auto_nbins(0.1) == 50           # round(50) inside range
    assert auto_nbins(0.01) == 500         # round(500) at the cap
    assert auto_nbins(0.5) == 50           # 10 -> floor clamp
    with pytest.warns(UserWarning, match="capped at 500"):
        assert auto_nbins(1e-4) == 500     # 50000 -> ceil clamp
    assert auto_nbins(0.01, n_res=1) == 100  # n_res scales the target


def test_log_widths_span_sorted_unique():
    """_log_widths spans [kmi, kma], is sorted and unique, and includes both ends."""
    w = _log_widths(3, 60, dlogq=0.3)
    assert w[0] == 3 and w[-1] == 60
    assert list(w) == sorted(set(w.tolist()))
    assert w.min() >= 3 and w.max() <= 60
    # dlogq -> 0 recovers the full linear set kmi..kma
    assert list(_log_widths(3, 10, dlogq=0.0)) == list(range(3, 11))


def test_preprocess_normalisation():
    """preprocess returns unit-sum weights and a zero-weighted-mean signal."""
    rng = np.random.default_rng(3)
    y = rng.normal(15.0, 0.5, 50)
    dy = rng.uniform(0.01, 0.05, 50)
    w_hat, x_tilde, mu = preprocess(y, dy)
    assert w_hat.sum() == pytest.approx(1.0)
    assert np.sum(w_hat * x_tilde) == pytest.approx(0.0, abs=1e-12)
    assert mu == pytest.approx(np.average(y, weights=1 / dy ** 2))


def test_multiband_eebls_reference_auto_nbins():
    """multiband_eebls_reference(nbins=None) auto-selects via auto_nbins."""
    rng = np.random.default_rng(5)
    bands = {b: (np.sort(rng.uniform(0, 50, 20)),
                 rng.normal(20.0, 0.03, 20), np.full(20, 0.03))
             for b in ("g", "r")}
    freqs = np.linspace(1.0, 3.0, 5)
    auto = multiband_eebls_reference(bands, freqs, q_min=0.1, q_max=0.3)
    explicit = multiband_eebls_reference(bands, freqs, nbins=auto_nbins(0.1),
                                         q_min=0.1, q_max=0.3)
    np.testing.assert_array_equal(auto.power, explicit.power)


def test_single_point_light_curve_does_not_crash():
    """A degenerate 1-point light curve yields finite, all-zero power (no divide-by-zero)."""
    t, y, dy = np.array([5.0]), np.array([20.0]), np.array([0.1])
    freqs = np.linspace(1.0, 2.0, 10)
    res = eebls(t, y, dy, freqs, nbins=50, q_max=0.1, min_points=1)
    assert np.all(np.isfinite(res.power))
    assert np.all(res.power == 0.0)  # zero variance -> nothing explained


# --------------------------------------------------------------------------- #
# Input validation                                                             #
# --------------------------------------------------------------------------- #
_GPU = pytest.mark.skipif(not gpu_available(), reason="no usable CUDA GPU / CuPy")
_VFREQS = np.linspace(0.5, 2.0, 5)


def _small_lc(n: int = 40, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 30.0, n))
    return t, rng.normal(20.0, 0.03, n), np.full(n, 0.03)


# (entry point, is_binned)
_SINGLE_ENTRIES = [
    pytest.param(sparse_bls, False, id="sparse_bls"),
    pytest.param(eebls, True, id="eebls"),
    pytest.param(sparse_bls_reference, False, id="sparse_bls_reference"),
    pytest.param(eebls_reference, True, id="eebls_reference"),
    pytest.param(eebls_gpu, True, id="eebls_gpu", marks=_GPU),
    pytest.param(eebls_gpu_fast, True, id="eebls_gpu_fast", marks=_GPU),
    pytest.param(eebls_fast, True, id="eebls_fast"),
]
_MULTI_ENTRIES = [
    pytest.param(multiband_sparse_bls, False, id="multiband_sparse_bls"),
    pytest.param(multiband_eebls, True, id="multiband_eebls"),
    pytest.param(multiband_sparse_bls_reference, False,
                 id="multiband_sparse_bls_reference"),
    pytest.param(multiband_eebls_reference, True, id="multiband_eebls_reference"),
    pytest.param(multiband_eebls_gpu, True, id="multiband_eebls_gpu", marks=_GPU),
    pytest.param(multiband_eebls_gpu_fast, True, id="multiband_eebls_gpu_fast",
                 marks=_GPU),
    pytest.param(multiband_eebls_fast, True, id="multiband_eebls_fast"),
]


def _bad_lc(case: str) -> tuple:
    t, y, dy = _small_lc()
    if case == "nan_y":
        y[3] = np.nan
    elif case == "inf_t":
        t[5] = np.inf
    elif case == "dy_zero":
        dy[2] = 0.0
    elif case == "dy_negative":
        dy[2] = -0.03
    elif case == "length_mismatch":
        y = y[:-1]
    elif case == "empty":
        t, y, dy = t[:0], y[:0], dy[:0]
    elif case == "two_d":
        t, y, dy = t.reshape(4, 10), y.reshape(4, 10), dy.reshape(4, 10)
    elif case == "dy_tiny_all":
        dy = np.full_like(dy, 1e-200)
    elif case == "dy_tiny_one":
        dy[0] = 1e-200
    elif case == "dy_huge_all":
        dy = np.full_like(dy, 1e200)
    elif case == "dy_huge_one":
        dy[0] = 1e200
    elif case == "y_huge":
        y[::2], y[1::2] = 1.5e308, -1.5e308   # finite, but x_tilde**2 overflows
    return t, y, dy


_BAD_LC_CASES = ["nan_y", "inf_t", "dy_zero", "dy_negative", "length_mismatch",
                 "empty", "two_d", "dy_tiny_all", "dy_tiny_one", "dy_huge_all",
                 "dy_huge_one", "y_huge"]

# (kwargs, needs a binned entry point)
_BAD_PARAMS = [
    pytest.param({"frequencies": -_VFREQS}, False, id="negative_freq"),
    pytest.param({"frequencies": np.r_[0.0, _VFREQS]}, False, id="zero_freq"),
    pytest.param({"frequencies": np.r_[np.nan, _VFREQS]}, False, id="nan_freq"),
    pytest.param({"frequencies": np.array([])}, False, id="empty_freq"),
    pytest.param({"q_max": 0.0}, False, id="q_max_zero"),
    pytest.param({"q_max": 1.0}, False, id="q_max_one"),
    pytest.param({"q_max": 1.5}, False, id="q_max_gt_one"),
    pytest.param({"q_max": np.nan}, False, id="q_max_nan"),
    pytest.param({"min_points": 0}, False, id="min_points_zero"),
    pytest.param({"min_points": 1.9}, False, id="min_points_float"),
    pytest.param({"min_points": np.nan}, False, id="min_points_nan"),
    pytest.param({"min_points": np.inf}, False, id="min_points_inf"),
    pytest.param({"min_points": True}, False, id="min_points_bool"),
    pytest.param({"q_min": 0.0}, True, id="q_min_zero"),
    pytest.param({"q_min": -0.01}, True, id="q_min_negative"),
    pytest.param({"q_min": np.nan}, True, id="q_min_nan"),
    pytest.param({"q_min": 0.2, "q_max": 0.1}, True, id="q_min_gt_q_max"),
    pytest.param({"nbins": 0}, True, id="nbins_zero"),
    pytest.param({"nbins": 50.9}, True, id="nbins_float"),
    pytest.param({"nbins": np.inf}, True, id="nbins_inf"),
    pytest.param({"nbins": 10**9}, True, id="nbins_huge"),
]


@pytest.mark.parametrize("case", _BAD_LC_CASES)
@pytest.mark.parametrize(("fn", "binned"), _SINGLE_ENTRIES)
def test_single_band_rejects_bad_light_curve(fn, binned, case):
    t, y, dy = _bad_lc(case)
    with pytest.raises(ValueError):
        fn(t, y, dy, _VFREQS)


@pytest.mark.parametrize("case", _BAD_LC_CASES)
@pytest.mark.parametrize(("fn", "binned"), _MULTI_ENTRIES)
def test_multiband_rejects_bad_band_naming_it(fn, binned, case):
    bands = {"g": _small_lc(seed=1), "u": _bad_lc(case)}
    with pytest.raises(ValueError, match="'u'"):
        fn(bands, _VFREQS)


@pytest.mark.parametrize(("kwargs", "needs_binned"), _BAD_PARAMS)
@pytest.mark.parametrize(("fn", "binned"), _SINGLE_ENTRIES)
def test_single_band_rejects_bad_params(fn, binned, kwargs, needs_binned):
    if needs_binned and not binned:
        pytest.skip("parameter only exists for binned entry points")
    kwargs = {"frequencies": _VFREQS, **kwargs}
    with pytest.raises(ValueError):
        fn(*_small_lc(), **kwargs)


@pytest.mark.parametrize(("kwargs", "needs_binned"), _BAD_PARAMS)
@pytest.mark.parametrize(("fn", "binned"), _MULTI_ENTRIES)
def test_multiband_rejects_bad_params(fn, binned, kwargs, needs_binned):
    if needs_binned and not binned:
        pytest.skip("parameter only exists for binned entry points")
    kwargs = {"frequencies": _VFREQS, **kwargs}
    with pytest.raises(ValueError):
        fn({"g": _small_lc(seed=1), "r": _small_lc(seed=2)}, **kwargs)


@pytest.mark.parametrize(("fn", "binned"), _MULTI_ENTRIES)
def test_multiband_rejects_empty_mapping(fn, binned):
    with pytest.raises(ValueError, match="at least one band"):
        fn({}, _VFREQS)


@pytest.mark.parametrize(("fn", "binned"), _SINGLE_ENTRIES)
def test_single_band_rejects_overflowing_phase(fn, binned):
    t, y, dy = _small_lc()
    with pytest.raises(ValueError, match="cycles"):
        fn(t + 1e300, y, dy, np.array([1e10]))      # |t| * f ~ 1e310 -> overflow
    with pytest.raises(ValueError, match="cycles"):
        fn(t * 1e14, y, dy, _VFREQS)                # ~1e15 cycles: no phase precision


@pytest.mark.parametrize(("fn", "binned"), _MULTI_ENTRIES)
def test_multiband_rejects_overflowing_phase(fn, binned):
    t, y, dy = _small_lc()
    bands = {"g": _small_lc(seed=1), "u": (t + 1e300, y, dy)}
    with pytest.raises(ValueError, match="cycles"):
        fn(bands, np.array([1e10]))


@_GPU
def test_gpu_fast_rejects_float32_overflow():
    t, y, dy = _small_lc()
    t_wide = np.linspace(0.0, 1e35, t.shape[0])     # centred |t| ~ 5e34 > 1e30
    freqs = np.array([1e-25, 2e-25])                # |t| * f ~ 1e10: fine in float64
    eebls_gpu(t_wide, y, dy, freqs, nbins=50)       # exact (float64) path accepts it
    with pytest.raises(ValueError, match="float32"):
        eebls_gpu_fast(t_wide, y, dy, freqs, nbins=50)
    with pytest.raises(ValueError, match="float32"):
        multiband_eebls_gpu_fast({"g": (t_wide, y, dy)}, freqs, nbins=50)


@_GPU
def test_gpu_fast_rejects_too_many_phase_cycles():
    """Each of |t| and f fits float32, but their product exceeds the float-float
    fold's accuracy range; the exact (float64) GPU path still accepts it."""
    t, y, dy = _small_lc()
    t_wide = np.linspace(-5e3, 5e3, t.shape[0])     # centred |t| = 5e3 d
    freqs = np.array([1e5, 1.1e5])                  # 5.5e8 cycles > 1e8, < 1e15
    eebls_gpu(t_wide, y, dy, freqs, nbins=50)
    with pytest.raises(ValueError, match="cycles"):
        eebls_gpu_fast(t_wide, y, dy, freqs, nbins=50)
    with pytest.raises(ValueError, match="cycles"):
        multiband_eebls_gpu_fast({"g": (t_wide, y, dy)}, freqs, nbins=50)
    # just inside the limit is accepted
    eebls_gpu_fast(t_wide, y, dy, np.array([1e4, 1.9e4]), nbins=50)


@_GPU
@pytest.mark.parametrize("dlogq", [np.nan, np.inf, -np.inf, -0.1, True, "0.3", None])
def test_gpu_fast_rejects_invalid_dlogq(dlogq):
    t, y, dy = _small_lc()
    for call in (lambda: eebls_gpu_fast(t, y, dy, _VFREQS, nbins=50, dlogq=dlogq),
                 lambda: multiband_eebls_gpu_fast({"g": (t, y, dy)}, _VFREQS, nbins=50,
                                                  dlogq=dlogq)):
        with pytest.raises(ValueError, match="dlogq"):
            call()


@_GPU
def test_gpu_fast_dlogq_zero_is_linear_search():
    t, y, dy = _small_lc()
    lin = eebls_gpu_fast(t, y, dy, _VFREQS, nbins=50, dlogq=0)
    dense = eebls_gpu_fast(t, y, dy, _VFREQS, nbins=50, dlogq=1e-6)
    np.testing.assert_allclose(lin.power, dense.power, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(("fn", "binned"), _SINGLE_ENTRIES)
def test_single_band_huge_min_points_rejects_every_window(fn, binned):
    res = fn(*_small_lc(), _VFREQS, min_points=10**12)   # beyond C int: capped
    assert np.all(res.power == 0.0)


@pytest.mark.parametrize(("fn", "binned"), _MULTI_ENTRIES)
def test_multiband_huge_min_points_rejects_every_window(fn, binned):
    bands = {"g": _small_lc(seed=1), "r": _small_lc(seed=2)}
    res = fn(bands, _VFREQS, min_points=10**12)
    assert np.all(res.power == 0.0)


def test_coadd_bands_validates():
    with pytest.raises(ValueError, match="'u'"):
        coadd_bands({"g": _small_lc(), "u": _bad_lc("nan_y")})


def test_auto_nbins_rejects_bad_q_min():
    for bad in (0.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            auto_nbins(bad)


def test_auto_nbins_warns_below_cliff():
    with pytest.warns(UserWarning, match="capped at 500"):
        assert auto_nbins(0.005) == 500
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert auto_nbins(0.01) == 500


# --------------------------------------------------------------------------- #
# NaN depths, n_bands_used                                                     #
# --------------------------------------------------------------------------- #
def _u_out_of_eclipse(lightcurve):
    """The fixture with every u point within 2 durations of mid-eclipse removed."""
    bands, truth = lightcurve
    t, y, dy = bands["u"]
    phase = ((t - truth.t0) / truth.period + 0.5) % 1.0 - 0.5
    keep = np.abs(phase) > 2.0 * truth.duration_frac
    out = dict(bands)
    out["u"] = (t[keep], y[keep], dy[keep])
    return out, truth


def test_unmeasured_band_depth_is_nan(lightcurve):
    bands, truth = _u_out_of_eclipse(lightcurve)
    freqs = np.array([1.0 / truth.period])
    results = {
        "multiband_sparse_bls": multiband_sparse_bls(bands, freqs, q_max=0.12),
        "multiband_sparse_bls_reference": multiband_sparse_bls_reference(
            bands, freqs, q_max=0.12),
        "multiband_eebls": multiband_eebls(bands, freqs, nbins=200, q_max=0.12),
        "multiband_eebls_reference": multiband_eebls_reference(
            bands, freqs, nbins=200, q_max=0.12),
    }
    for name, res in results.items():
        depths = dict(zip(res.bands, res.best_depth))
        assert np.isnan(depths["u"]), name
        assert all(np.isfinite(d) for b, d in depths.items() if b != "u"), name
        assert res.n_bands_used[0] == len(bands) - 1, name


def test_n_bands_used_range(lightcurve, freqs):
    bands, _ = lightcurve
    for res in (multiband_sparse_bls(bands, freqs, q_max=0.12),
                multiband_eebls(bands, freqs, nbins=200, q_max=0.12)):
        k = res.n_bands_used
        assert k is not None and k.dtype == np.int64 and k.shape == freqs.shape
        assert np.all((k >= 0) & (k <= len(bands)))
        assert np.all(k[res.power > 0] >= 1)
    t, y, dy = coadd_bands(bands)
    assert sparse_bls(t, y, dy, freqs).n_bands_used is None
    assert eebls(t, y, dy, freqs).n_bands_used is None


def test_flat_light_curve_selects_no_window():
    rng = np.random.default_rng(11)
    # y == 0 exactly: the weighted mean is exactly 0, so x_tilde is exactly 0
    # (a constant 20.0 leaves ~1e-15 rounding residue and a tiny chi2_flat).
    bands = {b: (np.sort(rng.uniform(0, 30, 30)), np.zeros(30), np.full(30, 0.03))
             for b in ("g", "r")}
    freqs = np.linspace(0.5, 2.0, 4)
    for fn, kw in ((multiband_sparse_bls, {}),
                   (multiband_sparse_bls_reference, {}),
                   (multiband_eebls, {"nbins": 50}),
                   (multiband_eebls_reference, {"nbins": 50})):
        res = fn(bands, freqs, **kw)
        assert np.all(res.power == 0.0), fn.__name__
        assert np.all(res.n_bands_used == 0), fn.__name__
        assert np.all(np.isnan(res.best_depth)), fn.__name__


# --------------------------------------------------------------------------- #
# Per-band error-scale diagnostics and rescaling                               #
# --------------------------------------------------------------------------- #
def _misscaled_bands(n: int = 300, seed: int = 4) -> dict:
    """Three flat bands; 'u' has its sigma under-reported by 2x."""
    rng = np.random.default_rng(seed)
    out = {}
    for b, true_sigma in (("u", 0.06), ("g", 0.03), ("r", 0.03)):
        t = np.sort(rng.uniform(0, 100, n))
        out[b] = (t, rng.normal(20.0, true_sigma, n), np.full(n, 0.03))
    return out


def test_band_chi2_red_flags_misscaled_band():
    bands = _misscaled_bands()
    freqs = np.linspace(0.5, 2.0, 20)
    for rescale in (False, True):
        res = multiband_eebls(bands, freqs, nbins=100, rescale_errors=rescale)
        chi2 = dict(zip(res.bands, res.band_chi2_red))
        assert chi2["u"] == pytest.approx(4.0, rel=0.2)   # pre-rescaling value
        assert chi2["g"] == pytest.approx(1.0, rel=0.2)
        assert chi2["r"] == pytest.approx(1.0, rel=0.2)


def test_prepare_bands_rescales_to_unit_chi2():
    bands = _misscaled_bands()
    prepared, chi2_red = _prepare_bands(bands, rescale_errors=True)
    for bi, (label, (_t, y, dy)) in enumerate(prepared.items()):
        np.testing.assert_allclose(dy, bands[label][2] * np.sqrt(chi2_red[bi]))
        _, x_tilde, _ = preprocess(y, dy)
        assert np.sum((x_tilde / dy) ** 2) / (len(y) - 1) == pytest.approx(1.0, rel=1e-9)


def test_rescale_errors_default_off():
    bands = _misscaled_bands()
    freqs = np.linspace(0.5, 2.0, 20)
    a = multiband_eebls(bands, freqs, nbins=100)
    b = multiband_eebls(bands, freqs, nbins=100, rescale_errors=False)
    c = multiband_eebls(bands, freqs, nbins=100, rescale_errors=True)
    np.testing.assert_array_equal(a.power, b.power)
    assert not np.array_equal(a.power, c.power)


def test_rescale_skips_zero_variance_band():
    bands = _misscaled_bands(n=60)
    t, _, dy = bands["g"]
    bands["g"] = (t, np.full_like(t, 20.0), dy)          # zero scatter
    with pytest.warns(UserWarning, match="'g'"):
        prepared, _ = _prepare_bands(bands, rescale_errors=True)
    np.testing.assert_array_equal(prepared["g"][2], dy)
    freqs = np.linspace(0.5, 2.0, 10)
    for fn, kw in ((multiband_eebls, {"nbins": 60}), (multiband_sparse_bls, {})):
        with pytest.warns(UserWarning, match="'g'"):
            res = fn(bands, freqs, rescale_errors=True, **kw)
        assert np.all(np.isfinite(res.power)), fn.__name__
        assert np.isfinite(res.best_power) and np.isfinite(res.best_t0)


@_GPU
def test_gpu_rescale_matches_cpu():
    bands = _misscaled_bands()
    freqs = np.linspace(0.5, 2.0, 200)
    cpu = multiband_eebls(bands, freqs, nbins=100, rescale_errors=True)
    gpu = multiband_eebls_gpu(bands, freqs, nbins=100, rescale_errors=True)
    np.testing.assert_allclose(gpu.power, cpu.power, rtol=1e-6, atol=1e-9)
    assert gpu.best_frequency == pytest.approx(cpu.best_frequency)
    at_best = multiband_eebls(bands, np.array([gpu.best_frequency]), nbins=100,
                              rescale_errors=True)
    assert gpu.best_t0 == pytest.approx(at_best.best_t0)
    assert gpu.best_duration == pytest.approx(at_best.best_duration)
    np.testing.assert_allclose(gpu.best_depth, at_best.best_depth)
    fast = multiband_eebls_gpu_fast(bands, freqs, nbins=100, rescale_errors=True)
    ref = multiband_eebls_reference(bands, freqs[:3], nbins=100, rescale_errors=True)
    for res in (gpu, fast, ref):
        np.testing.assert_array_equal(res.band_chi2_red, cpu.band_chi2_red)


# --------------------------------------------------------------------------- #
# Fast-kernel exact point-count gate                                           #
# --------------------------------------------------------------------------- #
def _old_min_r_disagrees(t, w_hat, n_bands, freq, nbins, kmi, kma,
                         min_points) -> bool:
    """True if, at ``freq``, some window passes the removed weighted-fraction
    gate (sum w_hat >= min_points * n_bands / n) but fails the exact count."""
    n = t.shape[0]
    ib = np.minimum((nbins * ((t * freq) % 1.0)).astype(int), nbins - 1)
    rbin = np.bincount(ib, weights=w_hat, minlength=nbins)
    cbin = np.bincount(ib, minlength=nbins)
    min_r = min_points * n_bands / n
    for i1 in range(nbins):
        for width in range(kmi, kma + 1):
            idx = (i1 + np.arange(width)) % nbins
            if rbin[idx].sum() >= min_r and cbin[idx].sum() < min_points:
                return True
    return False


@_GPU
@pytest.mark.parametrize("min_points", [3, 8])
def test_gpu_fast_count_gate_heteroscedastic_single(min_points):
    rng = np.random.default_rng(21)
    n = 200
    t = np.sort(rng.uniform(-50.0, 50.0, n))
    t -= np.median(t)
    dy = np.full(n, 0.03)
    dy[rng.choice(n, 20, replace=False)] = 0.003      # 20 points, 100x weight
    y = rng.normal(20.0, dy)
    freqs = np.linspace(0.5, 2.0, 150)
    nbins, q_max = 100, 0.12
    w_hat, _, _ = preprocess(y, dy)
    kmi, kma = gpu_mod._grid_params(nbins, 0.01, q_max)
    assert _old_min_r_disagrees(t, w_hat, 1, freqs[0], nbins, kmi, kma, min_points)
    exact = eebls_gpu(t, y, dy, freqs, nbins=nbins, q_max=q_max, min_points=min_points)
    fast = eebls_gpu_fast(t, y, dy, freqs, nbins=nbins, q_max=q_max,
                          min_points=min_points, dlogq=1e-4)
    np.testing.assert_array_equal(fast.power == 0.0, exact.power == 0.0)
    np.testing.assert_allclose(fast.power, exact.power, rtol=5e-3, atol=5e-5)


@_GPU
@pytest.mark.parametrize("min_points", [3, 8])
def test_gpu_fast_count_gate_heteroscedastic_multiband(min_points):
    rng = np.random.default_rng(22)
    bands = {}
    for b, n, sigma in (("u", 15, 0.003), ("g", 100, 0.03), ("r", 100, 0.03)):
        t = np.sort(rng.uniform(-50.0, 50.0, n))
        bands[b] = (t, rng.normal(20.0, sigma, n), np.full(n, sigma))
    bands = _centered(bands)
    freqs = np.linspace(0.5, 2.0, 150)
    nbins, q_max = 100, 0.12
    t_all = np.concatenate([v[0] for v in bands.values()])
    w_all = np.concatenate([preprocess(v[1], v[2])[0] for v in bands.values()])
    kmi, kma = gpu_mod._grid_params(nbins, 0.01, q_max)
    assert _old_min_r_disagrees(t_all, w_all, len(bands), freqs[0], nbins,
                                kmi, kma, min_points)
    exact = multiband_eebls_gpu(bands, freqs, nbins=nbins, q_max=q_max,
                                min_points=min_points)
    fast = multiband_eebls_gpu_fast(bands, freqs, nbins=nbins, q_max=q_max,
                                    min_points=min_points, dlogq=1e-4)
    np.testing.assert_array_equal(fast.power == 0.0, exact.power == 0.0)
    np.testing.assert_allclose(fast.power, exact.power, rtol=5e-3, atol=5e-5)


@_GPU
def test_gpu_fast_rejects_too_many_points(monkeypatch, lightcurve, freqs):
    bands, _ = lightcurve
    monkeypatch.setattr(gpu_mod, "_MAX_FAST_POINTS", 10)
    with pytest.raises(ValueError, match="fewer than 10 points"):
        eebls_gpu_fast(*bands["r"], freqs)
    with pytest.raises(ValueError, match="fewer than 10 points"):
        multiband_eebls_gpu_fast(bands, freqs)


@_GPU
@pytest.mark.parametrize("too_many", ["points", "frequencies"])
def test_gpu_rejects_counts_beyond_int32(monkeypatch, lightcurve, freqs, too_many):
    """Point/frequency counts are C ints in every kernel; larger ones must raise
    instead of wrapping (limit monkeypatched rather than allocating 2**31)."""
    bands, _ = lightcurve
    t, y, dy = bands["r"]
    if too_many == "points":
        limit, f = t.size - 1, freqs[:5]
    else:
        limit, f = freqs.size - 1, freqs
    monkeypatch.setattr(gpu_mod, "_MAX_INT32", limit)
    for call in (lambda: eebls_gpu(t, y, dy, f),
                 lambda: eebls_gpu_fast(t, y, dy, f),
                 lambda: multiband_eebls_gpu({"r": (t, y, dy)}, f),
                 lambda: multiband_eebls_gpu_fast({"r": (t, y, dy)}, f)):
        with pytest.raises(ValueError, match="32-bit kernel indices"):
            call()


# --------------------------------------------------------------------------- #
# Numerical robustness (pre-release review): constants, missing baseline, ties #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("fn", "binned"), _SINGLE_ENTRIES)
def test_constant_nonzero_signal_gives_zero_power(fn, binned):
    """A constant magnitude-20 curve (not 0) has no signal: power must be exactly 0.

    sum(w_hat * y) = 20 * (1 +- eps) used to leave a ~1e-15 residue that produced
    power ~0.3-0.5 (sparse) or ~9e15 (binned, all-points window).
    """
    kw = {"nbins": 100} if binned else {}
    t = np.linspace(0.0, 0.05, 10)                   # clustered: one window holds all
    res = fn(t, np.full(10, 20.0), np.full(10, 0.03), np.array([1.0]), q_max=0.1, **kw)
    assert np.all(res.power == 0.0)
    t, _, dy = _small_lc()
    res = fn(t, np.full_like(t, 20.0), dy, _VFREQS, **kw)
    assert np.all(res.power == 0.0)


@pytest.mark.parametrize(("fn", "binned"), _MULTI_ENTRIES)
def test_multiband_constant_bands_give_zero_power(fn, binned):
    (tg, _, dyg), (tr, _, dyr) = _small_lc(seed=1), _small_lc(seed=2)
    bands = {"g": (tg, np.full_like(tg, 20.0), dyg), "r": (tr, np.full_like(tr, 19.0), dyr)}
    assert np.all(fn(bands, _VFREQS).power == 0.0)


@pytest.mark.parametrize(("fn", "binned"), _SINGLE_ENTRIES)
def test_window_holding_every_point_is_rejected(fn, binned):
    """All points in one short phase range: a window containing all of them has
    no baseline, so it must not be scored (power stays a valid fraction)."""
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 0.05, 10)
    y = rng.normal(20.0, 0.03, 10)
    kw = {"nbins": 100} if binned else {}
    res = fn(t, y, np.full(10, 0.03), np.array([1.0]), q_max=0.1, **kw)
    assert 0.0 <= res.best_power <= 1.0 + 1e-9
    assert np.isfinite(res.best_depth)


def _clustered_u_bands():
    """'g' samples the whole phase with an eclipse at phase 0.5; every 'u' point
    sits inside the eclipse, so u has no out-of-eclipse baseline."""
    rng = np.random.default_rng(8)
    tg = np.sort(rng.uniform(0.0, 50.0, 200))
    yg = 20.0 + 0.5 * (np.abs((tg % 1.0) - 0.5) < 0.03) + rng.normal(0, 0.01, 200)
    tu = 10.0 + 0.5 + np.linspace(-0.01, 0.01, 6)    # all within the eclipse
    yu = 21.0 + rng.normal(0, 0.01, 6)
    return {"g": (tg, yg, np.full(200, 0.01)), "u": (tu, yu, np.full(6, 0.01))}


def test_band_without_baseline_is_not_measured():
    """A band whose every point is in the best window gets NaN depth and does not
    count towards n_bands_used (r(1-r) > 0 alone fails to rounding there)."""
    bands = _clustered_u_bands()
    freqs = np.array([1.0])
    for fn, kw in ((multiband_sparse_bls, {}),
                   (multiband_sparse_bls_reference, {}),
                   (multiband_eebls, {"nbins": 200}),
                   (multiband_eebls_reference, {"nbins": 200})):
        res = fn(bands, freqs, q_max=0.1, **kw)
        depths = dict(zip(res.bands, res.best_depth))
        assert np.isnan(depths["u"]), fn.__name__
        assert depths["g"] == pytest.approx(0.5, abs=0.05), fn.__name__
        assert res.n_bands_used[0] == 1, fn.__name__
        assert 0.0 <= res.best_power <= 1.0 + 1e-9, fn.__name__


@_GPU
def test_gpu_band_without_baseline_matches_cpu():
    bands = _clustered_u_bands()
    freqs = np.linspace(0.9, 1.1, 41)
    cpu = multiband_eebls(bands, freqs, nbins=200, q_max=0.1)
    gpu = multiband_eebls_gpu(bands, freqs, nbins=200, q_max=0.1)
    np.testing.assert_allclose(gpu.power, cpu.power, rtol=1e-6, atol=1e-9)
    assert np.all(gpu.power <= 1.0 + 1e-9)
    fast = multiband_eebls_gpu_fast(bands, freqs, nbins=200, q_max=0.1)
    assert np.all(fast.power <= 1.0 + 1e-6)
    i = int(np.argmin(np.abs(freqs - 1.0)))
    assert gpu.n_bands_used[i] == cpu.n_bands_used[i] == 1


@_GPU
def test_gpu_n_bands_used_matches_cpu(lightcurve, freqs):
    """Exact GPU kernel reports the CPU's k; the fast kernel does when asked and
    when it searches the same windows (centred t, every width)."""
    bands, _ = lightcurve
    bands = _centered(bands)
    kw = {"nbins": 200, "q_max": 0.12, "min_points": 3}
    cpu = multiband_eebls(bands, freqs, **kw)
    gpu = multiband_eebls_gpu(bands, freqs, **kw)
    assert gpu.n_bands_used.dtype == np.int64
    np.testing.assert_array_equal(gpu.n_bands_used, cpu.n_bands_used)

    fast = multiband_eebls_gpu_fast(bands, freqs, dlogq=1e-4,
                                    compute_n_bands_used=False, **kw)
    assert fast.n_bands_used is None  # opt-out
    fast_k = multiband_eebls_gpu_fast(bands, freqs, dlogq=1e-4, **kw)
    assert fast_k.n_bands_used.dtype == np.int64
    # float32 near-ties may pick another window at a few sub-peak frequencies
    assert np.mean(fast_k.n_bands_used == cpu.n_bands_used) > 0.99
    i = int(np.argmax(cpu.power))
    assert fast_k.n_bands_used[i] == cpu.n_bands_used[i]
    # the k variant computes the same periodogram
    np.testing.assert_allclose(fast_k.power, fast.power, rtol=1e-5, atol=1e-7)
    assert fast_k.best_frequency == fast.best_frequency
    # default (log-spaced) widths: k still within range and >= 1 where power > 0
    k = multiband_eebls_gpu_fast(bands, freqs, **kw).n_bands_used
    assert np.all((k >= 0) & (k <= len(bands)))


@_GPU
def test_gpu_flat_light_curve_n_bands_used_zero():
    rng = np.random.default_rng(11)
    bands = {b: (np.sort(rng.uniform(0, 30, 30)), np.zeros(30), np.full(30, 0.03))
             for b in ("g", "r")}
    freqs = np.linspace(0.5, 2.0, 4)
    for res in (multiband_eebls_gpu(bands, freqs, nbins=50),
                multiband_eebls_gpu_fast(bands, freqs, nbins=50)):
        assert np.all(res.power == 0.0)
        assert np.all(res.n_bands_used == 0)


_TIED_T = np.array([0.0, 0.05, 0.05, 0.05, 0.1, 0.4, 0.6, 0.8])
_TIED_Y = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("fn", [sparse_bls, sparse_bls_reference])
def test_sparse_finds_eclipse_with_tied_phases(fn):
    """Identical phases used to be read as a phase wrap (huge duration, early exit)."""
    res = fn(_TIED_T, _TIED_Y, np.ones(8), np.array([1.0]), q_max=0.15)
    assert res.best_power == pytest.approx(1.0)
    assert res.best_depth == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [multiband_sparse_bls, multiband_sparse_bls_reference])
def test_multiband_sparse_finds_eclipse_with_tied_phases(fn):
    res = fn({"a": (_TIED_T, _TIED_Y, np.ones(8))}, np.array([1.0]), q_max=0.15)
    assert res.best_power == pytest.approx(1.0)


def test_simultaneous_bands_are_never_split():
    """Bands observed at identical times: points at one phase must share in/out
    membership, so the result equals that of the tie-free reference oracle."""
    rng = np.random.default_rng(12)
    t = np.sort(rng.uniform(0.0, 20.0, 60))
    ecl = np.abs((t / 0.7 % 1.0) - 0.3) < 0.04
    bands = {b: (t, 20.0 + d * ecl + rng.normal(0, 0.02, 60), np.full(60, 0.02))
             for b, d in (("g", 0.4), ("r", 0.2))}
    freqs = np.array([1 / 0.7])
    mc = multiband_sparse_bls(bands, freqs, q_max=0.15)
    mr = multiband_sparse_bls_reference(bands, freqs, q_max=0.15)
    np.testing.assert_allclose(mc.power, mr.power, rtol=0, atol=1e-12)
    assert mc.best_power > 0.8
    # the best box contains whole tie groups: same membership in both bands
    phase = ((t - mc.best_t0) * freqs[0] + 0.5) % 1.0 - 0.5
    inside = np.abs(phase) < 0.5 * mc.best_duration * freqs[0]
    assert inside.sum() >= 3


# --------------------------------------------------------------------------- #
# Invariances and an independent weighted-least-squares oracle                 #
# --------------------------------------------------------------------------- #
def test_permutation_invariance(lightcurve, freqs):
    bands, _ = lightcurve
    rng = np.random.default_rng(5)
    shuffled = {}
    for b in reversed(list(bands)):                  # band order reversed too
        t, y, dy = bands[b]
        o = rng.permutation(len(t))
        shuffled[b] = (t[o], y[o], dy[o])
    for fn, kw in ((multiband_sparse_bls, {}), (multiband_eebls, {"nbins": 200})):
        a = fn(bands, freqs, q_max=0.12, **kw)
        b = fn(shuffled, freqs, q_max=0.12, **kw)
        np.testing.assert_allclose(a.power, b.power, rtol=1e-9, atol=1e-12)
    t, y, dy = coadd_bands(bands)
    o = rng.permutation(len(t))
    for fn in (sparse_bls, eebls):
        np.testing.assert_allclose(fn(t, y, dy, freqs).power,
                                   fn(t[o], y[o], dy[o], freqs).power,
                                   rtol=1e-9, atol=1e-12)


def test_offset_and_scale_invariance(lightcurve, freqs):
    """power depends neither on per-band baselines nor on a common scale of (y, dy)."""
    bands, _ = lightcurve
    shifted = {b: (t, y + 7.0 * i, dy) for i, (b, (t, y, dy)) in enumerate(bands.items())}
    scaled = {b: (t, 3.0 * y, 3.0 * dy) for b, (t, y, dy) in bands.items()}
    for fn, kw in ((multiband_sparse_bls, {}), (multiband_eebls, {"nbins": 200})):
        ref = fn(bands, freqs, q_max=0.12, **kw).power
        np.testing.assert_allclose(fn(shifted, freqs, q_max=0.12, **kw).power, ref,
                                   rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(fn(scaled, freqs, q_max=0.12, **kw).power, ref,
                                   rtol=1e-9, atol=1e-12)


def _wls_box_power(bands, period, t0, duration):
    """Independent oracle: fit baseline_b + depth_b * in_box by explicit weighted
    least squares per band and return (Delta chi2 / chi2_flat, depths)."""
    d_chi2 = chi2_flat = 0.0
    depths = []
    for t, y, dy in bands.values():
        phase = ((t - t0) / period + 0.5) % 1.0 - 0.5
        inside = (np.abs(phase) < 0.5 * duration / period).astype(float)
        w = 1.0 / dy ** 2
        sw = np.sqrt(w)
        flat = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        if 0 < inside.sum() < len(t):
            design = np.column_stack([np.ones_like(t), inside]) * sw[:, None]
            coef, *_ = np.linalg.lstsq(design, y * sw, rcond=None)
            resid = np.sum(w * (y - coef[0] - coef[1] * inside) ** 2)
            depths.append(coef[1])
        else:
            resid = flat
            depths.append(np.nan)
        d_chi2 += flat - resid
        chi2_flat += flat
    return d_chi2 / chi2_flat, np.array(depths)


def test_sparse_best_box_matches_weighted_least_squares(lightcurve, freqs):
    """The reported best power and depths equal an explicit WLS box fit at the
    reported (period, t0, duration): validates the objective, not just agreement
    between implementations that share code."""
    bands, _ = lightcurve
    mc = multiband_sparse_bls(bands, freqs, q_max=0.12)
    power, depths = _wls_box_power(bands, mc.best_period, mc.best_t0, mc.best_duration)
    assert mc.best_power == pytest.approx(power, rel=1e-9)
    np.testing.assert_allclose(mc.best_depth, depths, rtol=1e-9)

    t, y, dy = bands["r"]
    sc = sparse_bls(t, y, dy, freqs, q_max=0.12)
    power, depths = _wls_box_power({"r": (t, y, dy)}, sc.best_period, sc.best_t0,
                                   sc.best_duration)
    assert sc.best_power == pytest.approx(power, rel=1e-9)
    assert sc.best_depth == pytest.approx(depths[0], rel=1e-9)


@pytest.mark.parametrize("kwargs", [
    {"f_min": -1.0, "f_max": 2.0},
    {"f_min": 1.0, "f_max": np.inf},
    {"p_min": -0.5, "p_max": 1.0},
    {"p_min": 0.0, "p_max": 1.0},
    {"f_min": 1.0, "f_max": 2.0, "df": -1e-3},
    {"f_min": 1.0, "f_max": 2.0, "df": 0.0},
    {"f_min": 1.0, "f_max": 2.0, "q_min": -0.01},
    {"f_min": 1.0, "f_max": 2.0, "oversample": 0.0},
    {"f_min": 1.0, "f_max": 2.0, "df": 1e-3, "q_min": np.nan},   # explicit df too
    {"f_min": 1.0, "f_max": 2.0, "df": 1e-3, "oversample": 0.0},
])
def test_build_frequency_grid_rejects_bad_bounds(kwargs):
    t = np.linspace(0.0, 100.0, 50)
    with pytest.raises(ValueError):
        build_frequency_grid(t, **kwargs)


def test_build_frequency_grid_rejects_non_finite_times():
    with pytest.raises(ValueError, match="finite"):
        build_frequency_grid(np.array([0.0, np.nan, 10.0]), f_min=1.0, f_max=2.0)


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


@pytest.mark.parametrize("n_threads", [0, -1, 1.5, True, "2"])
def test_cpu_fast_rejects_invalid_n_threads(n_threads):
    t, y, dy = _small_lc()
    with pytest.raises(ValueError, match="n_threads"):
        eebls_fast(t, y, dy, _VFREQS, n_threads=n_threads)
    with pytest.raises(ValueError, match="n_threads"):
        multiband_eebls_fast({"g": (t, y, dy)}, _VFREQS, n_threads=n_threads)


@pytest.mark.parametrize("dlogq", [-0.1, np.nan, np.inf, True, "0.3"])
def test_cpu_fast_rejects_invalid_dlogq(dlogq):
    t, y, dy = _small_lc()
    with pytest.raises(ValueError, match="dlogq"):
        eebls_fast(t, y, dy, _VFREQS, dlogq=dlogq)
    with pytest.raises(ValueError, match="dlogq"):
        multiband_eebls_fast({"g": (t, y, dy)}, _VFREQS, dlogq=dlogq)


@_GPU
def test_cpu_fast_matches_gpu_fast(lightcurve, freqs):
    """The CPU and GPU fast searches evaluate the same windows: equal periodograms to
    the GPU's float32 precision."""
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    cpu = eebls_fast(t, y, dy, freqs, nbins=200, q_max=0.12)
    gpu = eebls_gpu_fast(t, y, dy, freqs, nbins=200, q_max=0.12)
    np.testing.assert_allclose(cpu.power, gpu.power, rtol=5e-3, atol=5e-5)
    mcpu = multiband_eebls_fast(bands, freqs, nbins=200, q_max=0.12)
    mgpu = multiband_eebls_gpu_fast(bands, freqs, nbins=200, q_max=0.12)
    np.testing.assert_allclose(mcpu.power, mgpu.power, rtol=5e-3, atol=5e-5)
    assert mcpu.best_frequency == pytest.approx(mgpu.best_frequency)
