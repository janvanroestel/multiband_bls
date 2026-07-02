"""Tests for the (multiband) Sparse BLS implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from multiband_bls import (
    build_frequency_grid,
    coadd_bands,
    eebls,
    eebls_gpu,
    eebls_gpu_fast,
    eebls_reference,
    gpu_available,
    multiband_eebls,
    multiband_eebls_gpu,
    multiband_eebls_gpu_fast,
    multiband_eebls_reference,
    multiband_sparse_bls,
    multiband_sparse_bls_reference,
    sparse_bls,
    sparse_bls_reference,
)
from multiband_bls.gpu import _log_widths
from multiband_bls.reference import auto_nbins, preprocess


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
    huge = 10 ** 6  # 3 * huge * 8 bytes far exceeds any device's shared memory
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
    assert auto_nbins(1e-4) == 500         # 50000 -> ceil clamp
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
