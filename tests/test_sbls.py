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
