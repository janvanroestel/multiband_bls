"""Reference implementations, helpers (grids, nbins, widths, co-add) and agreement with them."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from multiband_bls import (
    build_frequency_grid,
    coadd_bands,
    eebls,
    eebls_reference,
    multiband_eebls,
    multiband_eebls_reference,
    multiband_sparse_bls,
    multiband_sparse_bls_reference,
    sparse_bls,
    sparse_bls_reference,
)
from multiband_bls.gpu import _log_widths
from multiband_bls.reference import auto_nbins, preprocess


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


def test_auto_nbins_warns_below_cliff():
    with pytest.warns(UserWarning, match="capped at 500"):
        assert auto_nbins(0.005) == 500
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert auto_nbins(0.01) == 500


def test_width_bounds_respect_q_range():
    from multiband_bls.reference import width_bounds

    assert width_bounds(100, 0.01, 0.10) == (1, 10)
    assert width_bounds(100, 0.07, 0.07) == (7, 7)  # float noise: 0.07 * 100 > 7
    assert width_bounds(500, 0.0125, 0.1) == (7, 50)  # ceil(6.25)
    with pytest.raises(ValueError, match="no window width"):
        width_bounds(50, 0.011, 0.019)


def test_reference_accepts_sign(lightcurve, freqs):
    bands, _ = lightcurve
    t, y, dy = bands["r"]
    f = freqs[::20]
    for sgn in ("both", "positive", "negative"):
        np.testing.assert_allclose(eebls_reference(t, y, dy, f, sign=sgn).power,
                                   eebls(t, y, dy, f, sign=sgn).power, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(multiband_eebls_reference(bands, f, sign=sgn).power,
                                   multiband_eebls(bands, f, sign=sgn).power,
                                   rtol=1e-9, atol=1e-12)
    with pytest.raises(ValueError, match="sign"):
        eebls_reference(t, y, dy, f, sign="up")


@pytest.mark.parametrize("sgn", ["both", "positive", "negative"])
def test_sparse_sign_and_q_min_match_reference(lightcurve, freqs, sgn):
    bands, _ = lightcurve
    t, y, dy = bands["r"]
    f = freqs[::10]
    kw = {"q_min": 0.03, "q_max": 0.12, "sign": sgn}
    rc, rr = sparse_bls(t, y, dy, f, **kw), sparse_bls_reference(t, y, dy, f, **kw)
    np.testing.assert_allclose(rc.power, rr.power, rtol=1e-9, atol=1e-12)
    mc, mr = multiband_sparse_bls(bands, f, **kw), multiband_sparse_bls_reference(bands, f, **kw)
    np.testing.assert_allclose(mc.power, mr.power, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(mc.n_bands_used, mr.n_bands_used)
    np.testing.assert_allclose(mc.best_depth, mr.best_depth, rtol=1e-9, equal_nan=True)
