"""Core search correctness: recovery, binned/sparse agreement, diagnostics and invariances."""

from __future__ import annotations

import numpy as np
import pytest

from multiband_bls import (
    coadd_bands,
    eebls,
    eebls_reference,
    multiband_eebls,
    multiband_eebls_reference,
    multiband_sparse_bls,
    multiband_sparse_bls_reference,
    sparse_bls,
)
from multiband_bls.reference import _prepare_bands, preprocess

from .helpers import _clustered_u_bands, _misscaled_bands, _u_out_of_eclipse


def test_recovers_injected_period(lightcurve, freqs):
    bands, truth = lightcurve
    mc = multiband_sparse_bls(bands, freqs, q_max=0.12)
    # within one frequency bin of the truth
    assert abs(mc.best_frequency - 1.0 / truth.period) < 2e-4
    # per-band depths in the right order of magnitude (u deepest, y shallowest)
    depths = dict(zip(mc.bands, mc.best_depth, strict=True))
    assert depths["u"] > depths["y"]


def test_multiband_recovers_per_band_depths(lightcurve, freqs):
    bands, truth = lightcurve
    mc = multiband_sparse_bls(bands, freqs, q_max=0.12)
    for band, depth in zip(mc.bands, mc.best_depth, strict=True):
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


def test_multiband_eebls_recovers_period(lightcurve, freqs):
    """Binned multiband eeBLS recovers the period and the chromatic depths."""
    bands, truth = lightcurve
    mc = multiband_eebls(bands, freqs, nbins=200, q_max=0.12)
    assert abs(mc.best_frequency - 1.0 / truth.period) < 2e-4
    depths = dict(zip(mc.bands, mc.best_depth, strict=True))
    assert depths["u"] > depths["y"]  # bluest eclipse deepest


def test_multiband_eebls_reduces_to_single_band(lightcurve, freqs):
    """With one band (matched gates), multiband eeBLS == single-band eeBLS."""
    bands, _ = lightcurve
    one = {"r": bands["r"]}
    m = multiband_eebls(one, freqs, nbins=300, q_max=0.12, min_points=1)
    e = eebls(*bands["r"], freqs, nbins=300, q_max=0.12, min_points=1)
    np.testing.assert_allclose(m.power, e.power, rtol=0, atol=1e-9)


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
        depths = dict(zip(res.bands, res.best_depth, strict=True))
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


def test_band_chi2_red_flags_misscaled_band():
    bands = _misscaled_bands()
    freqs = np.linspace(0.5, 2.0, 20)
    for rescale in (False, True):
        res = multiband_eebls(bands, freqs, nbins=100, rescale_errors=rescale)
        chi2 = dict(zip(res.bands, res.band_chi2_red, strict=True))
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
        depths = dict(zip(res.bands, res.best_depth, strict=True))
        assert np.isnan(depths["u"]), fn.__name__
        assert depths["g"] == pytest.approx(0.5, abs=0.05), fn.__name__
        assert res.n_bands_used[0] == 1, fn.__name__
        assert 0.0 <= res.best_power <= 1.0 + 1e-9, fn.__name__


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


def test_binned_durations_within_q_range(lightcurve, freqs):
    bands, truth = lightcurve
    t, y, dy = bands["r"]
    q_max = 0.10
    f = np.array([1.0 / truth.period])
    for res in (eebls(t, y, dy, f, nbins=100, q_max=q_max),
                multiband_eebls(bands, f, nbins=100, q_max=q_max),
                eebls_reference(t, y, dy, f, nbins=100, q_max=q_max),
                multiband_eebls_reference(bands, f, nbins=100, q_max=q_max)):
        assert res.best_duration <= q_max * truth.period * (1 + 1e-9)
