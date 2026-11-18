"""Sparse (unbinned) searches: box edges, ties and the incremental multiband update."""

from __future__ import annotations

import numpy as np
import pytest

from multiband_bls import (
    eebls,
    multiband_eebls,
    multiband_sparse_bls,
    multiband_sparse_bls_reference,
    sparse_bls,
    sparse_bls_reference,
)
from multiband_bls.reference import _prepare_bands

from .helpers import _TIED_T, _TIED_Y, _make_lightcurve, _wls_box_power


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


def test_multiband_eebls_agrees_with_sparse(lightcurve, freqs):
    """Binned and unbinned multiband searches land on the same period."""
    bands, _ = lightcurve
    ee = multiband_eebls(bands, freqs, nbins=400, q_max=0.12)
    sp = multiband_sparse_bls(bands, freqs, q_max=0.12)
    assert abs(ee.best_frequency - sp.best_frequency) < 1e-3


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


def test_sparse_q_min_bounds_duration(lightcurve, freqs):
    bands, truth = lightcurve
    t, y, dy = bands["r"]
    f = np.array([1.0 / truth.period])
    q_min, q_max = 0.04, 0.08
    for fn, arg in ((sparse_bls, (t, y, dy, f)), (sparse_bls_reference, (t, y, dy, f)),
                    (multiband_sparse_bls, (bands, f)),
                    (multiband_sparse_bls_reference, (bands, f))):
        res = fn(*arg, q_min=q_min, q_max=q_max)
        assert q_min * truth.period <= res.best_duration < q_max * truth.period


def test_multiband_sparse_incremental_matches_bruteforce(lightcurve, freqs):
    """The cached per-band update agrees with re-summing every band per window."""
    bands, _ = lightcurve
    f = freqs[::25]
    res = multiband_sparse_bls(bands, f, q_max=0.12)
    # brute force: best exhaustive window over the merged phase-sorted samples
    from multiband_bls.reference import _merge_band_arrays

    prepared, _ = _prepare_bands(bands, False)
    t_all, wx, w, band, labels, band_w, chi2 = _merge_band_arrays(prepared)
    band_n = np.bincount(band, minlength=len(labels))
    best = np.zeros(f.size)
    for k, fk in enumerate(f):
        phi = (t_all * fk) % 1.0
        o = np.argsort(phi, kind="stable")
        ph, x, ww, bb = phi[o], wx[o], w[o], band[o]
        n = ph.size
        for i1 in range(n):
            if ph[i1 - 1] == ph[i1]:
                continue
            s = np.zeros(len(labels))
            r = np.zeros(len(labels))
            c = np.zeros(len(labels), int)
            for m in range(n - 1):
                i2 = (i1 + m) % n
                s[bb[i2]] += x[i2]
                r[bb[i2]] += ww[i2]
                c[bb[i2]] += 1
                lo = 0.5 * (ph[i1 - 1] + ph[i1]) - (0.5 if i1 == 0 else 0.0)
                hi = 0.5 * (ph[i2] + ph[(i2 + 1) % n]) + (0.5 if (i2 + 1) % n == 0 else 0.0)
                frac = (hi - lo) % 1.0
                if frac >= 0.12:
                    break
                if ph[(i2 + 1) % n] == ph[i2] or frac < 0.01 or m + 1 < 3:
                    continue
                ok = (c >= 1) & (c < band_n) & (r * (1 - r) > 0)
                if ok.any():
                    best[k] = max(best[k], np.sum(band_w[ok] * s[ok] ** 2 / (r[ok] * (1 - r[ok]))))
    np.testing.assert_allclose(res.power, best / chi2, rtol=1e-9, atol=1e-12)
