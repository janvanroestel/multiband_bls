"""Input validation and degenerate-input behaviour of every entry point."""

from __future__ import annotations

import numpy as np
import pytest

from multiband_bls import (
    build_frequency_grid,
    coadd_bands,
    eebls,
    eebls_fast,
    multiband_eebls,
    multiband_eebls_fast,
    multiband_eebls_reference,
    multiband_sparse_bls,
    multiband_sparse_bls_reference,
    sparse_bls,
    sparse_bls_reference,
)
from multiband_bls.reference import auto_nbins

from .helpers import (
    _BAD_LC_CASES,
    _BAD_PARAMS,
    _MULTI_ENTRIES,
    _SINGLE_ENTRIES,
    _VFREQS,
    _bad_lc,
    _signed_single,
    _small_lc,
)


def test_single_point_light_curve_does_not_crash():
    """A degenerate 1-point light curve yields finite, all-zero power (no divide-by-zero)."""
    t, y, dy = np.array([5.0]), np.array([20.0]), np.array([0.1])
    freqs = np.linspace(1.0, 2.0, 10)
    res = eebls(t, y, dy, freqs, nbins=50, q_max=0.1, min_points=1)
    assert np.all(np.isfinite(res.power))
    assert np.all(res.power == 0.0)  # zero variance -> nothing explained


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


def test_cpu_fast_rejects_int32_phase_overflow_in_core():
    """The vectorised fold truncates t * f to int32; the core refuses larger phases
    (the public entry points already stop at 1e8 cycles)."""
    from multiband_bls import _fast

    t = np.array([0.0, 1.0e9, -1.0e9])
    wx = np.zeros(3)
    w = np.full(3, 1.0 / 3.0)
    with pytest.raises(ValueError, match=r"2\*\*30"):
        _fast.eebls_fast_grid(t, wx, w, np.array([5.0]), 50, np.array([2], dtype=np.int32), 1)


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "3"])
def test_coarse_params_validated(bad):
    t, y, dy = _small_lc()
    for kw in ({"coarse_factor": bad}, {"n_refine": bad}):
        with pytest.raises(ValueError, match=r"coarse_factor|n_refine"):
            eebls_fast(t, y, dy, _VFREQS, **kw)
        with pytest.raises(ValueError, match=r"coarse_factor|n_refine"):
            multiband_eebls_fast({"g": (t, y, dy)}, _VFREQS, **kw)


def test_helper_validation_rejects_bool_and_strings():
    t = np.linspace(0.0, 10.0, 50)
    with pytest.raises(ValueError):
        build_frequency_grid(t, oversample=True)
    with pytest.raises(ValueError):
        build_frequency_grid(t, q_min="0.1")
    with pytest.raises(ValueError):
        sparse_bls(t, np.sin(t), np.ones(50), np.array([1.0]), q_max="0.1")
    with pytest.raises(ValueError):
        auto_nbins(0.1, n_res=0)
    with pytest.raises(ValueError):
        auto_nbins(0.1, n_res=True)
    with pytest.warns(UserWarning):
        assert auto_nbins(1e-308) == 500   # no OverflowError for a tiny q_min


def test_sparse_sign_selects_and_validates():
    bands, truth, (t, y, dy), _ = _signed_single()
    f = np.array([1.0 / truth.period])
    for fn, arg in ((sparse_bls, (t, y, dy, f)), (multiband_sparse_bls, (bands, f)),
                    (sparse_bls_reference, (t, y, dy, f)),
                    (multiband_sparse_bls_reference, (bands, f))):
        assert fn(*arg, sign="positive").best_power > 5 * fn(*arg, sign="negative").best_power
        with pytest.raises(ValueError, match="sign"):
            fn(*arg, sign="up")
