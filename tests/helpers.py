"""Shared synthetic light curves and parametrisation tables for the test suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from multiband_bls import (
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


def _centered(bands):
    """Shift all bands' times by the median of the pooled times.

    The fast GPU entry points median-subtract ``t`` internally (float32 range)
    while the exact ones fold raw ``t``; the phase-bin anchor therefore differs
    and binned BLS is sensitive to it. Centering up front makes the fast
    kernels' internal offset zero so exact and fast bin identically.
    """
    off = np.median(np.concatenate([b[0] for b in bands.values()]))
    return {k: (t - off, y, dy) for k, (t, y, dy) in bands.items()}


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
    pytest.param({"q_min": 0.0}, False, id="q_min_zero"),
    pytest.param({"q_min": -0.01}, False, id="q_min_negative"),
    pytest.param({"q_min": np.nan}, False, id="q_min_nan"),
    pytest.param({"q_min": 0.2, "q_max": 0.1}, False, id="q_min_gt_q_max"),
    pytest.param({"nbins": 0}, True, id="nbins_zero"),
    pytest.param({"nbins": 50.9}, True, id="nbins_float"),
    pytest.param({"nbins": np.inf}, True, id="nbins_inf"),
    pytest.param({"nbins": 10**9}, True, id="nbins_huge"),
]


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


def _clustered_u_bands():
    """'g' samples the whole phase with an eclipse at phase 0.5; every 'u' point
    sits inside the eclipse, so u has no out-of-eclipse baseline."""
    rng = np.random.default_rng(8)
    tg = np.sort(rng.uniform(0.0, 50.0, 200))
    yg = 20.0 + 0.5 * (np.abs((tg % 1.0) - 0.5) < 0.03) + rng.normal(0, 0.01, 200)
    tu = 10.0 + 0.5 + np.linspace(-0.01, 0.01, 6)    # all within the eclipse
    yu = 21.0 + rng.normal(0, 0.01, 6)
    return {"g": (tg, yg, np.full(200, 0.01)), "u": (tu, yu, np.full(6, 0.01))}


_TIED_T = np.array([0.0, 0.05, 0.05, 0.05, 0.1, 0.4, 0.6, 0.8])


_TIED_Y = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])


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


def _signed_single(flip: bool = False) -> tuple:
    rng = np.random.default_rng(7)
    bands, truth = _make_lightcurve(2.3, 0.05, rng)
    t, y, dy = bands["r"]
    if flip:  # brightening instead of dimming
        y = 2 * float(np.mean(y)) - y
    freqs = np.linspace(0.2, 1.0, 800)
    return bands, truth, (t, y, dy), freqs
