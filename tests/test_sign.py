"""The ``sign`` option (dips vs brightenings) across entry points."""

from __future__ import annotations

import numpy as np
import pytest

from multiband_bls import (
    eebls,
    eebls_fast,
    multiband_eebls,
    multiband_eebls_fast,
)

from .helpers import _signed_single


@pytest.mark.parametrize("fn,kw", [(eebls, {}), (eebls_fast, {"dlogq": 0.0})])
def test_sign_both_is_max_of_positive_and_negative(fn, kw):
    _, _, (t, y, dy), freqs = _signed_single()
    both = fn(t, y, dy, freqs, **kw).power
    pos = fn(t, y, dy, freqs, sign="positive", **kw).power
    neg = fn(t, y, dy, freqs, sign="negative", **kw).power
    np.testing.assert_allclose(both, np.maximum(pos, neg), rtol=1e-12)
    assert np.all(pos <= both + 1e-15) and np.all(neg <= both + 1e-15)


@pytest.mark.parametrize("flip", [False, True])
def test_sign_selects_dips_or_brightenings(flip):
    # magnitudes dim by +depth: "positive" finds it; flipped (brightening): "negative"
    _, truth, (t, y, dy), _ = _signed_single(flip)
    freqs = np.array([1.0 / truth.period])
    right, wrong = ("negative", "positive") if flip else ("positive", "negative")
    for fn in (eebls, eebls_fast):
        pr = fn(t, y, dy, freqs, sign=right)
        pw = fn(t, y, dy, freqs, sign=wrong)
        assert pr.best_power > 5 * pw.best_power
        assert (pr.best_depth > 0) != flip
        assert pr.best_power == pytest.approx(fn(t, y, dy, freqs).best_power)


def test_sign_fast_matches_exact_signed():
    _, _, (t, y, dy), freqs = _signed_single()
    for sgn in ("positive", "negative"):
        exact = eebls(t - np.median(t), y, dy, freqs, sign=sgn).power
        fast = eebls_fast(t, y, dy, freqs, dlogq=0.0, sign=sgn).power
        np.testing.assert_allclose(fast, exact, rtol=1e-9, atol=1e-12)


def test_sign_multiband():
    bands, truth, _, _ = _signed_single()
    freqs = np.array([1.0 / truth.period])
    flipped = {k: (t, 2 * float(np.mean(y)) - y, dy) for k, (t, y, dy) in bands.items()}
    for fn in (multiband_eebls, multiband_eebls_fast):
        both = fn(bands, freqs).best_power
        pos = fn(bands, freqs, sign="positive").best_power
        neg = fn(bands, freqs, sign="negative").best_power
        assert pos <= both + 1e-12 and pos == pytest.approx(both, rel=1e-2) and neg < 0.2 * pos
        assert fn(flipped, freqs, sign="negative").best_power > 5 * fn(
            flipped, freqs, sign="positive").best_power
    # fast (exact widths) vs exact on a grid, per sign (exact runs on centred times)
    fr = np.linspace(0.3, 0.6, 300)
    off = float(np.median(np.concatenate([t for t, _, _ in bands.values()])))
    centred = {k: (t - off, y, dy) for k, (t, y, dy) in bands.items()}
    for sgn in ("positive", "negative"):
        ex = multiband_eebls(centred, fr, sign=sgn).power
        fa = multiband_eebls_fast(bands, fr, dlogq=0.0, sign=sgn).power
        np.testing.assert_allclose(fa, ex, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("fn", [eebls, eebls_fast])
def test_sign_invalid_raises(fn):
    _, _, (t, y, dy), freqs = _signed_single()
    with pytest.raises(ValueError, match="sign"):
        fn(t, y, dy, freqs, sign="up")
