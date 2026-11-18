"""Tests for the bls() front door, the engine's backends and the compatibility facades."""

from __future__ import annotations

import numpy as np
import pytest

import multiband_bls as mb
from multiband_bls import DensityPrior, SearchConfig, bls, bls_prepare

from .golden_cases import _bands, _freqs, _single, flatten

B, F, S = _bands(), _freqs(), _single()

# (legacy function, bls kwargs, data kind)
_CPU = [
    (mb.sparse_bls, {"method": "sparse"}, "single"),
    (mb.eebls, {"method": "binned"}, "single"),
    (mb.eebls_fast, {"method": "fast"}, "single"),
    (mb.multiband_sparse_bls, {"method": "sparse"}, "multi"),
    (mb.multiband_eebls, {"method": "binned"}, "multi"),
    (mb.multiband_eebls_fast, {"method": "fast"}, "multi"),
]


def _same(a, b):
    fa, fb = flatten(a), flatten(b)
    assert set(fa) == set(fb)
    for k in fa:
        np.testing.assert_array_equal(fa[k], fb[k], err_msg=k)


@pytest.mark.parametrize(("legacy", "kw", "kind"), _CPU, ids=[c[0].__name__ for c in _CPU])
def test_bls_matches_legacy_cpu(legacy, kw, kind):
    data = S if kind == "single" else B
    old = legacy(*S, F) if kind == "single" else legacy(B, F)
    new = bls(data, F, backend="cpu", **kw)
    _same(old, new)
    assert new.backend == "cpu" and new.method == kw["method"]


@pytest.mark.parametrize("sign", ["positive", "negative"])
def test_bls_sign_and_rescale_match_legacy(sign):
    old = mb.multiband_eebls(B, F, rescale_errors=True, sign=sign, min_points=5)
    new = bls(B, F, method="binned", backend="cpu", rescale_errors=True,
              config=SearchConfig(sign=sign, min_points=5))
    _same(old, new)


def test_fast_options_pass_through():
    old = mb.multiband_eebls_fast(B, F, dlogq=0.0, coarse_factor=3, n_refine=4, nbins=90)
    new = bls(B, F, method="fast", backend="cpu", dlogq=0.0, coarse_factor=3, n_refine=4,
              config=SearchConfig(nbins=90))
    _same(old, new)


def test_result_records_provenance():
    res = bls(B, F, method="binned", backend="cpu", config=SearchConfig(q_min=0.02, q_max=0.1))
    assert (res.method, res.backend) == ("binned", "cpu")
    assert res.segments is not None and len(res.segments) == 1
    seg = res.segments[0]
    assert (seg.q_min, seg.q_max, seg.n_freqs) == (0.02, 0.1, F.size)
    assert seg.f_min == F.min() and seg.f_max == F.max() and seg.nbins == 250
    assert res.has_solution
    sparse = bls(S, F, method="sparse", backend="cpu")
    assert sparse.segments[0].nbins is None and sparse.segments[0].q_max == 0.15


def test_has_solution_false_without_detection():
    t, _, dy = S
    res = bls((t, np.ones_like(t), dy), F, method="binned", backend="cpu")
    assert not res.has_solution and res.best_power == 0.0


def test_period_range_builds_the_grid():
    t = S[0]
    res = bls(S, p_min=0.3, p_max=0.5, method="fast", backend="cpu")
    ref = mb.build_frequency_grid(t, p_min=0.3, p_max=0.5, q_min=0.01)
    np.testing.assert_array_equal(res.frequency, ref)
    cfg = SearchConfig(density_prior=DensityPrior((0.2, 5.0)))
    dres = bls(S, p_min=0.3, p_max=0.5, method="fast", backend="cpu", config=cfg)
    np.testing.assert_array_equal(
        dres.frequency, cfg.density_prior.grid(t, 0.3, 0.5))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "nope"}, "method"),
        ({"backend": "tpu"}, "backend"),
        ({"method": "sparse", "backend": "gpu"}, "no GPU"),
        ({"method": "sparse", "config": SearchConfig(nbins=50)}, "nbins"),
        ({"rescale_errors": True}, "multiband"),
        ({"p_min": 0.3}, "frequencies"),
        ({"frequencies": F, "p_min": 0.3, "p_max": 0.5}, "not both"),
        ({"config": SearchConfig(sign="sideways")}, "sign"),
    ],
)
def test_bls_rejects_bad_arguments(kwargs, match):
    kwargs.setdefault("frequencies", None if "p_min" in kwargs else F)
    with pytest.raises(ValueError, match=match):
        bls(S, backend=kwargs.pop("backend", "cpu"), **kwargs)


def test_bls_rejects_bad_data_shape():
    with pytest.raises(ValueError, match="data must be"):
        bls(S[:2], F)
    with pytest.raises(ValueError, match="frequencies"):
        bls(S)  # neither frequencies nor p_min/p_max


def test_auto_backend_falls_back_on_capability_limit():
    # nbins far beyond any GPU's shared memory: auto must still search, on the CPU
    res = bls(B, F[:20], method="binned", backend="auto", config=SearchConfig(nbins=9000))
    assert res.backend == "cpu" and res.method == "binned"


def test_auto_backend_is_cpu_for_sparse():
    assert bls(S, F, method="sparse").backend == "cpu"


@pytest.mark.gpu
@pytest.mark.parametrize(("legacy", "kw", "kind"), [
    (mb.eebls_gpu, {"method": "binned"}, "single"),
    (mb.eebls_gpu_fast, {"method": "fast"}, "single"),
    (mb.multiband_eebls_gpu, {"method": "binned"}, "multi"),
    (mb.multiband_eebls_gpu_fast, {"method": "fast"}, "multi"),
])
def test_bls_matches_legacy_gpu(legacy, kw, kind):
    if not mb.gpu_available():
        pytest.skip("no GPU")
    old = legacy(*S, F) if kind == "single" else legacy(B, F)
    new = bls(S if kind == "single" else B, F, backend="gpu", **kw)
    np.testing.assert_allclose(new.power, old.power, rtol=1e-4, atol=1e-6)
    assert new.best_frequency == old.best_frequency and new.backend == "gpu"
    auto = bls(S if kind == "single" else B, F, backend="auto", **kw)
    assert auto.backend == "gpu"


# --------------------------------------------------------------------------- #
# prepared data, mutation, hoisting                                            #
# --------------------------------------------------------------------------- #
def _copy_bands():
    return {k: tuple(a.copy() for a in v) for k, v in B.items()}


def test_each_call_sees_current_inputs_after_in_place_mutation():
    bands = _copy_bands()
    r1 = bls(bands, F, method="binned", backend="cpu")
    bands["g"][1][:] = bands["g"][1] + np.linspace(0.0, 1.0, bands["g"][1].size)  # edit y
    r2 = bls(bands, F, method="binned", backend="cpu")
    fresh = bls(_copy_bands() | {"g": bands["g"]}, F, method="binned", backend="cpu")
    assert not np.array_equal(r1.power, r2.power)
    np.testing.assert_array_equal(r2.power, fresh.power)
    bands["r"] = bands["r"][0][:50], bands["r"][1][:50], bands["r"][2][:50]  # edit mapping
    r3 = bls(bands, F, method="binned", backend="cpu")
    assert not np.array_equal(r2.power, r3.power)


def test_prepared_snapshot_is_immune_to_later_mutation():
    bands = _copy_bands()
    snap = bls_prepare(bands, method="fast")
    before = bls(snap, F)
    bands["g"][1][:] = 0.0
    bands["g"][2][:] = 1.0
    after = bls(snap, F)
    _same(before, after)
    _same(before, bls(B, F, method="fast", backend="cpu"))
    assert after.method == "fast" and after.backend == "cpu"


def test_prepared_snapshot_conflicts_and_period_range():
    snap = bls_prepare(S, method="binned")
    with pytest.raises(ValueError, match="conflicts"):
        bls(snap, F, method="fast")
    with pytest.raises(ValueError, match="rescale_errors"):
        bls(snap, F, rescale_errors=True)
    res = bls(snap, p_min=0.3, p_max=0.5)
    assert res.frequency.size == mb.build_frequency_grid(S[0], p_min=0.3, p_max=0.5).size
    with pytest.raises(ValueError, match="method"):
        bls_prepare(S, method="bad")
    with pytest.raises(ValueError, match="backend"):
        bls_prepare(S, backend="bad")
    with pytest.raises(ValueError, match="multiband"):
        bls_prepare(S, rescale_errors=True)
    snap_r = bls_prepare(B, rescale_errors=True)
    _same(bls(snap_r, F), bls(B, F, rescale_errors=True, backend="cpu"))


def test_multi_segment_search_preprocesses_and_refits_once(monkeypatch):
    from multiband_bls.backends import cpu

    calls = {"prepare": 0, "merge": 0, "refine": 0}
    for name, key in (("_prepare_bands", "prepare"), ("_merge_bands", "merge")):
        orig = getattr(cpu, name)

        def wrapped(*a, _orig=orig, _key=key, **k):
            calls[_key] += 1
            return _orig(*a, **k)

        monkeypatch.setattr(cpu, name, wrapped)
    orig_refine = cpu.FastMulti.refine

    def refine(self, *a, **k):
        calls["refine"] += 1
        return orig_refine(self, *a, **k)

    monkeypatch.setattr(cpu.FastMulti, "refine", refine)
    cfg = SearchConfig(density_prior=DensityPrior((0.2, 5.0)))
    f = np.geomspace(0.5, 20.0, 600)
    res = bls(B, f, method="fast", backend="cpu", config=cfg)
    assert len(res.segments) > 5
    assert calls == {"prepare": 1, "merge": 1, "refine": 1}


# --------------------------------------------------------------------------- #
# compatibility facades                                                        #
# --------------------------------------------------------------------------- #
def test_legacy_import_paths_and_identity():
    import multiband_bls.api as api
    import multiband_bls.gpu as gpu
    import multiband_bls.periodogram as periodogram
    import multiband_bls.reference as reference
    import multiband_bls.result as result

    for name in ("sparse_bls", "eebls", "multiband_sparse_bls", "multiband_eebls",
                 "eebls_fast", "multiband_eebls_fast"):
        assert getattr(api, name) is getattr(mb, name)
    for name in ("eebls_gpu", "eebls_gpu_fast", "multiband_eebls_gpu",
                 "multiband_eebls_gpu_fast", "gpu_available"):
        assert getattr(gpu, name) is getattr(mb, name)
    assert periodogram.BLSResult is result.BLSResult is mb.BLSResult
    for name in ("preprocess", "auto_nbins", "variance_explained", "build_frequency_grid",
                 "coadd_bands", "width_bounds", "_prepare_bands", "_merge_band_arrays",
                 "_validate_search_params", "_check_sign", "_check_phase_range",
                 "sparse_bls_reference", "eebls_reference"):
        assert callable(getattr(reference, name))
    assert reference.build_frequency_grid is mb.build_frequency_grid


def test_registry_lists_all_ten_implementations():
    from multiband_bls.backends import available_implementations, get_implementation

    keys = available_implementations()
    cpu = [k for k in keys if k[1] == "cpu"]
    assert len(cpu) == 6
    assert len(keys) in (6, 10)
    assert ("binned", "gpu", True) in keys
    with pytest.raises(ValueError, match="no implementation"):
        get_implementation("sparse", "gpu", False)


# --------------------------------------------------------------------------- #
# inspection regressions                                                       #
# --------------------------------------------------------------------------- #
def test_snapshot_exposes_nothing_mutable():
    snap = bls_prepare(S, method="binned")
    base = bls(snap, F)
    with pytest.raises(ValueError, match="read-only"):
        snap.times[:] += 0.1
    # a result's diagnostics must not alias the snapshot's state
    msnap = bls_prepare(B, method="binned")
    r1 = bls(msnap, F)
    r1.band_chi2_red[:] = -1.0
    r2 = bls(msnap, F)
    assert np.all(r2.band_chi2_red != -1.0)
    _same(base, bls(snap, F))
    # the snapshot's own times do not alias what the search used
    assert not np.shares_memory(snap.times, snap._state.t_raw)


def test_missing_compiled_cores_fall_back_to_reference():
    import subprocess
    import sys

    code = (
        "import sys, warnings\n"
        "for m in ('_sbls', '_msbls', '_eebls', '_meebls', '_fast'):\n"
        "    sys.modules['multiband_bls.' + m] = None\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import multiband_bls as m\n"
        "assert m.sparse_bls is m.sparse_bls_reference and m.eebls is m.eebls_reference\n"
        "assert any('Cython' in str(x.message) for x in w)\n"
        "missing = [n for n in m.__all__ if not hasattr(m, n)]\n"
        "assert not missing, missing\n"
        "ns = {}\n"
        "exec('from multiband_bls import *', ns)\n"
        "for fn in (m.bls, m.eebls_fast, m.eebls_gpu):\n"
        "    try:\n"
        "        fn(None)\n"
        "    except ImportError:\n"
        "        pass\n"
        "    else:\n"
        "        raise SystemExit('no ImportError from ' + fn.__name__)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_evaluated_mask_marks_interpolated_frequencies():
    f = np.linspace(_freqs()[0], _freqs()[-1], 400)
    full = bls(B, f, method="fast", backend="cpu")
    assert full.evaluated is None  # every frequency evaluated
    coarse = bls(B, f, method="fast", backend="cpu", coarse_factor=4, n_refine=3)
    assert coarse.evaluated is not None and coarse.evaluated.shape == f.shape
    assert 0 < coarse.evaluated.sum() < f.size
    assert coarse.evaluated[int(np.argmax(coarse.power))]  # the peak is a real evaluation
    assert coarse.options.coarse_factor == 4 and coarse.options.n_refine == 3
    # duplicates / descending order map the mask back to the caller's order
    shuffled = np.concatenate([f[::-1], f[:50]])
    rev = bls(B, shuffled, method="fast", backend="cpu", coarse_factor=4, n_refine=3)
    assert rev.evaluated.shape == shuffled.shape
    # a segmented (density) search assembles the mask in caller order too
    cfg = SearchConfig(density_prior=DensityPrior((0.2, 5.0)))
    seg = bls(B, np.geomspace(0.5, 20.0, 800), method="fast", backend="cpu", config=cfg,
              coarse_factor=3)
    assert seg.evaluated is not None and seg.evaluated.shape == (800,)


def test_options_and_fallback_are_recorded():
    res = bls(B, F, method="fast", backend="cpu", rescale_errors=True,
              config=SearchConfig(sign="negative", min_points=7), dlogq=0.2)
    o = res.options
    assert (o.sign, o.min_points, o.dlogq, o.rescale_errors) == ("negative", 7, 0.2, True)
    assert o.requested_backend == "cpu" and o.fallback is None
    sparse = bls(S, F, method="sparse", backend="cpu").options
    assert sparse.dlogq is None and sparse.coarse_factor == 1
    auto = bls(B, F[:20], method="binned", backend="auto", config=SearchConfig(nbins=9000))
    assert auto.backend == "cpu" and auto.options.requested_backend == "auto"
    if mb.gpu_available():  # the CPU result must say why it is not on the GPU
        assert auto.options.fallback is not None and "GPU limit" in auto.options.fallback
    else:
        assert auto.options.fallback is None


def test_q_bounds_and_widening_metadata():
    cfg = SearchConfig(density_prior=DensityPrior((1.0, 1.0), (1.0, 1.0)))
    res = bls(S, np.array([1.0, 2.0, 5.0]), method="binned", backend="cpu", config=cfg)
    assert any(seg.widened for seg in res.segments)
    lo, hi = res.q_bounds(res.frequency)
    assert np.all(np.isfinite(lo)) and np.all(hi >= lo)
    assert np.isnan(res.q_bounds(1e-6)[0]).all()  # outside every segment
    fixed = bls(S, F, method="binned", backend="cpu", config=SearchConfig(q_min=0.02))
    assert not fixed.segments[0].widened
    np.testing.assert_array_equal(fixed.q_bounds(F)[0], 0.02)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_bls_validates_n_threads_for_fast(bad):
    with pytest.raises(ValueError, match="n_threads"):
        bls(S, F, method="fast", backend="cpu", n_threads=bad)
    # the same validation applies whatever backend "auto" ends up choosing
    with pytest.raises(ValueError, match="n_threads"):
        bls(S, F, method="fast", backend="auto", n_threads=bad)
    bls(S, F, method="binned", backend="cpu", n_threads=bad)  # unused by other methods


def test_legacy_wrappers_record_their_options():
    res = mb.multiband_eebls(B, F, rescale_errors=True, sign="negative", min_points=5)
    assert res.options.rescale_errors and res.options.sign == "negative"
    assert res.options.min_points == 5 and res.options.requested_backend == "cpu"
    assert not mb.multiband_eebls(B, F).options.rescale_errors
    fast = mb.multiband_eebls_fast(B, F, coarse_factor=3, dlogq=0.2, rescale_errors=True)
    assert fast.options.coarse_factor == 3 and fast.options.dlogq == 0.2
    assert fast.options.rescale_errors
    if mb.gpu_available():
        g = mb.multiband_eebls_gpu_fast(B, F, rescale_errors=True)
        assert g.options.rescale_errors and g.options.requested_backend == "gpu"


@pytest.mark.gpu
def test_gpu_multi_segment_search_preprocesses_once(monkeypatch):
    if not mb.gpu_available():
        pytest.skip("no GPU")
    from multiband_bls.backends import gpu as bgpu

    counts = {"merge": 0, "single": 0, "prepare": 0}
    for name, key in (("_merge_bands", "merge"), ("_preprocess_single", "single"),
                      ("_prepare_bands", "prepare")):
        orig = getattr(bgpu, name)

        def wrapped(*a, _orig=orig, _key=key, **k):
            counts[_key] += 1
            return _orig(*a, **k)

        monkeypatch.setattr(bgpu, name, wrapped)
    cfg = SearchConfig(density_prior=DensityPrior((0.2, 5.0)))
    f = np.geomspace(0.5, 20.0, 600)
    res = bls(B, f, method="fast", backend="gpu", config=cfg)
    assert len(res.segments) > 5
    assert counts == {"merge": 1, "single": 0, "prepare": 1}
    counts.update(merge=0, single=0, prepare=0)
    bls(S, f, method="fast", backend="gpu", config=cfg)
    assert counts["single"] == 1 and counts["merge"] == 0


# --------------------------------------------------------------------------- #
# third review: interpolated winners, unrepresentable widths, sparse signatures #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", ["descending", "shuffled", "ascending"])
def test_coarse_search_never_selects_an_interpolated_frequency(order):
    cfg = SearchConfig(nbins=20, q_min=0.05, q_max=0.2, min_points=1)
    checked = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0.0, 1.0, 6))
        y, dy = rng.normal(size=6), np.ones(6)
        f = np.linspace(0.1, 2.0, 100)
        if order == "descending":
            f = f[::-1].copy()
        elif order == "shuffled":
            f = rng.permutation(f)
        res = bls((t, y, dy), f, method="fast", backend="cpu", config=cfg,
                  coarse_factor=4, n_refine=1)
        if res.evaluated is None:
            continue
        i = int(np.flatnonzero(res.frequency == res.best_frequency)[0])
        if res.best_power > 0:
            checked += 1
            assert res.evaluated[i], f"seed {seed}: interpolated frequency chosen"
            assert res.best_power == res.power[i]
            exact = bls((t, y, dy), np.array([res.best_frequency]), method="fast",
                        backend="cpu", config=cfg)
            assert exact.power[0] == pytest.approx(res.best_power, rel=1e-9)
        else:  # nothing passed anywhere: the caller's first frequency
            assert res.best_frequency == f[0]
    assert checked > 20


@pytest.mark.parametrize("method", ["binned", "fast"])
def test_unrepresentable_width_range_is_rejected_everywhere(method):
    cfg = SearchConfig(nbins=10, q_min=0.011, q_max=0.019)  # no whole bin fits
    with pytest.raises(ValueError, match="no window width"):
        bls(S, F, method=method, backend="cpu", config=cfg)
    with pytest.raises(ValueError, match="no window width"):
        bls(B, F, method=method, backend="cpu", config=cfg)
    if mb.gpu_available():
        with pytest.raises(ValueError, match="no window width"):
            bls(B, F, method=method, backend="gpu", config=cfg)
    kw = {"nbins": 10, "q_min": 0.011, "q_max": 0.019}
    legacy = {"binned": (mb.eebls, mb.multiband_eebls),
              "fast": (mb.eebls_fast, mb.multiband_eebls_fast)}[method]
    with pytest.raises(ValueError, match="no window width"):
        legacy[0](*S, F, **kw)
    with pytest.raises(ValueError, match="no window width"):
        legacy[1](B, F, **kw)
    if method == "binned":
        with pytest.raises(ValueError, match="no window width"):
            mb.eebls_reference(*S, F, **kw)
        with pytest.raises(ValueError, match="no window width"):
            mb.multiband_eebls_reference(B, F, **kw)


def test_legacy_sparse_positional_order_is_unchanged():
    import inspect

    for fn in (mb.sparse_bls, mb.multiband_sparse_bls, mb.sparse_bls_reference,
               mb.multiband_sparse_bls_reference):
        params = inspect.signature(fn).parameters
        assert params["q_min"].kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        positional = [n for n, p in params.items()
                      if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
        # data, frequencies, then the arguments that existed before q_min was added
        assert positional[positional.index("frequencies") + 1:][:2] == ["q_max", "min_points"]
    # the fifth positional argument is q_max, as it always was
    pos = mb.sparse_bls(*S, F, 0.1)
    assert pos.segments[0].q_max == 0.1 and pos.segments[0].q_min == 0.01
    np.testing.assert_array_equal(pos.power, mb.sparse_bls(*S, F, q_max=0.1).power)
    mpos = mb.multiband_sparse_bls(B, F, 0.1, 4)
    assert (mpos.segments[0].q_max, mpos.options.min_points) == (0.1, 4)
    with pytest.raises(TypeError):
        mb.sparse_bls(*S, F, 0.1, 3, "both", 0.02)  # q_min cannot be positional
