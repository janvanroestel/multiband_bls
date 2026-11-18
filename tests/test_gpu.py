"""GPU (CuPy) entry points: parity with the CPU, limits and signed searches."""

from __future__ import annotations

import numpy as np
import pytest

import multiband_bls.gpu as gpu_mod
from multiband_bls import (
    coadd_bands,
    eebls,
    eebls_fast,
    eebls_gpu,
    eebls_gpu_fast,
    gpu_available,
    multiband_eebls,
    multiband_eebls_fast,
    multiband_eebls_gpu,
    multiband_eebls_gpu_fast,
    multiband_eebls_reference,
)
from multiband_bls.reference import preprocess

from .helpers import (
    _GPU,
    _VFREQS,
    _centered,
    _clustered_u_bands,
    _misscaled_bands,
    _old_min_r_disagrees,
    _signed_single,
    _small_lc,
)


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


@_GPU
def test_gpu_fast_coarse_to_fine_recovers_full_peak(lightcurve, oversampled_freqs):
    freqs = oversampled_freqs
    bands, _ = lightcurve
    t, y, dy = coadd_bands(bands)
    full = eebls_gpu_fast(t, y, dy, freqs, nbins=200, q_max=0.12)
    coarse = eebls_gpu_fast(t, y, dy, freqs, nbins=200, q_max=0.12,
                            coarse_factor=3, n_refine=5)
    assert coarse.power.shape == full.power.shape
    assert coarse.best_frequency == full.best_frequency
    assert coarse.best_power == pytest.approx(full.best_power)

    mfull = multiband_eebls_gpu_fast(bands, freqs, nbins=200, q_max=0.12)
    mcoarse = multiband_eebls_gpu_fast(bands, freqs, nbins=200, q_max=0.12,
                                       coarse_factor=3, n_refine=5)
    assert mcoarse.best_frequency == mfull.best_frequency
    assert mcoarse.best_power == pytest.approx(mfull.best_power)
    assert mfull.n_bands_used is not None and mcoarse.n_bands_used is None


@_GPU
@pytest.mark.parametrize("nbins", [60, 200, 300])
def test_gpu_fast_multiband_matches_cpu_across_sweep_paths(lightcurve, freqs, nbins):
    """The multiband GPU sweep distributes (start, width) pairs over threads for small
    nbins and loops widths per start for large nbins; both agree with the CPU search."""
    bands, _ = lightcurve
    cpu = multiband_eebls_fast(bands, freqs, nbins=nbins, q_min=0.02, q_max=0.12)
    gpu = multiband_eebls_gpu_fast(bands, freqs, nbins=nbins, q_min=0.02, q_max=0.12)
    np.testing.assert_allclose(cpu.power, gpu.power, rtol=5e-3, atol=5e-5)
    assert gpu.n_bands_used is not None
    assert gpu.n_bands_used.shape == freqs.shape
    assert gpu.n_bands_used.min() >= 0 and gpu.n_bands_used.max() <= len(bands)


@_GPU
@pytest.mark.parametrize("scale", [1e-25, 1e-20, 1e19, 1e25])
def test_gpu_fast_is_flux_scale_invariant(lightcurve, freqs, scale):
    """float32 kernels run in median-uncertainty units, so tiny/huge flux scales
    (CGS fluxes, ADU) neither underflow to 0 nor overflow to inf."""
    bands, _ = lightcurve
    scaled = {k: (t, y * scale, dy * scale) for k, (t, y, dy) in bands.items()}
    t, y, dy = coadd_bands(bands)
    ts, ys, dys = coadd_bands(scaled)
    ref = eebls_gpu_fast(t, y, dy, freqs, nbins=200, q_max=0.12)
    got = eebls_gpu_fast(ts, ys, dys, freqs, nbins=200, q_max=0.12)
    np.testing.assert_allclose(got.power, ref.power, rtol=1e-3, atol=1e-6)
    mref = multiband_eebls_gpu_fast(bands, freqs, nbins=200, q_max=0.12)
    mgot = multiband_eebls_gpu_fast(scaled, freqs, nbins=200, q_max=0.12)
    assert np.all(np.isfinite(mgot.power))
    np.testing.assert_allclose(mgot.power, mref.power, rtol=1e-3, atol=1e-6)


@_GPU
@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_gpu_coarse_params_validated(bad):
    t, y, dy = _small_lc()
    for kw in ({"coarse_factor": bad}, {"n_refine": bad}):
        with pytest.raises(ValueError, match=r"coarse_factor|n_refine"):
            eebls_gpu_fast(t, y, dy, _VFREQS, **kw)
        with pytest.raises(ValueError, match=r"coarse_factor|n_refine"):
            multiband_eebls_gpu_fast({"g": (t, y, dy)}, _VFREQS, **kw)


@_GPU
def test_gpu_n_bands_used_kept_when_coarse_search_falls_back(lightcurve, oversampled_freqs):
    """A grid shorter than 4 * coarse_factor is evaluated in full, so n_bands_used is
    still reported; on a long grid the coarse search drops it."""
    bands, _ = lightcurve
    short = oversampled_freqs[:11]
    kw = {"nbins": 200, "q_max": 0.12, "coarse_factor": 3}
    res = multiband_eebls_gpu_fast(bands, short, **kw)
    full = multiband_eebls_gpu_fast(bands, short, nbins=200, q_max=0.12)
    assert res.n_bands_used is not None
    np.testing.assert_array_equal(res.n_bands_used, full.n_bands_used)
    np.testing.assert_array_equal(res.power, full.power)
    assert multiband_eebls_gpu_fast(bands, oversampled_freqs, **kw).n_bands_used is None


@pytest.mark.skipif(not gpu_available(), reason="no GPU")
@pytest.mark.parametrize("sgn", ["positive", "negative"])
def test_sign_gpu_matches_cpu(sgn):
    bands, _, (t, y, dy), _ = _signed_single()
    f = np.linspace(0.3, 0.6, 300)
    np.testing.assert_allclose(eebls_gpu(t, y, dy, f, sign=sgn).power,
                               eebls(t, y, dy, f, sign=sgn).power, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(multiband_eebls_gpu(bands, f, sign=sgn).power,
                               multiband_eebls(bands, f, sign=sgn).power, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(eebls_gpu_fast(t, y, dy, f, sign=sgn).power,
                               eebls_fast(t, y, dy, f, sign=sgn).power, rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(multiband_eebls_gpu_fast(bands, f, sign=sgn).power,
                               multiband_eebls_fast(bands, f, sign=sgn).power,
                               rtol=2e-3, atol=1e-4)


@pytest.mark.skipif(not gpu_available(), reason="no GPU")
def test_sign_gpu_selects_and_validates():
    bands, truth, (t, y, dy), _ = _signed_single()
    f = np.array([1.0 / truth.period])
    for fn, arg in ((eebls_gpu, (t, y, dy, f)), (eebls_gpu_fast, (t, y, dy, f)),
                    (multiband_eebls_gpu, (bands, f)), (multiband_eebls_gpu_fast, (bands, f))):
        assert fn(*arg, sign="positive").best_power > 5 * fn(*arg, sign="negative").best_power
        with pytest.raises(ValueError, match="sign"):
            fn(*arg, sign="up")
