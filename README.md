# multiband_bls

Fast multiband Box-fitting Least Squares (BLS) for multi-band photometric surveys
(e.g. Vera Rubin Observatory, BlackGEM, ZTF), with both **Cython (CPU)** and **CUDA (GPU)**
backends. Designed for eclipse-like signals that have a significantly different depth
in different bands.

## Why multiband?
The multiband method searches for step-function-like signals in multiple bands
where the duration and phase of a signal is shared between bands, but the depth
of the step-function is different per band. The multiband search shares one 
grid of **period, epoch, and duration** across all bands while fitting an 
**depth per band**, combined with **matched-filter weighting** 
(each band weighted by `W_b = Σ 1/σ²`) — the
generalized likelihood-ratio optimum for a shared transit shape with free per-band
depths.

This is relevant in two cases: the signal is intrinsically different per band
(e.g. eclipsing white dwarf–M-dwarf binaries), or extrinsic, where the flux calibration
of the light curves is not consistently normalised across bands.

Two algorithmic variants are provided:

| Variant | Complexity | When to use |
|---|---|---|
| **Sparse BLS (SBLS)** — unbinned, box edges at observed phases | `O(N²/freq)` | sparse surveys, `N ≲ few thousand` |
| **Binned BLS (eeBLS)** — classic phase-folded bins | `O(N/freq)` | dense surveys, large `N` |

Both report `power = Δχ²/χ²_flat` — the fraction of variance explained by the best
box, in `[0, 1]`.

## Implementations

| Backend | Functions | Requirement |
|---|---|---|
| **Cython (CPU)** | `sparse_bls`, `eebls`, `multiband_sparse_bls`, `multiband_eebls` | C compiler, Cython ≥ 3 |
| **CUDA (GPU)** | `eebls_gpu`, `eebls_gpu_fast`, `multiband_eebls_gpu`, `multiband_eebls_gpu_fast` | CuPy + CUDA GPU |
| Pure Python (reference) | `sparse_bls_reference`, `eebls_reference`, `multiband_sparse_bls_reference`, `multiband_eebls_reference` | NumPy only |

## Install

```bash
pip install -e .            # builds the four Cython extension modules
pip install -e '.[test]'    # + pytest, astropy (for the test suite)
```

If the Cython extensions are not compiled, all four search functions fall back
automatically to the pure-Python reference implementations (with a warning),
so the package is importable even without a C compiler.

GPU support requires CuPy; install separately for your CUDA version:
```bash
pip install cupy-cuda12x    # or cupy-cuda11x
```

## Quick start

### CPU (Cython)

```python
import numpy as np
from multiband_bls import build_frequency_grid, multiband_sparse_bls

# bands maps each label to (t, mag, dy) arrays — supply your own light curves
# bands = {"u": (t_u, mag_u, dy_u), "g": ..., "r": ..., "i": ..., "z": ..., "y": ...}

t_all = np.concatenate([b[0] for b in bands.values()])
freqs = build_frequency_grid(t_all, p_min=0.12, p_max=0.28, q_min=0.01)
res = multiband_sparse_bls(bands, freqs, q_max=0.12)

print(res.best_period, res.best_t0, res.best_duration)
print(dict(zip(res.bands, res.best_depth)))  # per-band depths (mag)
```

### GPU (CUDA)
needs cupy to be installed correctly (https://cupy.dev/)

```python
from multiband_bls import multiband_eebls_gpu, gpu_available

if gpu_available():
    res = multiband_eebls_gpu(bands, freqs, nbins=300)
    # same SBLSResult; ~13× faster than Cython on a laptop RTX 3050 Ti
```

> **Grid resolution matters.** The transit peak in frequency is narrow
> (`~ q_transit / baseline`). Use `df ≲ 0.5 · q_transit / baseline`; the helper
> `build_frequency_grid` computes this automatically.

## Public API

### Multiband

| Function | Backend | Purpose |
|---|---|---|
| `multiband_sparse_bls(bands, freqs, q_max, min_points)` | Cython | unbinned multiband BLS |
| `multiband_eebls(bands, freqs, nbins, qmin, qmax, min_points)` | Cython | binned multiband BLS |
| `multiband_eebls_gpu(bands, freqs, nbins, ...)` | CUDA | binned multiband BLS, GPU |
| `multiband_eebls_gpu_fast(bands, freqs, nbins, ...)` | CUDA | float32 + warp-shuffle variant |
| `multiband_sparse_bls_reference(...)`, `multiband_eebls_reference(...)` | Python | reference oracle |

### Single-band

| Function | Backend | Purpose |
|---|---|---|
| `sparse_bls(t, y, dy, freqs, q_max, min_points)` | Cython | unbinned BLS |
| `eebls(t, y, dy, freqs, nbins, qmin, qmax, min_points)` | Cython | binned BLS |
| `eebls_gpu(t, y, dy, freqs, nbins, ...)` | CUDA | binned BLS, GPU |
| `eebls_gpu_fast(t, y, dy, freqs, nbins, ..., dlogq)` | CUDA | float32 + warp-shuffle variant |
| `sparse_bls_reference(...)`, `eebls_reference(...)` | Python | reference oracle |

### Helpers

| Name | Import | Purpose |
|---|---|---|
| `build_frequency_grid(t, p_min, p_max, q_min, ...)` | `multiband_bls` | build a properly spaced frequency grid |
| `coadd_bands(bands)` | `multiband_bls` | merge a band dict into a single `(t, y, dy)` tuple |
| `gpu_available()` | `multiband_bls` | returns `True` if a CUDA GPU is detected |
| `SBLSResult` | `multiband_bls` | result dataclass (see below) |
| `preprocess(y, dy)` | `multiband_bls.reference` | weight-normalise a light curve |

### SBLSResult fields

All entry points return an `SBLSResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `frequency` | `ndarray` | trial frequencies (input grid) |
| `power` | `ndarray` | Δχ²/χ²_flat at each frequency |
| `period` | `ndarray` | convenience property: `1 / frequency` |
| `best_frequency` | `float` | frequency of the highest peak |
| `best_period` | `float` | period of the highest peak |
| `best_t0` | `float` | mid-transit epoch at the best period |
| `best_duration` | `float` | transit duration at the best period |
| `best_depth` | `float` or `ndarray` | depth (single-band) or per-band depths array (multiband) |
| `best_power` | `float` | peak power value |
| `bands` | `tuple[str, ...]` or `None` | band labels for multiband results |

## GPU backend

`eebls_gpu` and `multiband_eebls_gpu` use CuPy `RawKernel`s — one CUDA thread-block
per trial frequency, with shared-memory phase bins. The periodogram runs entirely on
the GPU; the best period's `t0`/duration/depth are recomputed once on the CPU. Returns
the same `SBLSResult` as the Cython core.

`eebls_gpu_fast` and `multiband_eebls_gpu_fast` trade float64 → float32 and add a
prefix-sum + warp-shuffle reduction for additional throughput at the cost of
~single-precision accuracy. The `dlogq` parameter (default 0.3) controls the
log-spacing of transit-width samples; smaller values give denser coverage at
higher compute cost.

> **Note:** `multiband_eebls_reference` has a fixed `nbins=300` default (no
> auto-selection), unlike the Cython/GPU functions where `nbins=None` triggers
> automatic bin-count selection via `auto_nbins(qmin)`.

`gpu_available()` returns `True` if a usable CUDA GPU is detected.

## Performance

* Cython cores are ~300–400× faster than the pure-Python reference.
* GPU (`multiband_eebls_gpu`) is **~13× faster** than the Cython core on a laptop
  RTX 3050 Ti (4.4 s vs 56 s for a 221k-frequency multiband search); substantially
  more on a datacenter GPU.
* Period recovery agrees with `astropy.timeseries.BoxLeastSquares`; SBLS pins the
  frequency more precisely because it places box edges at observed phases rather
  than on a fixed grid.

## References

* A. Panahi & S. Zucker, *Sparse Box-fitting Least Squares*, PASP **133**, 024502
  (2021), arXiv:2103.06193.
* G. Kovács, S. Zucker & T. Mazeh, *A box-fitting algorithm in the detection of
  periodic transits*, A&A **391**, 369 (2002).
* J. VanderPlas & Ž. Ivezić, *Periodograms for Multiband Astronomical Time Series*,
  ApJ **812**, 18 (2015) — multiband shared-shape idea.
* *cuvarbase* — GPU-accelerated variability tools for astronomy,
  https://github.com/johnh2o2/cuvarbase
