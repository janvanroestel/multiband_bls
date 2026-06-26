# multiband_bls

Fast multiband Box-fitting Least Squares (BLS) for multi-band photometric surveys
such as Vera Rubin Observatory / LSST, with both **Cython (CPU)** and **CUDA (GPU)**
backends. Aimed at eclipsing systems where the transit depth varies strongly with
wavelength — white-dwarf–M-dwarf (WD-MD) binaries being the prime target.

## Why multiband?

A WD-MD eclipse is strongly chromatic: a hot blue WD occulted by a cool red MD is
deep in `u`/`g` and nearly invisible in `z`/`y`. A standard single-band or naive
co-add search discards this structure (or is confused by it). The multiband search
instead shares one grid of **period, epoch, and duration** across all bands while
fitting an **independent baseline and depth per band**, combined with
**matched-filter weighting** (each band weighted by `W_b = Σ 1/σ²`) — the
generalized likelihood-ratio optimum for a shared transit shape with free per-band
depths.

`min_points` is a minimum on the *total* in-transit count across all bands, so a
deep but sparsely sampled band (e.g. `u` with one or two in-eclipse points) still
contributes — the whole point of multiband for these systems.

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
| **Cython (CPU)** | `multiband_sparse_bls`, `multiband_eebls` | C compiler, Cython ≥ 3 |
| **CUDA (GPU)** | `multiband_eebls_gpu`, `multiband_eebls_gpu_fast` | CuPy + CUDA GPU |
| Pure Python (reference) | `multiband_sparse_bls_reference`, `multiband_eebls_reference` | NumPy only |

Single-band equivalents (`sparse_bls`, `eebls`, `eebls_gpu`, …) are also included.

## Install

```bash
pip install -e .            # builds the four Cython extension modules
pip install -e '.[test]'    # + pytest, astropy (for the test suite)
```

GPU support requires CuPy; install separately for your CUDA version:
```bash
pip install cupy-cuda12x    # or cupy-cuda11x
```

## Quick start

### CPU (Cython)

```python
import numpy as np
from multiband_bls import multiband_sparse_bls

# bands maps each label to (t, mag, dy) arrays — supply your own light curves
# bands = {"u": (t_u, mag_u, dy_u), "g": ..., "r": ..., "i": ..., "z": ..., "y": ...}

freqs = np.arange(1/0.28, 1/0.12, 3e-5)
res = multiband_sparse_bls(bands, freqs, q_max=0.12)

print(res.best_period, res.best_t0, res.best_duration)
print(dict(zip(res.bands, res.best_depth)))  # per-band depths (mag)
```

### GPU (CUDA)

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

### Multiband (the main contribution)

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
| `eebls(t, y, dy, freqs, nbins, qmin, qmax, min_bin_points)` | Cython | binned BLS |
| `eebls_gpu(...)`, `eebls_gpu_fast(...)` | CUDA | binned BLS, GPU |

### Helpers

`build_frequency_grid`, `preprocess`, `coadd_bands`, `gpu_available`, `SBLSResult`


## GPU backend

`multiband_eebls_gpu` uses CuPy `RawKernel`s — one CUDA thread-block per trial
frequency, with shared-memory phase bins. The periodogram runs entirely on the GPU;
the best period's `t0`/duration/depth are recomputed once on the CPU. Returns the
same `SBLSResult` as the Cython core.

`multiband_eebls_gpu_fast` trades float64 → float32 and adds a prefix-sum + warp-shuffle
reduction for additional throughput at the cost of ~single-precision accuracy.

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
