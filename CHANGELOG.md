# Changelog

## [Unreleased]

### Added

- `bls()`: one entry point for every search. `bls(data, frequencies | p_min/p_max,
  method="binned" | "fast" | "sparse", backend="cpu" | "gpu" | "auto", config=...)`
  takes `(t, y, dy)` or a mapping of bands and dispatches to the same implementations as
  the existing functions (output identical, enforced by golden-output tests).
  `backend="auto"` falls back to the CPU, keeping the method, only when the GPU reports a
  capability limit. `bls_prepare()` returns a read-only snapshot of the validated,
  weighted (and uploaded) data for repeated searches.
- `SearchConfig` (durations, `nbins`, `min_points`, `sign`; unset `q_min`/`q_max` resolve
  per method) and `DensityPrior`: a stellar-density-based duration range. A transit lasts
  `q = T/P ∝ P^(-2/3)` for a given density, so a fixed `(q_min, q_max)` tests widths no
  real transit can have. With `SearchConfig(density_prior=DensityPrior((rho_min, rho_max)))`
  the frequency grid is cut into segments of slowly varying duty cycle, each searched with
  its own `q_min`, `q_max` and `nbins`, and `bls(p_min=, p_max=)` builds a grid whose step
  follows the minimum duration (uniform in `f^(1/3)`): 9-70x fewer points than a uniform
  grid resolving the same narrowest box for 3-year baselines and period ranges of
  0.1-10 d to 30 s-1 d (`density_duration_bounds` and `density_frequency_grid` expose the
  pieces). `power` is maximised over a different width range in each segment, so its noise
  floor drifts slowly with frequency.
- `BLSResult.method`, `.backend`, `.segments` (the widths and bins actually searched) and
  `.has_solution` (`best_power > 0`).
- A segment search preprocesses and merges the data once and does one exact best-fit refit
  at the global peak; previously each refinement re-validated and re-preprocessed.

### Changed (internal; public names and results unchanged)

- The ten search functions are thin wrappers over a backend contract (`engine.py`,
  `backends/`): `prepare` once, `run` per frequency segment, `refine` once. `reference.py`
  now holds only the pure-Python oracles; validation, weights, band merging and grids moved
  to `_validate`, `_weights`, `_prep`, `grids`, and `BLSResult` to `result` (the old import
  paths still work). The CUDA sources moved from Python strings to `kernels/*.cu` (package
  data, byte-identical). The test suite is split by topic.
- `eebls_fast` and `multiband_eebls_fast`: CPU counterparts of the `*_gpu_fast`
  entry points (prefix-sum phase bins, log-spaced widths via `dlogq`) in a new
  Cython core `_fast.pyx`, in double precision. The frequency loop runs without the
  GIL, and `n_threads` splits the grid over threads. With `dlogq=0` the periodogram
  equals `eebls` / `multiband_eebls` on median-centred times.
- Opt-in coarse-to-fine frequency search for `eebls_fast`, `multiband_eebls_fast`,
  `eebls_gpu_fast` and `multiband_eebls_gpu_fast` (`coarse_factor`, `n_refine`;
  default `coarse_factor=1` = full grid). Every `coarse_factor`-th frequency is
  searched, then the full grid around the `n_refine` highest separated coarse peaks,
  cutting the work by roughly `coarse_factor`. Only meaningful on a grid oversampled
  by at least `coarse_factor` (3 by default from `build_frequency_grid`); it can miss
  weak or narrow peaks, so validate recovery against the full grid first. In 60 random
  injections (4 bands x 60 points) `coarse_factor` 2-4 recovered all 26 signals the
  full grid recovered, at 2.0-3.8x the speed. `n_bands_used` is `None` for
  `multiband_eebls_gpu_fast` when the coarse search skips frequencies (grids shorter
  than `4 * coarse_factor` are evaluated in full and keep it).
- `sparse_bls` and `multiband_sparse_bls` (and their reference versions) take `q_min`
  (default 0.01) and `sign`, matching the binned entry points. The binned reference
  searches (`eebls_reference`, `multiband_eebls_reference`) also take `sign`, so the
  pure-Python fallback accepts every documented argument.

### Changed

- Sparse searches now exclude windows shorter than `q_min` (default 0.01 of the
  period); previously there was no minimum duration. Pass a smaller `q_min` for the
  old behaviour.
- Binned searches enforce `q_min <= duration <= q_max` exactly: the window widths are
  now `ceil(q_min * nbins)` to `floor(q_max * nbins)` (previously `floor(q_min * nbins)`
  to `floor(q_max * nbins) + 1`, which could exceed `q_max`). A `ValueError` is raised
  if no integer width fits. Results can differ slightly from earlier versions.
- `multiband_sparse_bls` updates the combined statistic incrementally (O(1) per window
  extension instead of O(bands)): about 1.6x faster with 6 bands, 2.7x with 12.

- GPU multiband kernels are compiled per band count (`-DNB`), which removes local-memory
  spills (256 B/thread exact, 32 B/thread fast) and makes the fast kernel about 10-18%
  faster. The fast kernel's launch-bounds hint now follows the shared memory per block
  instead of a fixed 4 blocks per SM (about 11% faster kernel time at 6-8 bands, 500
  bins). Results are unchanged up to atomic-ordering noise.

### Fixed

- Coarse-to-fine search no longer recurses forever when the frequency grid contains
  duplicates.
- Fast entry points (`eebls_fast`, `multiband_eebls_fast`, `eebls_gpu_fast`,
  `multiband_eebls_gpu_fast`): `best_t0`, `best_duration` and `best_depth` are now
  refined on the same median-centred times the search folded, so the refined window
  passes the same `min_points` gate as the scored one (`best_t0` is still reported
  on the original time axis).
- GPU fast kernels now work in units of the median uncertainty: flux scales such as
  1e-20 (CGS) or 1e25 previously underflowed or overflowed the float32 kernels and
  returned power 0 or inf. The power is scale-invariant, so results are unchanged
  otherwise.
- GPU fast kernels reject light curves with `n >= 2**23` points (was `2**24`): above
  that the float32 `count - 0.5` gates round to the count itself.
- Coarse-to-fine search on a descending or unsorted frequency grid.
- Build flags: `-fno-finite-math-only` is added to `-O3 -ffast-math` so the NaN that the
  multiband cores store for unmeasured bands is well defined; MSVC builds use `/O2`.
- `multiband_eebls_fast` converts the band index and counts the points per band once
  instead of in every thread chunk.
- `build_frequency_grid` and the `q_*` fraction checks reject `bool` and string
  arguments; `auto_nbins` validates `n_res` and no longer raises `OverflowError` for a
  tiny `q_min`.

### Changed

- CPU fast cores (`_fast.pyx`): the per-frequency fold/bin/prefix-sum and window sweep
  are now plain C in `_fast_kernels.h`. The sweep lists the start bins passing the
  `min_points` gate first and evaluates the statistic only on those; there is no
  wrap-around branch in the window sums; the phase fold is a separate vectorised
  pass (int32 truncation, guarded at `max|t| * max(f) < 2**30`); and the window
  evaluators are AVX2-multiversioned on x86-64 Linux/GCC. Powers agree with the
  previous implementation to ~1e-15. Interleaved single-thread A/B: 0.9-1.4x
  (best for many sparse bands, 1.1-1.25x for dense light curves).
- `multiband_eebls_gpu_fast`: for `nbins < 224` the sweep distributes the
  (start, width) pairs over all threads (1.05-1.7x faster for small `nbins`, e.g.
  1.24x at `nbins=100`; unchanged above). Powers and `n_bands_used` are unchanged.

- **Python 3.9 is no longer supported** (end-of-life since October 2025, and no
  longer installable on GitHub's Ubuntu 26.04 runners). `requires-python` is now
  `>=3.10`; CI, the mypy target and the release wheels start at 3.10.
- The width/grid helpers (`_log_widths`, `_grid_params`, `_check_dlogq`) and the
  best-period refinement (`_result`) moved from `gpu.py` to `api.py`, where both the
  GPU and the new CPU fast entry points use them. `gpu.py` re-imports them, so
  `multiband_bls.gpu._log_widths` etc. still work.

### Removed

- The unused trapezoid-shape code path in `_eebls.pyx` (`f_arr`/`tau`); `eebls_grid`
  no longer takes `f_arr` or returns `best_tau`. Not part of the public API.

## [0.1.0] — 2026-10-17

Initial release.

- Single-band and multiband Sparse BLS (`sparse_bls`, `multiband_sparse_bls`)
- Single-band and multiband binned BLS (`eebls`, `multiband_eebls`)
- GPU-accelerated binned BLS via CuPy (`eebls_gpu`, `multiband_eebls_gpu`)
- Fast GPU variants using prefix-sum phase bins, log-spaced transit widths, and a float-float phase fold (`eebls_gpu_fast`, `multiband_eebls_gpu_fast`)
- Frequency grid builder (`build_frequency_grid`) with period and frequency interfaces
- Pure-Python reference implementations for testing and verification

### Robustness and diagnostics (pre-release review, 2026-10-14)

- **Input validation** in every entry point: non-finite values, `dy <= 0`,
  over/underflowing `1/dy²`, unequal lengths, empty arrays/bands, non-positive
  frequencies, `q_min`/`q_max` outside `0 < q_min ≤ q_max < 1`, and non-integer
  or `< 1` `min_points`/`nbins` now raise `ValueError` (naming the band) instead
  of silently returning an all-NaN periodogram or truncating. Also rejected:
  `y` large enough to overflow the mean-subtracted signal or its χ²,
  `max|t|·max(f) > 1e15` cycles (phase overflow), and `nbins > 100 000`.
  GPU entry points also reject more than 2³¹−1 points or frequencies
  (32-bit kernel indices); the fast GPU entry points reject more than 1e8 phase
  cycles (median-centred `max|t|·max(f)`, float32 phase accuracy) and a
  non-finite or negative `dlogq`.
- `auto_nbins` raises on `q_min <= 0` and warns when the 500-bin cap leaves fewer
  than 5 bins across the narrowest transit (`q_min < 0.01`).
- Multiband `best_depth` is NaN (was 0) for a band with no in-transit or no
  out-of-transit data at the best solution.
- New `BLSResult.n_bands_used` (all multiband entry points;
  `multiband_eebls_gpu_fast(..., compute_n_bands_used=False)` skips it): number of bands
  contributing at each frequency's best window — the noise floor of `power`
  grows with it.
- New `BLSResult.band_chi2_red` (multiband): per-band flat-model χ²/dof of the
  input uncertainties, and an opt-in `rescale_errors=True` on all multiband entry
  points that rescales each band's `dy` to χ²/dof = 1.
- **Fast GPU kernels use an exact point count:** `eebls_gpu_fast` /
  `multiband_eebls_gpu_fast` now enforce `min_points` as an exact global
  in-transit count (plus the per-band ≥1-point rule) instead of a
  weighted-fraction threshold that was biased for heteroscedastic `dy`. Costs
  ~1.2–1.4× in speed for single-band at larger N; fast-kernel periodograms on
  heteroscedastic data change.
- **Numerical fixes:** an exactly constant signal now gives zero power (the weighted
  mean left a ~1e-15 residue that produced spurious power, up to ~1e16 for a box
  holding every point); a band — or, single-band, the light curve — only
  contributes to a box with points both inside and outside it; the sparse
  searches no longer mistake tied phases for a phase wrap (a perfect eclipse
  with simultaneous points could return zero power) and never split points of
  identical phase between in- and out-of-eclipse.
- `build_frequency_grid` rejects non-positive or non-finite bounds, `df`,
  `oversample`, `q_min` and non-finite times.
- Cython cores use 64-bit integer buffers explicitly (portable to platforms
  where C `long` is 32-bit).
- Release workflow: cp39 wheels, wheel tests no longer need `pytest-cov`, smoke
  tests require the compiled cores, explicit `contents: write` permission.
- Generated Cython `.c` files are no longer tracked or shipped in the sdist;
  builds always cythonize the `.pyx` sources. CI builds and smoke-tests a wheel
  from the sdist.
