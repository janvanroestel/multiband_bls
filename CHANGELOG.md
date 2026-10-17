# Changelog

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
