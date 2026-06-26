# Changelog

## [0.1.0] — unreleased

Initial release.

- Single-band and multiband Sparse BLS (`sparse_bls`, `multiband_sparse_bls`)
- Single-band and multiband binned BLS (`eebls`, `multiband_eebls`)
- GPU-accelerated binned BLS via CuPy (`eebls_gpu`, `multiband_eebls_gpu`)
- Fast GPU variants using incremental accumulation over log-spaced widths (`eebls_gpu_fast`, `multiband_eebls_gpu_fast`)
- Frequency grid builder (`build_frequency_grid`) with period and frequency interfaces
- Pure-Python reference implementations for testing and verification
