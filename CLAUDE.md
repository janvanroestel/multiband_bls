# multiband_BLS — Claude Code project notes

## Package layout

```
src/multiband_bls/
  reference.py      — pure-Python reference implementations; auto_nbins() lives here
  api.py            — typed entry points backed by compiled Cython cores
  gpu.py            — CuPy GPU entry points (eebls_gpu, multiband_eebls_gpu, ...)
  periodogram.py    — SBLSResult dataclass
  _eebls.pyx        — single-band binned BLS (box only)
  _meebls.pyx       — multiband binned BLS
  _msbls.pyx        — multiband sparse BLS
  _sbls.pyx         — single-band sparse BLS
tests/
  test_sbls.py      — correctness + recovery tests; uses an inline synthetic fixture
```

Demo scripts, benchmarks, and simulation utilities live in the companion paper repo
(`../multiband_BLS_paper/scripts/`), not here.

## Environments

- **cupy_env**: the working conda env — has CuPy and all dependencies.
  Always run scripts with `conda run -n cupy_env python ...` or activate first.
- Cython extensions are pre-compiled (`.so` files checked in).
  Recompile with `pip install -e . --no-build-isolation` if `.pyx` files change.

## Key algorithmic choices

### auto_nbins
`reference.auto_nbins(qmin)` = `clip(round(5/qmin), 50, 500)` — ensures at
least 5 bins inside the narrowest transit. Passing `nbins=None` to any entry
point triggers this automatically.

### Combined statistic
Multiband BLS combines bands with matched-filter weighting (each band weighted
by its total inverse variance W_b). `power` = ΔΧ²/Χ²_flat ∈ [0, 1].
