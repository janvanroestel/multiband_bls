"""Search backends: one :class:`~multiband_bls.engine.Implementation` per (method, backend)."""

from __future__ import annotations

from ..engine import Implementation

__all__ = ["get_implementation", "available_implementations"]


def _registry() -> dict[tuple[str, str, bool], type[Implementation]]:
    from . import cpu

    reg: dict[tuple[str, str, bool], type[Implementation]] = {
        (c.method, c.backend, c.multiband): c for c in cpu.IMPLEMENTATIONS
    }
    try:  # the GPU module needs CuPy only when a kernel is built, not at import
        from . import gpu
    except ImportError:  # pragma: no cover
        return reg
    reg.update({(c.method, c.backend, c.multiband): c for c in gpu.IMPLEMENTATIONS})
    return reg


def available_implementations() -> list[tuple[str, str, bool]]:
    """``(method, backend, multiband)`` keys of every implementation."""
    return sorted(_registry())


def get_implementation(method: str, backend: str, multiband: bool) -> Implementation:
    """Instantiate the implementation for ``(method, backend, multiband)``."""
    try:
        return _registry()[(method, backend, multiband)]()
    except KeyError:
        raise ValueError(
            f"no implementation for method={method!r}, backend={backend!r}, "
            f"multiband={multiband}; available: {available_implementations()}"
        ) from None
