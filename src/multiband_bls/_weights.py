"""Inverse-variance weights and mean subtraction (Eqs. 1-2)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def preprocess(y: Array, dy: Array) -> tuple[Array, Array, float]:
    """Return ``(w_hat, x_tilde, mu)`` from Eqs. 1-2.

    ``w_hat`` are the normalised weights (sum to 1) and ``x_tilde`` is the
    weighted-mean-subtracted signal.

    Parameters
    ----------
    y :
        Magnitudes or fluxes, 1-D float array.
    dy :
        1-sigma uncertainties, 1-D float array, same length as ``y``.
    """
    y = np.asarray(y, dtype=float)
    w = 1.0 / np.asarray(dy, dtype=float) ** 2
    W = w.sum()
    w_hat = w / W
    if y.size and np.all(y == y[0]):
        # Exactly constant signal: sum(w_hat * y) = y[0] * (1 +- eps) would leave
        # a ~1e-15 residue in x_tilde and hence a spurious, noise-driven power.
        mu = float(y[0])
    else:
        mu = float(np.sum(w_hat * y))
    x_tilde = y - mu
    return w_hat, x_tilde, mu
