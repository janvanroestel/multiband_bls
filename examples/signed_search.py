"""Unsigned vs signed multiband BLS on a light curve with two periodic signals.

A weak, real eclipse (dimming; magnitudes go UP) at P_ECL competes with a stronger
periodic brightening (magnitudes go DOWN) at P_BRT, e.g. an outburst/systematic.
The unsigned search picks the stronger brightening; ``sign="positive"`` (dimmings only)
recovers the eclipse; ``sign="negative"`` recovers the brightening.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from multiband_bls import BLSResult, multiband_eebls_fast

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("signed_example")

Array = npt.NDArray[np.float64]
P_ECL, P_BRT = 1.7305, 0.6180      # days
Q_ECL, Q_BRT = 0.05, 0.05          # fractional durations
DEPTH_ECL = {"g": 0.08, "r": 0.06, "i": 0.04}      # dimming (+mag)
DEPTH_BRT = {"g": -0.12, "r": -0.10, "i": -0.08}   # brightening (-mag)
SIGMA, BASE = 0.04, 20.0
OUT = Path(__file__).with_name("signed_search.png")


def make_bands(rng: np.random.Generator) -> dict[str, tuple[Array, Array, Array]]:
    bands = {}
    for b, n in zip("gri", (150, 250, 200)):
        t = np.sort(rng.uniform(0.0, 365.0, n))
        y = BASE + rng.normal(0.0, SIGMA, n)
        for P, q, dep in ((P_ECL, Q_ECL, DEPTH_ECL), (P_BRT, Q_BRT, DEPTH_BRT)):
            ph = ((t / P + 0.5) % 1.0) - 0.5
            y = y + dep[b] * (np.abs(ph) < q / 2)
        bands[b] = (t, y, np.full(n, SIGMA))
    return bands


def fold_panel(ax: plt.Axes, bands: dict, res: BLSResult, title: str) -> None:
    P = res.best_period
    for (b, (t, y, _)), c in zip(bands.items(), ("tab:green", "tab:red", "tab:brown")):
        ph = ((t - res.best_t0) / P + 0.5) % 1.0 - 0.5
        ax.plot(ph, y - np.mean(y), ".", ms=3, color=c, alpha=0.6, label=b)
    ax.invert_yaxis()
    ax.set_xlabel("phase"); ax.set_ylabel("mag - mean (brighter up)")
    ax.set_title(f"{title}: P = {P:.4f} d", fontsize=10)


def main() -> None:
    rng = np.random.default_rng(3)
    bands = make_bands(rng)
    freqs = np.linspace(0.3, 3.0, 40000)
    res = {s: multiband_eebls_fast(bands, freqs, q_min=0.03, q_max=0.08, sign=s)
           for s in ("both", "positive", "negative")}
    for s, r in res.items():
        log.info("%-8s best P = %.4f d  power = %.3f", s, r.best_period, r.best_power)
    log.info("truth: eclipse P = %.4f d, brightening P = %.4f d", P_ECL, P_BRT)

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=(2, 1))
    for i, (s, title) in enumerate((("both", "unsigned"), ("positive", 'sign="positive" (dimmings)'),
                                    ("negative", 'sign="negative" (brightenings)'))):
        ax = fig.add_subplot(gs[i, 0]); r = res[s]
        ax.plot(1 / r.frequency, r.power, lw=0.7, color="k", zorder=3)
        ax.axvline(P_ECL, color="tab:blue", ls="--", lw=1.5, alpha=0.6, zorder=1, label="true eclipse")
        ax.axvline(P_BRT, color="tab:orange", ls="--", lw=1.5, alpha=0.6, zorder=1, label="true brightening")
        ax.set_xscale("log"); ax.set_ylim(0, max(res["both"].power.max() * 1.1, 0.05))
        ax.set_xlabel("period (d)"); ax.set_ylabel(r"power $\Delta\chi^2/\chi^2_{flat}$")
        ax.set_title(title, fontsize=10)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        fold_panel(fig.add_subplot(gs[i, 1]), bands, r, title)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    log.info("saved %s", OUT)


if __name__ == "__main__":
    main()
