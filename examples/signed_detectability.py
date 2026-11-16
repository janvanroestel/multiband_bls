"""Eclipse detectable with sign="positive" but not with the unsigned search.

Detection = the peak power within 1% of the true frequency exceeds the 1% false-alarm
threshold, which is calibrated separately for each search mode as the 99th percentile
of the grid-maximum power over pure-noise light curves with the same sampling.
Because a signed search only has to beat noise of one sign, its threshold is lower;
there is a narrow range of eclipse depths that clears the signed but not the
unsigned threshold. Panels: (a) noise-maximum distributions and thresholds,
(b) detection fraction vs depth, (c) periodograms of one example light curve.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from multiband_bls import multiband_eebls_fast

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("signed_detect")

Array = npt.NDArray[np.float64]
Bands = dict[str, tuple[Array, Array, Array]]
P_ECL, Q_ECL, SIGMA, BASE = 1.7305, 0.05, 0.05, 20.0
CHROM = {"u": 1.6, "g": 1.3, "r": 1.0, "i": 0.8, "z": 0.6, "y": 0.5}   # relative depths
NVIS = {"u": 40, "g": 60, "r": 120, "i": 120, "z": 100, "y": 100}
FREQS = np.arange(0.40, 1.20, 5e-5)
N_NOISE, N_TRIAL, FAP = 400, 80, 0.01
DEPTHS = (0.060, 0.070, 0.080, 0.090, 0.100, 0.115)     # r-band depth (mag)
NT = os.cpu_count() or 1
OUT = Path(__file__).with_name("signed_detectability.png")


def make_bands(rng: np.random.Generator, depth: float) -> Bands:
    bands: Bands = {}
    for b, n in NVIS.items():
        t = np.sort(rng.uniform(0.0, 365.0, n))
        y = BASE + rng.normal(0.0, SIGMA, n)
        ph = ((t / P_ECL + 0.5) % 1.0) - 0.5
        y = y + depth * CHROM[b] * (np.abs(ph) < Q_ECL / 2)
        bands[b] = (t, y, np.full(n, SIGMA))
    return bands


def search(bands: Bands, sign: str) -> Array:
    return multiband_eebls_fast(bands, FREQS, q_min=0.03, q_max=0.08, sign=sign,
                                n_threads=NT).power


def peak_at_truth(power: Array) -> float:
    near = np.abs(FREQS - 1.0 / P_ECL) < 0.01 / P_ECL
    return float(power[near].max())


def main() -> None:
    rng = np.random.default_rng(11)
    noise = {"both": [], "positive": []}
    for _ in range(N_NOISE):
        b = make_bands(rng, 0.0)
        for s in noise:
            noise[s].append(search(b, s).max())
    thr = {s: float(np.quantile(v, 1 - FAP)) for s, v in noise.items()}
    log.info("1%% FAP thresholds: unsigned %.4f, positive %.4f", thr["both"], thr["positive"])

    frac = {s: [] for s in noise}
    example = None
    for d in DEPTHS:
        hit = {s: 0 for s in noise}
        for k in range(N_TRIAL):
            b = make_bands(rng, d)
            pk = {s: peak_at_truth(search(b, s)) for s in noise}
            ok = {s: pk[s] > thr[s] for s in noise}
            for s in noise:
                hit[s] += ok[s]
            if ok["positive"] and not ok["both"] and (
                    example is None or pk["positive"] - thr["positive"] > example[2]):
                example = (d, b, pk["positive"] - thr["positive"])
        for s in noise:
            frac[s].append(hit[s] / N_TRIAL)
        log.info("depth %.3f: detected unsigned %.2f  signed %.2f", d, frac["both"][-1],
                 frac["positive"][-1])
    if example is None:
        raise SystemExit("no example light curve found; adjust DEPTHS")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(min(map(min, noise.values())), max(map(max, noise.values())), 40)
    for s, c, lab in (("both", "k", "unsigned"), ("positive", "tab:blue", 'sign="positive"')):
        ax[0].hist(noise[s], bins, histtype="step", color=c, label=lab)
        ax[0].axvline(thr[s], color=c, ls="--")
    ax[0].set_xlabel("maximum periodogram power (pure noise)"); ax[0].set_ylabel("count")
    ax[0].set_title("(a) noise maximum; dashed = 1% FAP threshold", fontsize=10)
    ax[0].legend(fontsize=8)
    for s, c, lab in (("both", "k", "unsigned"), ("positive", "tab:blue", 'sign="positive"')):
        ax[1].plot(DEPTHS, frac[s], "o-", color=c, label=lab)
    ax[1].set_xlabel("eclipse depth in r (mag)"); ax[1].set_ylabel("fraction detected")
    ax[1].axvline(example[0], color="gray", ls=":")
    ax[1].set_title("(b) detection rate", fontsize=10); ax[1].legend(fontsize=8)
    for s, c, lab in (("both", "k", "unsigned"), ("positive", "tab:blue", 'sign="positive"')):
        ax[2].plot(1 / FREQS, search(example[1], s), lw=0.7, color=c, label=lab, alpha=0.9)
        ax[2].axhline(thr[s], color=c, ls="--", lw=1)
    ax[2].axvline(P_ECL, color="tab:red", ls="--", lw=1.5, alpha=0.5, zorder=0)
    ax[2].set_xlabel("period (d)"); ax[2].set_ylabel(r"power $\Delta\chi^2/\chi^2_{flat}$")
    ax[2].set_title(f"(c) one light curve, depth {example[0]:.3f} mag", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT, dpi=150)
    log.info("saved %s", OUT)


if __name__ == "__main__":
    main()
