"""Figure for the README: what the DensityPrior parameters do (cf. Hippke & Heller 2019, Fig. 5)."""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiband_bls import DensityPrior
from multiband_bls.density import central_duration_fraction

log = logging.getLogger(__name__)

RHO = (0.2, 5.0)
DUR = (0.1, 1.5)
prior = DensityPrior(density_range=RHO, duration_range=DUR)
P = np.logspace(-1, 2, 400)
f = 1.0 / P

q_center_lo = central_duration_fraction(f, RHO[0])  # least dense star, central transit
q_center_hi = central_duration_fraction(f, RHO[1])  # densest star, central transit
q_lo, q_hi = prior.bounds(f)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
# fixed-width search a plain BLS would use
ax.axhspan(0.01, 0.15, color="0.85", zorder=0, label="fixed $q_{min}$–$q_{max}$ (0.01–0.15)")
# allowed region
ax.fill_between(P, q_lo, q_hi, color="C0", alpha=0.25, label="allowed by the prior")
ax.plot(P, q_lo, color="C0", lw=1.2, label=r"$q_{lo}$: $\rho_{max}$ × duration$_{min}$")
ax.plot(P, q_hi, color="C0", lw=1.2, label=r"$q_{hi}$: $\rho_{min}$ × duration$_{max}$")
ax.plot(P, q_center_lo, "k--", lw=0.9, label=r"central transit, $\rho_{min}$")
ax.plot(P, q_center_hi, "k:", lw=0.9, label=r"central transit, $\rho_{max}$")
ax.axhline(prior.q_cap, color="C3", lw=1, label="q_cap")

# segments: each searched with its own (q_min, q_max) box, a new one once q_lo or q_hi
# has grown by q_step
freqs = np.sort(f)
for idx, spec in prior.segments(freqs, binned=False):
    p0, p1 = 1.0 / freqs[idx].max(), 1.0 / freqs[idx].min()
    ax.hlines([spec.q_min, spec.q_max], p0, p1, color="C1", lw=1.6)
ax.plot([], [], color="C1", lw=1.6, label="searched $q_{min}$, $q_{max}$ per segment")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, 100)
ax.set_ylim(2e-3, 0.8)
ax.set_xlabel("Period P [d]")
ax.set_ylabel("Duty cycle q = T/P")
ax.set_title("DensityPrior(density_range=(0.2, 5), duration_range=(0.1, 1.5))", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig("figures/density_prior.png", dpi=150)
log.info("saved figures/density_prior.png")
