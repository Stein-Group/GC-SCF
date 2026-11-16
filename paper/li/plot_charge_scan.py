#!/usr/bin/env python3
"""
Li40 point-charge scan: Delta E(d) coupled vs uncoupled (top) and Delta N_e(d)
(bottom), paper Fig. 8.

Reads results/charge_scan/center/charge_scan.npz and
results/charge_scan/canonical_ref.npz, writes figures/fig_charge_scan_center.png
and the plotted numbers to figures/fig_charge_scan_center.csv.
"""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from common import THIS_DIR, RESULTS_DIR

HA_TO_KCAL = 627.5094740631
FIG_DIR = THIS_DIR / "figures"
cm = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "figure.dpi": 300,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.5})


def main():
    cpl = np.load(RESULTS_DIR / "charge_scan" / "center" / "charge_scan.npz")
    ref = np.load(RESULTS_DIR / "charge_scan" / "canonical_ref.npz")
    ok = cpl["converged"].astype(bool)
    dist, E_cpl, dNe = cpl["distance_ang"][ok], cpl["E_tot_full"][ok], cpl["dNe"][ok]
    m = np.isin(np.round(ref["distance_ang"], 6), np.round(dist, 6))
    E_ref = ref["E_tot_full"][m]
    y_cpl = (E_cpl - E_cpl[-1]) * HA_TO_KCAL
    y_ref = (E_ref - E_ref[-1]) * HA_TO_KCAL
    eta = float(cpl["gamma"])

    fig, (ax_e, ax_n) = plt.subplots(2, 1, sharex=True, figsize=(9.3 * cm, 7.0 * cm),
                                     gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.0,
                                                  "left": 0.16, "right": 0.95, "top": 0.96, "bottom": 0.13})
    ax_e.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax_e.plot(dist, y_ref, color="#7a7a7a", ls="--", marker="^", ms=4, label="Li$_{40}$, uncoupled")
    ax_e.plot(dist, y_cpl, color="#1f5fa8", marker="o", ms=4,
              label=rf"Li$_{{40}}$, coupled ($\eta={eta*1e3:.1f}$ mHa)")
    ax_e.set_ylabel(r"$\Delta E$ (kcal mol$^{-1}$)"); ax_e.grid(True, ls=":", alpha=0.4)
    ax_e.legend(fontsize=7.5, loc="upper right")
    ax_n.plot(dist, dNe, color="#1f5fa8", marker="o", ms=4); ax_n.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax_n.set_ylabel(r"$\Delta N_e$"); ax_n.set_xlabel(r"$d$ (Å)"); ax_n.grid(True, ls=":", alpha=0.4)
    ax_n.set_xlim(dist[0], dist[-1])

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "fig_charge_scan_center.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    with (FIG_DIR / "fig_charge_scan_center.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["d_ang", "dE_coupled_kcal", "dE_uncoupled_kcal", "dNe"])
        w.writerows(zip(dist, y_cpl, y_ref, dNe))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
