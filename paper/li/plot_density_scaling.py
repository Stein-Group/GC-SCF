#!/usr/bin/env python3
"""
SI Fig. S17 (Li40 density scaling): integrated absolute density perturbation
  int |rho^GC(r) - rho^can(r)| d^3r
at eta = 5 mHa versus the number of coupled atoms (center 4, surface center 7,
edges 29), with a linear fit through the origin.

Reads results/broadening/<scheme>/sweep.npz and results/canonical.npz
(run_broadening_sweep.py). Writes figures/fig_density_scaling_eta5e-03.png
and the numbers to figures/fig_density_scaling_eta5e-03.csv.
"""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from pyscf.dft import numint

from common import THIS_DIR, RESULTS_DIR, COUPLING_PRESETS, build_mol

FIG_DIR = THIS_DIR / "figures"
SCHEMES = ["center", "surface_center", "edges"]
ETA_TARGET = 5e-3
NX, NY, NZ, PAD, BOHR = 80, 80, 40, 2.0, 1.8897259886   # same grid as plot_density_maps.py

cm = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 300})


def main():
    mol, coords = build_mol()
    dm_can = np.load(RESULTS_DIR / "canonical.npz")["dm"]
    lo, hi = coords.min(0) - PAD, coords.max(0) + PAD
    x, y, z = (np.linspace(lo[i], hi[i], n) for i, n in enumerate((NX, NY, NZ)))
    dvol = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")
    ao = numint.eval_ao(mol, np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]) * BOHR)

    rows = []
    for scheme in SCHEMES:
        d = np.load(RESULTS_DIR / "broadening" / scheme / "sweep.npz")
        ok = d["converged"].astype(bool)
        k = int(np.flatnonzero(ok)[np.argmin(np.abs(d["eta"][ok] - ETA_TARGET))])
        val = float(np.abs(numint.eval_rho(mol, ao, d["P"][k] - dm_can)).sum() * dvol)
        rows.append({"scheme": scheme, "n_coupled": len(COUPLING_PRESETS[scheme]),
                     "eta_Ha": float(d["eta"][k]), "int_abs_drho_e": val})
        print(f"{scheme:15s} n_coupled={rows[-1]['n_coupled']:2d}  int|drho| = {val:.4f} e")

    n = np.array([r["n_coupled"] for r in rows], float)
    v = np.array([r["int_abs_drho_e"] for r in rows])
    slope = float(n @ v / (n @ n))

    fig, ax = plt.subplots(figsize=(8 * cm, 7 * cm),
                           gridspec_kw={"left": 0.18, "right": 0.97, "top": 0.88, "bottom": 0.18})
    ax.scatter(n, v, s=60, zorder=3, color="#2166ac")
    for r in rows:
        ax.annotate(r["scheme"].replace("_", " "), (r["n_coupled"], r["int_abs_drho_e"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5)
    xf = np.array([0, n.max() * 1.1])
    ax.plot(xf, slope * xf, "k--", lw=0.9, alpha=0.6, label=f"slope = {slope:.4f} e/atom")
    ax.set_xlabel("number of coupled atoms"); ax.set_ylabel(r"$\int|\Delta\rho|\,\mathrm{d}^3r$  [e]")
    ax.set_xlim(0, n.max() * 1.2); ax.set_ylim(0, v.max() * 1.25)
    ax.legend(fontsize=7.5); ax.grid(True, ls=":", alpha=0.3)
    fig.suptitle(rf"$\eta = {rows[0]['eta_Ha'] * 1e3:.0f}$ mHa  (PBE0)", fontsize=9)

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"fig_density_scaling_eta{rows[0]['eta_Ha']:.0e}.png"
    fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"slope = {slope:.4f} e/atom\nsaved {out}")


if __name__ == "__main__":
    main()
