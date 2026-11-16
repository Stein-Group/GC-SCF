#!/usr/bin/env python3
"""
Paper Fig. 3b: the three highest occupied (left) and three lowest unoccupied
(right) grand-canonical levels of H6 vs mu (MINAO: six levels in total), as 3D "curtains"
(x = level energy e_k, y = mu, z = occupation f_k). Red: levels with a
broadening gamma_k > 0.15 * eta (i.e. noticeable weight on the coupled AOs),
blue: the others. Dashed lines: canonical HOMO and LUMO.

Reads results/mu_sweep/H6_eps0_eta0.1.npz (run_mu_sweep.py),
writes figures/mu_sweep_H6.png.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from run_mu_sweep import OUT as DATA
from run_fragment_curves import THIS_DIR

FIG = THIS_DIR / "figures" / "mu_sweep_H6.png"
N_OCC = N_VIRT = 3
COUPLED_THRESHOLD = 0.15
COL_UNC, COL_CPL = "#4C72B0", "#C44E52"

cm = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "figure.dpi": 300,
                     "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})


def curtain(ax, x, y, z1, z2):
    verts = [[(x[i], y[i], z1[i]), (x[i + 1], y[i + 1], z1[i + 1]),
              (x[i + 1], y[i + 1], z2[i + 1]), (x[i], y[i], z2[i])] for i in range(len(x) - 1)]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.22, edgecolor="none", facecolor="gray"))


def main():
    d = np.load(DATA)
    mu, e, f, g = d["mu_grid"], d["mo_energy"], d["mo_occ_gc"], d["mo_gamma"]
    homo, lumo, eps, eta = float(d["homo"]), float(d["lumo"]), float(d["eps"]), float(d["eta"])

    # follow the levels by energy order at each mu
    order = np.argsort(e, axis=1)
    def level(k):
        idx = order[:, k]
        rows = np.arange(len(mu))
        return e[rows, idx], f[rows, idx], bool(np.any(g[rows, idx] > COUPLED_THRESHOLD * eta))

    fig = plt.figure(figsize=(9.5 * cm, 5.5 * cm))
    ax_o = fig.add_subplot(1, 2, 1, projection="3d")
    ax_v = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.08, top=0.95, wspace=0.18)
    ax_o.set_title("occupied", pad=0); ax_v.set_title("unoccupied", pad=0)

    # H6/MINAO: 6 levels, 3 occupied -> lowest three = occupied, highest three = unoccupied
    nlev = e.shape[1]
    for k in range(N_OCC):
        x, z, cpl = level(k)
        ax_o.plot(x, mu, z, color=COL_CPL if cpl else COL_UNC, lw=2.0)
        ax_o.plot(x, mu, np.full_like(z, 1.0), color="0.6")
        ax_o.plot(x, mu, np.full_like(z, 2.0), color="k")
        curtain(ax_o, x, mu, z, np.full_like(z, 2.0))
    for k in range(nlev - N_VIRT, nlev):
        x, z, cpl = level(k)
        ax_v.plot(x, mu, z, color=COL_CPL if cpl else COL_UNC, lw=2.0)
        ax_v.plot(x, mu, np.zeros_like(z), color="0.6")
        curtain(ax_v, x, mu, np.zeros_like(z), z)

    for ax, zb, zt in ((ax_o, 1.0, 2.0), (ax_v, 0.0, 0.0)):
        ax.autoscale()
        x0, x1 = ax.get_xlim()
        for m, lab in ((homo, "HOMO"), (lumo, "LUMO")):
            ax.plot([x0, x1], [m, m], [zb, zb], ls="--", color="k", lw=1.3, label=lab if ax is ax_o else None)
            ax.plot([x0, x1], [m, m], [zt, zt], ls="--", color="k", lw=1.3)
        ax.view_init(elev=25.0, azim=-60.0)
        ax.tick_params(axis="x", pad=-2); ax.tick_params(axis="y", pad=-3); ax.tick_params(axis="z", pad=-3)
        ax.set_xlabel(r"MO energy $e_i$ (Ha)", labelpad=-6)
        ax.set_ylabel(r"$\mu$ (Ha)", labelpad=-4)
        ax.set_zlabel("occupation", labelpad=-7)
    ax_o.legend(loc="upper right", fontsize=7, framealpha=0.5)
    ax_v.text2D(0.98, 0.95, rf"$\varepsilon = {eps:.1f}$ Ha" "\n" rf"$\gamma = {eta:.1f}$ Ha",
                transform=ax_v.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"))

    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=300, facecolor="white"); plt.close(fig)
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
