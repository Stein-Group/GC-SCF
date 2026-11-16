#!/usr/bin/env python3
"""
Plot the density comparison written by run_density_comparison.py
(paper Fig. 3d, SI Figs. S10b and S12).

Three panels per system: (i) fragment density, (ii) local H40 density,
(iii) difference. White cross: point charge. One colour scale for all
systems plotted together (density and difference separately).

Writes figures/density_comparison_<label>.png.

Usage:
    python plot_density_comparison.py            # all systems found
    python plot_density_comparison.py H6 H6_alt  # selected systems
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from run_density_comparison import RESULTS_DIR, THIS_DIR

FIG_DIR = THIS_DIR / "figures"
cm = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "figure.dpi": 300,
                     "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})


def label_tex(label: str) -> str:
    return rf"H$_{{{label.split('_')[0][1:]}}}$"


def plot_one(label, d, vmax_rho, vmax_diff):
    ext = tuple(d["extent"])
    xpc = float(d["x_charge"]) - float(d["center"][0])
    ypc = -float(d["center"][1])

    fig = plt.figure(figsize=(9.0 * cm, 4.0 * cm))
    gs = fig.add_gridspec(1, 6, width_ratios=[0.99, 0.99, 0.055, 0.6, 0.99, 0.055], wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax1 = fig.add_subplot(gs[0, 2])
    fig.add_subplot(gs[0, 3]).axis("off")
    ax3 = fig.add_subplot(gs[0, 4])
    cax2 = fig.add_subplot(gs[0, 5])

    kw = dict(origin="lower", extent=ext)
    ax1.imshow(d["rho_frag"], cmap="magma", vmin=0, vmax=vmax_rho, **kw)
    im2 = ax2.imshow(d["rho_ring"], cmap="magma", vmin=0, vmax=vmax_rho, **kw)
    im3 = ax3.imshow(d["rho_diff"], cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff, **kw)

    for ax, txt in ((ax1, rf"(i) {label_tex(label)}"), (ax2, r"(ii) H$_{40}$"), (ax3, "(iii) Diff.")):
        ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.plot([xpc], [ypc], marker="+", ms=6, mew=1.8, color="white", zorder=10)
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0))

    for im, cax, ax, lab in ((im2, cax1, ax2, r"$\rho(\mathbf{r})$"), (im3, cax2, ax3, r"$\Delta\rho(\mathbf{r})$")):
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(lab, rotation=90, labelpad=6)
    fig.subplots_adjust(left=0.0, right=0.8, bottom=0.05, top=0.98, wspace=0.1)
    fig.canvas.draw()
    for cax, ax in ((cax1, ax2), (cax2, ax3)):     # colour bars as tall as the panels
        a, c = ax.get_position(), cax.get_position()
        cax.set_position([c.x0, a.y0, c.width, a.height])

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"density_comparison_{label}.png"
    fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
    print(f"saved {out}")


def main():
    labels = sys.argv[1:] or sorted(p.name for p in RESULTS_DIR.iterdir()
                                    if (p / "density_comparison.npz").exists())
    data = {l: np.load(RESULTS_DIR / l / "density_comparison.npz") for l in labels}
    vmax_rho = max(max(d["rho_frag"].max(), d["rho_ring"].max()) for d in data.values())
    vmax_diff = max(np.abs(d["rho_diff"]).max() for d in data.values())
    print(f"colour scale: rho_max = {vmax_rho:.4e}, |diff|_max = {vmax_diff:.4e}")
    for l, d in data.items():
        plot_one(l, d, vmax_rho, vmax_diff)


if __name__ == "__main__":
    main()
