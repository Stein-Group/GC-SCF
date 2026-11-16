#!/usr/bin/env python3
"""
H-ring potential curves of the approaching point charge.

  single  (default): one fragment against H40, H38 and the uncoupled fragment,
           with N_e(r) of the coupled fragment on a second axis and the residual
           relative to H40 in the lower panel
           -> paper Fig. 3c (H6), SI Fig. S10a (H6_alt)
  overlay : all fragments H6-H20 (coupled solid, uncoupled dashed) with H40,
           H38, an inset around the minimum and the residuals
           -> SI Fig. S11

Reads results/fragment_curves/<label>/curve.npz (run_fragment_curves.py) and
results/reference_curves/<system>/curve.npz (run_reference_curves.py).
Fragment curves are interpolated onto the r grid of the H40 reference.

Usage:
    python plot_curves.py                 # Fig. 3c (H6)
    python plot_curves.py H6_alt          # Fig. S10a
    python plot_curves.py --overlay       # Fig. S11

Writes figures/curve_<label>.png or figures/curves_overlay.png.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from run_fragment_curves import THIS_DIR, FRAGMENTS

FRAG_DIR = THIS_DIR / "results" / "fragment_curves"
REF_DIR = THIS_DIR / "results" / "reference_curves"
FIG_DIR = THIS_DIR / "figures"
OVERLAY = ["H6", "H8_a", "H10", "H12", "H14", "H16", "H18", "H20"]

RMAX, YLIM, RESID_LIM = 12.0, (-32.0, 65.0), (-5.0, 5.0)
INSET_XLIM, INSET_YLIM = (1.2, 4.0), (-25.0, 0.0)
COL = {"H40": "#1a1a1a", "H38": "#55A868", "unc": "#ff9900", "cpl": "#472a79"}

cm = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "figure.dpi": 300,
                     "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})


def tex(label: str) -> str:
    return rf"H$_{{{label.split('_')[0][1:]}}}$"


def reference(system: str):
    d = np.load(REF_DIR / system / "curve.npz")
    return d["r_ang"], d["DeltaE_kcal"]


def fragment(label: str, r):
    d = np.load(FRAG_DIR / label / "curve.npz")
    return (np.interp(r, d["r_ang"], d["DeltaOmega_kcal"]), np.interp(r, d["r_ang"], d["Ne"]),
            float(d["eps"]), float(d["eta"]))


def add_rmin(ax, rmin, label=False):
    ax.axvline(rmin - 0.03, color="k", ls="--", lw=1.0, alpha=0.7)
    if label:
        ax.text(rmin + 0.08, ax.get_ylim()[1] * 0.92, r"$r_\min$", fontsize=8, rotation=90, va="top")


def finish_axes(ax, axr, rmin):
    ax.set_ylabel(r"$\Delta\Omega(r)$ (kcal/mol)", labelpad=1.5)
    ax.set_xlim(0, RMAX); ax.set_ylim(*YLIM); ax.grid(True, ls=":", alpha=0.6)
    ax.set_yticks(ax.get_yticks()[1:])
    axr.axhline(0, color="k", lw=1.0, alpha=0.6)
    axr.axvspan(-0.03, rmin, color="grey", alpha=0.5, zorder=0)
    axr.set_xlabel(r"$r$ (Å)"); axr.set_ylabel(r"$\Delta\Delta\Omega(r)$", labelpad=5)
    axr.set_xlim(0, RMAX); axr.set_ylim(*RESID_LIM); axr.grid(True, ls=":", alpha=0.6)


def plot_single(label: str):
    r, y40 = reference("H40")
    n = label.split("_")[0]
    y38, yunc = np.interp(r, *reference("H38")), np.interp(r, *reference(n))
    ycpl, Ne, eps, eta = fragment(label, r)
    rmin = float(r[np.argmin(y40)])

    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(9.3 * cm, 4.8 * 1.2 * cm),
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    ax.plot(r, y40, lw=1.5, color=COL["H40"], label=r"H$_{40}$, Ref.")
    ax.plot(r, y38, lw=1.5, color=COL["H38"], alpha=0.9, label=r"H$_{38}$, uncoupled")
    ax.plot(r, yunc, lw=1.5, color=COL["unc"], label=rf"{tex(label)}, uncoupled")
    ax.plot(r, ycpl, lw=1.5, color=COL["cpl"],
            label=rf"{tex(label)}, coupled ($\varepsilon={eps:.3f}$, $\eta={eta:.3f}$)")
    ax2 = ax.twinx()
    ax2.plot(r, Ne, ls=":", lw=1.5, color=COL["cpl"], label=r"$N_e(r)$")
    ax2.set_ylabel(r"$N_e(r)$", color=COL["cpl"]); ax2.tick_params(axis="y", labelcolor=COL["cpl"])
    finish_axes(ax, axr, rmin)
    add_rmin(ax, rmin, label=True); add_rmin(axr, rmin)
    for y, c in ((y38, "H38"), (yunc, "unc"), (ycpl, "cpl")):
        axr.plot(r, y - y40, lw=1.5, color=COL[c])
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7, framealpha=0.5)
    save(fig, f"curve_{label}")


def plot_overlay():
    r, y40 = reference("H40")
    y38 = np.interp(r, *reference("H38"))
    rmin = float(r[np.argmin(y40)])
    labels = [l for l in OVERLAY if (FRAG_DIR / l / "curve.npz").exists()]
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(labels)))

    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(8.5 * cm, 6.0 * cm),
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    axins = inset_axes(ax, width="20%", height="30%", loc="upper left",
                       bbox_to_anchor=(0.15, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=1.05)
    for a in (ax, axins):
        a.plot(r, y40, color=COL["H40"], lw=0.8, zorder=20, label=r"H$_{40}$ reference" if a is ax else None)
        a.plot(r, y38, color="#666666", lw=0.8, ls=":", alpha=0.75, label=r"H$_{38}$ uncoupled" if a is ax else None)
    axr.plot(r, y38 - y40, color="#666666", lw=0.8, ls=":", alpha=0.75)

    for label, c in zip(labels, colors):
        yunc = np.interp(r, *reference(label.split("_")[0]))
        ycpl = fragment(label, r)[0]
        for a in (ax, axins):
            a.plot(r, yunc, color=c, lw=0.8, ls="--", alpha=0.38)
            a.plot(r, ycpl, color=c, lw=1.45 if a is ax else 1.1, alpha=0.82,
                   label=tex(label) if a is ax else None)
        axr.plot(r, yunc - y40, color=c, lw=0.9, ls="--", alpha=0.38)
        axr.plot(r, ycpl - y40, color=c, lw=1.15, alpha=0.75)

    finish_axes(ax, axr, rmin)
    axr.patches[0].set_alpha(0.25)
    add_rmin(ax, rmin); add_rmin(axr, rmin); add_rmin(axins, rmin)
    ax.text(rmin + 0.05, YLIM[0] + 0.15 * (YLIM[1] - YLIM[0]), r"$r_\mathrm{min}$", fontsize=8, rotation=90)
    axins.set_xlim(*INSET_XLIM); axins.set_ylim(*INSET_YLIM)
    axins.grid(True, ls=":", alpha=0.45); axins.tick_params(labelsize=6.2, pad=1)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.45", lw=0.6, alpha=0.7)
    leg = ax.legend(loc="upper right", fontsize=6.2, ncol=2, framealpha=0.5)
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([0], [0], color="0.2", lw=1.45, label="coupled"),
                       Line2D([0], [0], color="0.2", lw=1.0, ls="--", alpha=0.55, label="uncoupled")],
              loc="lower right", fontsize=6.2, framealpha=0.5)
    save(fig, "curves_overlay")


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", nargs="?", default="H6", choices=list(FRAGMENTS))
    ap.add_argument("--overlay", action="store_true", help="all fragments in one figure (SI Fig. S11)")
    a = ap.parse_args()
    plot_overlay() if a.overlay else plot_single(a.label)


if __name__ == "__main__":
    main()
