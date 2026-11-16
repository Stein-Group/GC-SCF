#!/usr/bin/env python3
"""
Paper Fig. 4 and SI Fig. S13.

  figures/fig4a_parameters.png    optimized eta and eps vs fragment size (from run_fragment_curves.py)
  figures/fig4b_energy_scales.png Delta E_HL, Delta_mu, k_B T_eff (with fit error) and k_B T_eff / Delta_mu
  figures/figS13_mu_alignment.png canonical HOMO/LUMO, closest GC levels around mu, and mu

Fig. 4b and S13 read results/electronic_structure/gaps_teff.csv (run_electronic_structure.py).
"""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from run_electronic_structure import OUT_DIR, SYSTEMS, REFERENCES
from run_fragment_curves import THIS_DIR, FRAGMENTS

FIG_DIR = THIS_DIR / "figures"
cm = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "figure.dpi": 300,
                     "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})


def tex(label):
    return rf"H$_{{{label.split('_')[0][1:]}}}$"


def colors(n):
    return plt.get_cmap("viridis")(np.linspace(0.12, 0.88, n))


def read_rows():
    with (OUT_DIR / "gaps_teff.csv").open() as fh:
        rows = {r["system"]: r for r in csv.DictReader(fh)}
    val = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan
    return rows, val


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved {out}")


def fig4a():
    n = np.array([int(s.split("_")[0][1:]) for s in SYSTEMS], float)
    eps = np.array([FRAGMENTS[s][1] for s in SYSTEMS]); eta = np.array([FRAGMENTS[s][2] for s in SYSTEMS])
    c = colors(len(SYSTEMS))
    fig, (ax_g, ax_e) = plt.subplots(2, 1, sharex=True, figsize=(8.5 * cm, 4.2 * cm), gridspec_kw={"hspace": 0.0})
    for ax, y, mk in ((ax_g, eta, "s"), (ax_e, eps, "o")):
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        ax.plot(n, y, color="0.2", lw=0.9, alpha=0.7, zorder=1)
        ax.scatter(n, y, c=c, s=28, marker=mk, edgecolor="k", linewidth=0.35, zorder=3)
        ax.grid(True, ls=":", alpha=0.55)
    ax_g.set_ylabel(r"$\eta$ (Ha)"); ax_e.set_ylabel(r"$\epsilon$ (Ha)")
    ax_g.set_ylim(0, 0.21); ax_g.set_yticks([0, 0.04, 0.08, 0.12, 0.16, 0.20])
    ax_e.set_ylim(-0.07, 0.0); ax_e.set_yticks([-0.06, -0.04, -0.02])
    ax_e.set_xticks(n); ax_e.set_xticklabels([tex(s) for s in SYSTEMS]); ax_e.set_xlabel("fragment size")
    for ax in (ax_g, ax_e):
        ax.yaxis.set_label_coords(-0.19, 0.5)
    save(fig, "fig4a_parameters")


def fig4b(rows, val):
    labels = SYSTEMS + list(REFERENCES)
    x = {s: float(i) for i, s in enumerate(SYSTEMS)}
    x |= {"H38": len(SYSTEMS) + 1.5, "H40": len(SYSTEMS) + 2.5}
    col = dict(zip(SYSTEMS, colors(len(SYSTEMS)))) | {"H38": "#55A868", "H40": "0.05"}
    fig, (ax, axb) = plt.subplots(2, 1, sharex=True, figsize=(8.5 * cm, 5.4 * cm),
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    for key, lab, ls, mk, c, refs in (("gap_q0", r"$\Delta E_\mathrm{HL}$", "-", "o", "0.15", True),
                                      ("delta_mu_q0", r"$\Delta_\mu$", "--", "s", "0.35", False),
                                      ("kbt_q0", r"$k_\mathrm{B}T_\mathrm{eff}$", ":", "D", "0.55", False)):
        xs = [x[s] for s in SYSTEMS]; ys = [val(rows[s], key) for s in SYSTEMS]
        ax.plot(xs, ys, color=c, lw=1.15, ls=ls, marker=mk, ms=4, label=lab, zorder=2)
        if refs:
            ax.plot([x[s] for s in REFERENCES], [val(rows[s], key) for s in REFERENCES],
                    color=c, lw=1.15, ls=ls, marker=mk, ms=4, zorder=2)
        for s in labels:
            y = val(rows[s], key)
            if not np.isfinite(y):
                continue
            if key == "kbt_q0" and np.isfinite(val(rows[s], "kbt_se_q0")):
                ax.errorbar(x[s], y, yerr=val(rows[s], "kbt_se_q0"), fmt="none", ecolor="0.35",
                            elinewidth=0.7, capsize=2.0, capthick=0.7, zorder=4)
            ax.scatter(x[s], y, color=col[s], s=24, marker=mk, edgecolor="k", linewidth=0.35, zorder=5)
    ax.set_ylabel("energy scale (Ha)"); ax.set_ylim(0, 0.42); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.grid(True, ls=":", alpha=0.55); ax.legend(fontsize=7, loc="upper right", framealpha=0.5)
    ratio = [val(rows[s], "kbt_over_delta_mu_q0") for s in SYSTEMS]
    axb.plot([x[s] for s in SYSTEMS], ratio, color="0.2", lw=1.1, alpha=0.8, zorder=1)
    axb.scatter([x[s] for s in SYSTEMS], ratio, c=[col[s] for s in SYSTEMS], s=26,
                edgecolor="k", linewidth=0.35, zorder=3)
    axb.set_ylabel(r"$k_\mathrm{B}T_\mathrm{eff}/\Delta_\mu$"); axb.set_ylim(0.25, 0.60); axb.set_yticks([0.3, 0.5])
    axb.set_xlabel("system"); axb.grid(True, ls=":", alpha=0.55)
    axb.set_xticks([x[s] for s in labels]); axb.set_xticklabels([tex(s) for s in labels])
    ax.yaxis.set_label_coords(-0.15, 0.5); axb.yaxis.set_label_coords(-0.15, 0.5)
    save(fig, "fig4b_energy_scales")


def figS13(rows, val):
    n = np.array([int(s.split("_")[0][1:]) for s in SYSTEMS], float)
    series = (("homo_q0", "HOMO", "#333333", "v", "--"), ("lumo_q0", "LUMO", "#333333", "^", "--"),
              ("gc_below_mu_q0", "GC-HOMO", "#4393c3", "s", "--"), ("gc_above_mu_q0", "GC-LUMO", "#4393c3", "D", "--"),
              ("mu", r"$\mu$ (neutrality constraint)", "#d6604d", "o", "-"))
    fig, ax = plt.subplots(figsize=(8.5 * cm, 4.8 * cm))
    for key, lab, c, mk, ls in series:
        y = [val(rows[s], key) for s in SYSTEMS]
        ax.plot(n, y, color=c, lw=1.0, ls=ls, alpha=0.8, zorder=1)
        ax.scatter(n, y, color=c, s=28, marker=mk, edgecolor=c, linewidth=0.5, label=lab, zorder=3)
    ax.set_ylabel("energy (Ha)"); ax.set_xlabel("fragment size")
    ax.set_xticks(n); ax.set_xticklabels([tex(s) for s in SYSTEMS])
    ax.grid(True, ls=":", alpha=0.55); ax.legend(fontsize=7, loc="best", framealpha=0.5)
    save(fig, "figS13_mu_alignment")


def main():
    fig4a()
    if (OUT_DIR / "gaps_teff.csv").exists():
        rows, val = read_rows()
        fig4b(rows, val); figS13(rows, val)
    else:
        print("no gaps_teff.csv yet: run run_electronic_structure.py first (Fig. 4b, S13 skipped)")


if __name__ == "__main__":
    main()
