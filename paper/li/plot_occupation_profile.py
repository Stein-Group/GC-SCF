#!/usr/bin/env python3
"""
Li40 MO energies and projected occupations vs eta (paper Fig. 6, SI Fig. S16).

For each eta, the three highest occupied and three lowest unoccupied canonical
(eta = 0) MOs C_i are tracked: energies e_i = C_i^T F[P(eta)] C_i and
grand-canonical occupations f_i = C_i^T S P(eta) S C_i. The self-consistent mu(eta)
is shown in black, the canonical HOMO/LUMO as dashed lines, and the canonical
gap as a vertical line.

Reads results/broadening/<scheme>/sweep.npz and results/canonical.npz,
writes figures/fig_occupation_profile_<scheme>.png and the plotted numbers
to figures/fig_occupation_profile_<scheme>.csv.
"""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from common import THIS_DIR, RESULTS_DIR, COUPLING_PRESETS, build_mol, make_mf

FIG_DIR = THIS_DIR / "figures"
NOCC, N_LEVELS, ETA_MAX = 60, 3, 0.05
# y range of the MO-energy panel (mHa); None = automatic
E_YLIM = {"center": (-100, -20), "surface_center": None, "edges": None}

cm = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 300,
                     "lines.linewidth": 1.2})


def level_color(k):
    if k < N_LEVELS:
        return plt.cm.Blues_r(0.3 + 0.55 * (N_LEVELS - k - 1) / max(N_LEVELS - 1, 1))
    return plt.cm.Reds(0.3 + 0.55 * (k - N_LEVELS) / max(N_LEVELS - 1, 1))


def spline(x, y, n=300):
    ok = np.isfinite(y) & (x <= ETA_MAX)
    if ok.sum() < 3:
        return np.array([]), np.array([])
    xs = np.log10(x[ok]); spl = make_interp_spline(xs, y[ok], k=3)
    xf = np.linspace(xs.min(), xs.max(), n)
    return 10 ** xf, spl(xf)


def main():
    mol, _ = build_mol()
    mf = make_mf(mol); mf.grids.build()
    Hcore, S = mf.get_hcore(), mol.intor("int1e_ovlp")
    can = np.load(RESULTS_DIR / "canonical.npz")
    homo, lumo = float(can["homo"]), float(can["lumo"]); gap = lumo - homo
    idx = np.arange(NOCC - N_LEVELS, NOCC + N_LEVELS)
    C = can["mo_coeff"][:, idx]; SC = S @ C
    FIG_DIR.mkdir(exist_ok=True)

    for scheme in COUPLING_PRESETS:
        f = RESULTS_DIR / "broadening" / scheme / "sweep.npz"
        if not f.exists():
            print(f"{scheme}: no sweep yet, skipped"); continue
        d = np.load(f)
        eta, conv, mu = d["eta"], d["converged"].astype(bool), d["mu"] * 1e3
        e = np.full((len(eta), len(idx)), np.nan); n = np.full_like(e, np.nan)
        for i, P in enumerate(d["P"]):
            if conv[i]:
                F = Hcore + mf.get_veff(mol, P)
                e[i] = np.diag(C.T @ F @ C) * 1e3
                n[i] = np.diag(SC.T @ P @ SC)
        show = conv & (eta <= ETA_MAX)

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10 * cm, 8 * cm), sharex=True,
                                       gridspec_kw={"hspace": 0.0, "left": 0.16, "right": 0.97,
                                                    "top": 0.97, "bottom": 0.1625})
        ax0.text(0.03, 0.88, f"coupling at {scheme.replace('_', ' ')}", transform=ax0.transAxes,
                 fontsize=8.5, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))
        for j in range(len(idx)):
            col, big = level_color(j), j in (N_LEVELS - 1, N_LEVELS)
            for ax, y in ((ax0, e[:, j]), (ax1, n[:, j])):
                xs, ys = spline(eta, y)
                if len(xs): ax.plot(xs, ys, color=col, lw=1.6 if big else 0.9)
                ax.scatter(eta[show], y[show], color=col, s=30 if big else 15, zorder=3)
        ax0.axhline(homo * 1e3, color="#2166ac", ls="--", lw=0.8, alpha=0.5, label="canonical HOMO/LUMO")
        ax0.axhline(lumo * 1e3, color="#d73027", ls="--", lw=0.8, alpha=0.5)
        xs, ys = spline(eta, mu)
        if len(xs): ax0.plot(xs, ys, color="k", lw=1.5, label="GC μ(η)")
        ax0.scatter(eta[show], mu[show], color="k", s=25, zorder=3)
        for ax in (ax0, ax1): ax.axvline(gap, color="gray", ls="--", lw=0.9)
        ax1.plot([], [], color="gray", ls="--", lw=0.9, label=f"canonical gap ({gap*1e3:.1f} mHa)")
        ax0.set_ylabel(r"MO energies $e_i$ (mHa)"); ax0.set_xscale("log")
        if E_YLIM.get(scheme): ax0.set_ylim(*E_YLIM[scheme])
        ax0.legend(fontsize=7.5, loc="upper right"); ax0.grid(True, which="both", ls=":", alpha=0.55)
        ax1.set_ylabel(r"Occupation $f$"); ax1.set_xlabel("η (Ha)")
        ax1.set_ylim(-0.1, 2.1); ax1.set_yticks([0, 0.5, 1, 1.5, 2])
        ax1.legend(fontsize=7.5, loc="center left"); ax1.grid(True, which="both", ls=":", alpha=0.55)
        out = FIG_DIR / f"fig_occupation_profile_{scheme}.png"
        fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)

        with (FIG_DIR / f"fig_occupation_profile_{scheme}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["eta_Ha", "mu_mHa", "converged"] + [f"e_MO{k}_mHa" for k in idx] + [f"n_MO{k}" for k in idx])
            for i in range(len(eta)):
                w.writerow([eta[i], mu[i], conv[i], *e[i], *n[i]])
        print(f"saved {out}")


if __name__ == "__main__":
    main()
