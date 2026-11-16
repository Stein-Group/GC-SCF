#!/usr/bin/env python3
"""
Li40 density-difference maps at eta = 5 mHa (paper Fig. 7).

Delta rho(r) = rho^GC(r) - rho^can(r) for the three coupling schemes, evaluated
on a 3D grid and integrated along y (side view, XZ plane) and along z (top
view, XY plane, shown rotated by 90 deg). Black dots: Li atoms; white dots:
coupled atoms. One colour scale for all panels.

Reads results/broadening/<scheme>/sweep.npz and results/canonical.npz
(written by run_broadening_sweep.py).

Usage:
    python plot_density_maps.py              # Delta rho
    python plot_density_maps.py --absolute   # rho^GC itself

Writes figures/fig_density_<mode>_eta5e-03.png and the projected maps
(figure data) to figures/fig_density_<mode>_eta5e-03.npz.
"""
from __future__ import annotations

import argparse

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pyscf.dft import numint

from common import THIS_DIR, RESULTS_DIR, COUPLING_PRESETS, build_mol

FIG_DIR = THIS_DIR / "figures"
SCHEMES = ["center", "surface_center", "edges"]
ETA_TARGET = 5e-3            # Ha
NX, NY, NZ = 80, 80, 40      # grid points
PAD = 2.0                    # padding around the cluster (A)
BOHR = 1.8897259886          # 1 A in bohr
SCALE = 0.26 / 2.54          # figure inches per A (same scale in all panels)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.dpi": 300})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--absolute", action="store_true", help="plot rho^GC instead of Delta rho")
    mode = "absolute" if ap.parse_args().absolute else "diff"

    mol, coords = build_mol()
    dm_can = np.load(RESULTS_DIR / "canonical.npz")["dm"]

    # real-space grid, AO values evaluated once
    lo, hi = coords.min(0) - PAD, coords.max(0) + PAD
    x, y, z = (np.linspace(lo[i], hi[i], n) for i, n in enumerate((NX, NY, NZ)))
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")
    ao = numint.eval_ao(mol, np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]) * BOHR)
    dy, dz = y[1] - y[0], z[1] - z[0]

    maps, eta_used = {}, None
    for scheme in SCHEMES:
        f = RESULTS_DIR / "broadening" / scheme / "sweep.npz"
        if not f.exists():
            print(f"{scheme}: no sweep file, skipped"); continue
        d = np.load(f)
        ok = d["converged"].astype(bool)
        k = int(np.flatnonzero(ok)[np.argmin(np.abs(d["eta"][ok] - ETA_TARGET))])
        dm = d["P"][k] - dm_can if mode == "diff" else d["P"][k]
        rho = numint.eval_rho(mol, ao, dm).reshape(NX, NY, NZ)
        maps[scheme] = {"side": rho.sum(1) * dy,      # (x, z)
                        "top": rho.sum(2) * dz}       # (x, y)
        eta_used = float(d["eta"][k])
        print(f"{scheme}: eta = {eta_used:.1e} Ha")
    if not maps:
        return

    vmax = max(np.abs(m[v]).max() for m in maps.values() for v in ("side", "top"))
    if mode == "diff":
        norm, cmap, dot = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), "RdBu_r", "k"
    else:
        norm, cmap, dot = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax), "coolwarm", "0.9"

    # side view: x horizontal, z vertical; top view rotated by 90 deg: y horizontal, -x vertical
    side_ext = [x[0], x[-1], z[-1], z[0]]
    top_ext = [y[0], y[-1], -x[-1], -x[0]]
    w = [SCALE * (x[-1] - x[0]), SCALE * (y[-1] - y[0])]
    h = [SCALE * (z[-1] - z[0]), SCALE * (x[-1] - x[0])]
    m_lr, m_top, cb_gap, cb_h, m_bot = 0.06, 0.06, 0.14, 0.14, 0.22
    row_h = max(h)
    fig_w = 2 * m_lr + sum(w)
    fig_h = m_top + len(maps) * row_h + cb_gap + cb_h + m_bot
    fig = plt.figure(figsize=(fig_w, fig_h))

    for r, (scheme, m) in enumerate(maps.items()):
        coupled = np.isin(np.arange(mol.natm), COUPLING_PRESETS[scheme])
        top = fig_h - m_top - r * row_h
        panels = [
            (m["side"].T, side_ext, coords[:, 0], coords[:, 2], "side view (XZ)", "upper"),
            (m["top"], top_ext, coords[:, 1], -coords[:, 0], "top view (XY)", "upper"),
        ]
        left = m_lr
        for c, (img, ext, u, v, label, origin) in enumerate(panels):
            ax = fig.add_axes([left / fig_w, (top - h[c]) / fig_h, w[c] / fig_w, h[c] / fig_h])
            left += w[c]
            im = ax.imshow(img, extent=ext, origin=origin, cmap=cmap, norm=norm,
                           interpolation="bilinear", aspect="auto")
            ax.scatter(u, v, c=dot, s=6, lw=0, zorder=3)
            ax.scatter(u[coupled], v[coupled], c="white", s=12, lw=0.6, edgecolors="k", zorder=4)
            ax.set_xticks([]); ax.set_yticks([])
            bar = 1.0 / (ext[1] - ext[0])
            ax.plot([0.05, 0.05 + bar], [0.05, 0.05], transform=ax.transAxes, color=dot, lw=2)
            ax.text(0.05 + bar / 2, 0.07, "1 Å", transform=ax.transAxes, ha="center", fontsize=7, color=dot)
            ax.text(0.97, 0.03, label, transform=ax.transAxes, ha="right", fontsize=7,
                    color="0.25" if mode == "diff" else "0.75")
            if c == 0:
                ax.text(0.03, -0.04, scheme.replace("_", " "), transform=ax.transAxes, va="top",
                        fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))

    cax = fig.add_axes([m_lr / fig_w, m_bot / fig_h, sum(w) / fig_w, cb_h / fig_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(("Δρ" if mode == "diff" else "ρ") + " (e Å$^{-2}$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"fig_density_{mode}_eta{eta_used:.0e}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    np.savez_compressed(out.with_suffix(".npz"), x=x, y=y, z=z, eta=eta_used, mode=mode,
                        **{f"{s}__{v}": maps[s][v] for s in maps for v in ("side", "top")})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
