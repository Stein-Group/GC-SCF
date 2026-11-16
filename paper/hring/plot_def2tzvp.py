#!/usr/bin/env python3
"""
Paper Fig. 5 (def2-TZVP), from the results of run_def2tzvp.py.

  figures/fig5b_def2tzvp_curves.png
      Delta Omega(r) of H40, H38, uncoupled H6 and H6 with the s and s+p
      couplings (parameters in the legend), N_e(r) of the coupled fragments on
      the right axis, residuals relative to H40 below; dashed line at the
      minimum of the H40 curve.
  figures/fig5a_def2tzvp_density_r<r>.png  (one per density point)
      Delta rho(r) = rho_frag(r) - w(r) rho_H40(r) in the ring plane, chain
      horizontal, charge above (white cross); left half s coupling, right half
      s+p coupling. Hirshfeld weight w = sum_{fragment atoms} rho0_A /
      sum_{all 40 atoms} rho0_B with free-atom H 1s densities rho0 = exp(-2 r).
      The maps are also written as .npz (figure data).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from run_def2tzvp import OUT, COUPLED, DENSITY_R, R_MIN_TZVP, FRAG_XYZ, REFS, build_mol
from run_fragment_curves import THIS_DIR

FIG_DIR = THIS_DIR / "figures"
ANG2BOHR = 1.8897259886
RING_IDS = list(range(17, 23))                 # H40 atoms at the H6 positions
SPAN, NX, NY = (3.0, 6.0), 241, 401            # plane (A) and grid, as for the MINAO maps
U_LIM, V_LIM, VMAX = (-2.55, 2.55), (-0.85, 1.50), 0.13
STYLE = {"H6_s": ("#2a788e", "--", (0, (6, 3)), 1.3), "H6_sp": ("#472a79", "-", ":", 1.6)}

cm = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.0, "figure.dpi": 300})


def curve(name):
    return np.load(OUT / name / "curve.npz")


def legend_label(name):
    p = COUPLED[name]
    if "eps_p" in p:
        return (rf"H$_6$, s + p (Ha): $\varepsilon_s$ = {p['eps_s']:.3f}, $\eta_s$ = {p['eta_s']:.3f},"
                "\n" rf"$\varepsilon_p$ = {p['eps_p']:.3f}, $\eta_p$ = {p['eta_p']:.3f}")
    return rf"H$_6$, s (Ha): $\varepsilon_s$ = {p['eps_s']:.3f}, $\eta_s$ = {p['eta_s']:.3f}"


def fig5b():
    ref = curve("H40")
    r, y40 = ref["r_ang"], ref["DeltaE_kcal"]
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(8.5 * cm, 6.5 * cm),
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    ax.plot(r, y40, lw=1.5, color="#1a1a1a", label=r"H$_{40}$, ref.")
    for name, col, lab in (("H38", "#55A868", r"H$_{38}$, uncoupled"), ("H6", "#ff9900", r"H$_6$, uncoupled")):
        y = np.interp(r, curve(name)["r_ang"], curve(name)["DeltaE_kcal"])
        ax.plot(r, y, lw=1.5, color=col, label=lab); axr.plot(r, y - y40, lw=1.5, color=col)
    ax2 = ax.twinx()
    for name in COUPLED:
        c = curve(name); col, ls, ls_ne, lw_ne = STYLE[name]
        y = np.interp(r, c["r_ang"], c["DeltaOmega_kcal"])
        ax.plot(r, y, lw=1.5, color=col, ls=ls, label=legend_label(name))
        axr.plot(r, y - y40, lw=1.5, color=col, ls=ls)
        ax2.plot(r, np.interp(r, c["r_ang"], c["Ne"]), color=col, ls=ls_ne, lw=lw_ne,
                 label=r"$N_e(r)$" if name == "H6_sp" else None)
    for a in (ax, axr):
        a.axvline(R_MIN_TZVP, color="k", ls="--", lw=1.0, alpha=0.7); a.grid(True, ls=":", alpha=0.6)
    ax.text(R_MIN_TZVP + 0.08, 140, r"$r_\min$", rotation=90, va="top", fontsize=7)
    ax.set_xlim(0, 12); ax.set_ylim(-95, 150); ax.set_yticks([-50, 0, 50])
    ax.set_ylabel(r"$\Delta\Omega(r)$ (kcal/mol)")
    ax2.set_yticks([6.25, 6.5, 6.75, 7.0, 7.25]); ax2.set_ylabel(r"$N_e(r)$")
    axr.axhline(0, color="k", lw=1.0, alpha=0.6); axr.axvspan(-0.03, R_MIN_TZVP, color="grey", alpha=0.5, zorder=0)
    axr.set_ylim(-5, 5); axr.set_xlabel(r"$r$ (Å)"); axr.set_ylabel(r"$\Delta\Delta\Omega(r)$")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=5.5, framealpha=0.8)
    save(fig, "fig5b_def2tzvp_curves", bbox_inches="tight")


def promolecule(pts_bohr, atoms_ang):
    d = np.linalg.norm(pts_bohr[:, None, :] - atoms_ang[None] * ANG2BOHR, axis=-1)
    return np.exp(-2.0 * d)


def rho_on(mol, dm, pts_ang):
    ao = mol.eval_gto("GTOval_sph", pts_ang * ANG2BOHR)
    return np.einsum("pi,ij,pj->p", ao, np.real(dm), ao, optimize=True).reshape(NY, NX)


def fig5a(r):
    tag = f"r{r:.4f}"
    ring, frag = build_mol(REFS["H40"]), build_mol(FRAG_XYZ)
    xyz = ring.atom_coords(unit="Ang")
    center = xyz[RING_IDS].mean(0); center[2] = 0.0
    gx = np.linspace(-SPAN[0] / 2, SPAN[0] / 2, NX); gy = np.linspace(-SPAN[1] / 2, SPAN[1] / 2, NY)
    A, B = np.meshgrid(gx, gy, indexing="xy")
    pts = np.stack([center[0] + A, center[1] + B, np.zeros_like(A)], -1).reshape(-1, 3)
    pro = promolecule(pts * ANG2BOHR, xyz)
    w = (pro[:, RING_IDS].sum(1) / pro.sum(1)).reshape(NY, NX)
    rho_ref = w * rho_on(ring, np.load(OUT / "density" / f"dm_H40__{tag}.npy"), pts)
    diff = {n: rho_on(frag, np.load(OUT / "density" / f"dm_{n}__{tag}.npy"), pts) - rho_ref for n in COUPLED}

    # chain horizontal (u = y), charge above (v = x - x_center); left half s, right half s+p
    mixed = diff["H6_sp"].T.copy()
    mid = NY // 2
    mixed[:, :mid] = diff["H6_s"].T[:, :mid]
    from run_fragment_curves import X0
    v_pc = X0 + r - center[0]

    gap, cb_w, right, side = 0.15, 0.15, 1.30, 0.05
    W = 8.5; map_w = W - side - gap - cb_w - right
    map_h = map_w * (V_LIM[1] - V_LIM[0]) / (U_LIM[1] - U_LIM[0]); H = map_h + 0.1
    fig = plt.figure(figsize=(W * cm, H * cm))
    ax = fig.add_axes([side / W, 0.05 / H, map_w / W, map_h / H])
    im = ax.imshow(mixed, origin="lower", extent=(gy[0], gy[-1], gx[0], gx[-1]), cmap="coolwarm",
                   vmin=-VMAX, vmax=VMAX)
    ax.set_xlim(*U_LIM); ax.set_ylim(*V_LIM); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(0.0, color="k", ls="--", lw=0.8, zorder=5)
    ax.plot([0.0], [v_pc], marker="+", ms=6, mew=1.8, color="white", zorder=10, clip_on=False)
    cax = fig.add_axes([(side + map_w + gap) / W, 0.05 / H, cb_w / W, map_h / H])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.1, 0.0, 0.1]); cb.set_label(r"$\Delta\rho(\mathbf{r})$")
    save(fig, f"fig5a_def2tzvp_density_{tag}")
    np.savez_compressed(FIG_DIR / f"fig5a_def2tzvp_density_{tag}.npz", u=gy, v=gx, drho_s=diff["H6_s"].T,
                        drho_sp=diff["H6_sp"].T, r_charge=r, v_charge=v_pc)
    print(f"r = {r:.4f} A: max|drho| s = {np.abs(diff['H6_s']).max():.3f}, s+p = {np.abs(diff['H6_sp']).max():.3f}")


def save(fig, name, **kw):
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=300, facecolor="white", **kw); plt.close(fig)
    print(f"saved {out}")


def main():
    fig5b()
    for r in DENSITY_R:
        fig5a(r)


if __name__ == "__main__":
    main()
