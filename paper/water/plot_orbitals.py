#!/usr/bin/env python3
"""
Plot the grand-canonical vs. canonical MO energies of water (Fig. 2b).

Reads one converged GC calculation from results/calculations/ (run
run_all_calculations.py first), recomputes the canonical (Sigma = 0) MOs,
and draws the GC levels with a schematic Lorentzian broadening (width from
gamma_k = -Im Z_kk, Appendix C), the GC occupations and mu.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pyscf import gto, scf, dft


# ============================================================
# User settings
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = THIS_DIR / "results" / "calculations"
OUTPUT_DIR = THIS_DIR / "results" / "figures" / "orbitals"

BASIS = "sto-3g"          # minao, sto-3g
THEORY_NAME = "rhf"       # rhf, rks_pbe
GAMMA = 1.0e-1
METHOD = "EDIIS_OMEGA"    # CDIIS, ANDERSON, EDIIS_E, EDIIS_OMEGA, HYBRID
START = "sad"             # sad, canonical_same_theory

N_SHOW = 6                # number of orbitals shown
SKIP_CORE = 1             # skip the O 1s orbital

FIG_CM = 5.65


# ============================================================
# Plot defaults
# ============================================================

cm = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "font.size": 8.5,
    "figure.dpi": 300,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# ============================================================
# Helpers
# ============================================================

def gamma_tag(gamma: float) -> str:
    return f"{gamma:.0e}".replace("+", "").replace(".", "p")


def calc_id() -> str:
    return (
        f"water__{THEORY_NAME}"
        f"__basis-{BASIS}"
        f"__gamma-{gamma_tag(GAMMA)}"
        f"__method-{METHOD.lower()}"
        f"__start-{START}"
    )


def read_xyz_atoms(path: Path) -> str:
    lines = path.read_text().splitlines()
    natoms = int(lines[0])
    return "\n".join(lines[2:2 + natoms])


def build_mol_from_config(cfg: dict):
    geom = cfg["system"]["geometry"]
    xyz_path = THIS_DIR / geom["file"]   # relative to paper/water/

    mol = gto.Mole()
    mol.atom = read_xyz_atoms(xyz_path)
    mol.unit = geom.get("units", "angstrom")
    mol.basis = cfg["system"]["basis"]
    mol.charge = int(cfg["system"]["charge"])
    mol.spin = int(cfg["system"]["spin"])
    mol.build()
    return mol


def run_canonical_mos(mol, cfg: dict):
    """Canonical (Sigma = 0) MO energies at the same level of theory."""
    theory = cfg["electronic_structure"]["theory"].upper()
    xc = cfg["electronic_structure"]["xc"]

    if theory == "RHF":
        mf = scf.RHF(mol)
    elif theory == "RKS":
        mf = dft.RKS(mol)
        mf.xc = xc
    else:
        raise ValueError(f"Unknown theory: {theory}")

    mf.kernel()
    if not mf.converged:
        print("WARNING: canonical reference did not converge")

    return np.asarray(mf.mo_energy, float)


def select_lowest_orbitals_with_idx(mo_energy, n_show=8, skip=0):
    mo_energy = np.asarray(mo_energy, float)
    order = np.argsort(mo_energy)
    sel = order[skip: skip + n_show]
    return sel, mo_energy[sel]


def sci_eta(val: float) -> str:
    if val == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10**exp)
    if np.isclose(mant, 1.0):
        return rf"10^{{{exp}}}"
    return rf"{mant:.2f}\times 10^{{{exp}}}"


def blob_width(Egrid, E0, gamma, *, max_width=0.35, gamma_ref=None, floor=1e-12):
    """Half-width of the schematic Lorentzian blob along x."""
    g = max(float(gamma), floor)
    shape = (g * g) / ((Egrid - E0) ** 2 + g * g)

    if gamma_ref is None:
        s = 1.0
    else:
        gref = max(float(gamma_ref), floor)
        s = min(g / gref, 1.0)

    return (max_width * s) * shape


# ============================================================
# Plot
# ============================================================

def plot_orbitals_with_canonical(*, eps_gc, eps_can, mo_occ_gc, mo_gamma,
                                 ne, mu_ref, gamma, out_png):
    sel, eps_gc_win = select_lowest_orbitals_with_idx(eps_gc, n_show=N_SHOW, skip=SKIP_CORE)
    _, eps_can_win = select_lowest_orbitals_with_idx(eps_can, n_show=N_SHOW, skip=SKIP_CORE)

    occ_win = np.clip(np.asarray(mo_occ_gc, float)[sel], 0.0, 2.0)

    # level broadenings gamma_k = -Im Z_kk (Appendix C), aligned with eps_gc
    gamma_win = np.asarray(mo_gamma, float)[sel]
    gamma_ref = float(np.max(gamma_win)) if np.any(np.isfinite(gamma_win)) else float(gamma)

    fig_w = FIG_CM * cm
    fig, ax = plt.subplots(figsize=(fig_w, fig_w), dpi=300)
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.14, top=0.92)

    e_min = float(min(np.min(eps_gc_win), np.min(eps_can_win)))
    e_max = float(max(np.max(eps_gc_win), np.max(eps_can_win)))
    pad = 0.20 * max(1e-6, e_max - e_min)
    y_lo, y_hi = e_min - pad, e_max + pad
    y_span = y_hi - y_lo

    ax.set_xlim(-0.7, N_SHOW - 0.3)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Orbital energy (Ha)")
    ax.set_xticks(range(N_SHOW))
    ax.set_xticklabels(["" for _ in range(N_SHOW)])
    ax.grid(True, ls=":", alpha=0.30)

    # chemical potential
    ax.axhline(mu_ref, ls="--", lw=1.0, color="k", alpha=0.7)

    Egrid = np.linspace(y_lo, y_hi, 800)
    stick_half = 0.18

    for k, (E_gc, E_can, occ, gam) in enumerate(zip(eps_gc_win, eps_can_win, occ_win, gamma_win)):
        # GC Lorentzian blob
        wblob = blob_width(Egrid, E_gc, gamma=float(gam), max_width=0.45, gamma_ref=gamma_ref)
        ax.fill_betweenx(Egrid, k - wblob, k + wblob, facecolor="0.2",
                         edgecolor="none", alpha=0.20, zorder=2)

        # GC level (black)
        ax.plot([k - stick_half, k + stick_half], [E_gc, E_gc], color="k", lw=2.0,
                solid_capstyle="butt", zorder=4, label="GC" if k == 0 else None)

        # canonical level (red)
        ax.plot([k - stick_half, k + stick_half], [E_can, E_can], color="red", lw=1.2,
                solid_capstyle="butt", zorder=5, label="canonical" if k == 0 else None)

        ax.text(k, E_gc + 0.035 * y_span, f"{occ:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="k", zorder=6)

    ax.text(
        0.95, 0.03,
        "\n".join([rf"$N_e = {ne:.2f}$", rf"$\eta = {sci_eta(gamma)}\ \mathrm{{Ha}}$"]),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="black",
                  linewidth=0.8, alpha=0.97),
        zorder=10,
    )

    ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=1.0,
              fontsize=7.5, borderpad=0.3, handlelength=1.8)

    fig.savefig(out_png, dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calc_dir = RESULTS_ROOT / calc_id()
    if not calc_dir.exists():
        raise FileNotFoundError(f"Missing calculation directory:\n{calc_dir}")

    cfg = yaml.safe_load((calc_dir / "resolved_config.yaml").read_text())
    summary = json.loads((calc_dir / "summary.json").read_text())
    data = np.load(calc_dir / "result.npz")

    mol = build_mol_from_config(cfg)
    eps_can = run_canonical_mos(mol, cfg)

    out_png = OUTPUT_DIR / f"water_orbitals__{calc_id().removeprefix('water__')}.png"

    plot_orbitals_with_canonical(
        eps_gc=np.asarray(data["mo_energy"], float),
        eps_can=eps_can,
        mo_occ_gc=np.asarray(data["mo_occ_gc"], float),
        mo_gamma=np.asarray(data["mo_gamma"], float),
        ne=float(summary["Ne"]),
        mu_ref=float(summary["mu_ref"]),
        gamma=float(summary["gamma"]),
        out_png=out_png,
    )
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
