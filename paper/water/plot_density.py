#!/usr/bin/env python3
"""
Plot the AO density-matrix change of water (Fig. 2c).

dP = P(eta = 1e-1) - P(eta_ref) in the AO basis, read from
results/calculations/ (run run_all_calculations.py first). The log density
vanishes at eta = 0, so the smallest converged broadening, eta_ref = 1e-6 Ha,
serves as the canonical reference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pyscf import gto
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle


# ============================================================
# User settings
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = THIS_DIR / "results" / "calculations"
OUTPUT_DIR = THIS_DIR / "results" / "figures" / "density_matrix"

BASIS = "sto-3g"          # minao, sto-3g
THEORY_NAME = "rhf"       # rhf, rks_pbe
METHOD = "EDIIS_OMEGA"    # CDIIS, ANDERSON, EDIIS_E, EDIIS_OMEGA, HYBRID
START = "sad"             # sad, canonical_same_theory

GAMMA_REF = 1.0e-6
GAMMA_TARGET = 1.0e-1

FIG_CM = 5.67


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


def calc_id(gamma: float) -> str:
    return (
        f"water__{THEORY_NAME}"
        f"__basis-{BASIS}"
        f"__gamma-{gamma_tag(gamma)}"
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


def load_calculation(gamma: float):
    cid = calc_id(gamma)
    calc_dir = RESULTS_ROOT / cid
    if not calc_dir.exists():
        raise FileNotFoundError(f"Missing calculation directory:\n{calc_dir}")

    return {
        "id": cid,
        "data": np.load(calc_dir / "result.npz"),
        "config": yaml.safe_load((calc_dir / "resolved_config.yaml").read_text()),
    }


def pretty_ao_labels(mol):
    """AO labels like 'O 2p$_x$' for the axis ticks."""
    labels = []
    for lab in mol.ao_labels():
        parts = lab.split()
        orb = (parts[2].replace("px", r"p$_x$")
                       .replace("py", r"p$_y$")
                       .replace("pz", r"p$_z$"))
        labels.append(f"{parts[1]} {orb}")
    return labels


# ============================================================
# Plot
# ============================================================

def plot_dm_heatmap(*, dP, mol, sigma_idxs, out_png):
    vmax = max(float(np.max(np.abs(dP))), 1.0e-12)

    fig_w = FIG_CM * cm
    fig, ax = plt.subplots(figsize=(fig_w, fig_w), dpi=300)
    fig.subplots_adjust(left=0.20, right=0.82, bottom=0.16, top=0.88)

    im = ax.imshow(dP, origin="upper", interpolation="nearest",
                   vmin=-vmax, vmax=vmax, cmap="coolwarm")

    ax.set_title(r"$\Delta\mathbf{P}(\eta)$", pad=3.0)
    ax.set_aspect("equal", adjustable="box")

    nao = dP.shape[0]
    ticks = np.arange(nao)
    labels = pretty_ao_labels(mol)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="both", labelsize=6.5)

    # box around the AO block that carries the self-energy
    idxs = sorted(set(int(i) for i in sigma_idxs if 0 <= int(i) < nao))
    if idxs:
        i0, i1 = idxs[0], idxs[-1]
        ax.add_patch(Rectangle((i0 - 0.5, i0 - 0.5), i1 - i0 + 1, i1 - i0 + 1,
                               fill=False, lw=1.2, ec="k", zorder=6))
        ax.text(0.18, 0.03, r"$\Sigma$ applied here", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
                zorder=7)

    cax = make_axes_locatable(ax).append_axes("right", size="6.0%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_png, dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = load_calculation(GAMMA_REF)
    target = load_calculation(GAMMA_TARGET)
    mol = build_mol_from_config(ref["config"])

    dP = np.asarray(np.real_if_close(target["data"]["P"] - ref["data"]["P"], tol=1e-9), float)

    print(f"Reference : {ref['id']}")
    print(f"Target    : {target['id']}")
    print(f"||dP||_F  : {np.linalg.norm(dP):.6e}")
    print(f"max|dP|   : {np.max(np.abs(dP)):.6e}")

    out_png = (
        OUTPUT_DIR
        / f"water_dP_aobasis__{THEORY_NAME}__basis-{BASIS}"
        f"__gamma-{gamma_tag(GAMMA_TARGET)}__ref-{gamma_tag(GAMMA_REF)}"
        f"__method-{METHOD.lower()}__start-{START}.png"
    )

    plot_dm_heatmap(dP=dP, mol=mol, sigma_idxs=target["data"]["sigma_idxs"], out_png=out_png)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
