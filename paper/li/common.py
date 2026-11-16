"""
Shared settings and helpers for the Li40 calculations (paper Sec. IV.C,
Figs. 6-7, SI Section S4).

Level of theory (paper, Sec. III): RKS/PBE0, STO-3G, grid level 3.
Coupling: Sigma = eps - i*eta on the valence AOs (n >= 2) of selected Li atoms,
eps = 0. Three coupling topologies (SI Section S4.1): center, surface_center,
edges.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, dft

from gcec import make_sigma_ao_diagonal
from gcec.scf_loop import scf_gc

THIS_DIR = Path(__file__).resolve().parent
LI_XYZ = THIS_DIR / "li_bcc_110_big.xyz"
RESULTS_DIR = THIS_DIR / "results"

BASIS = "sto-3g"
XC = "pbe0"
GRID_LEVEL = 3

BOHR_ANG = 0.52917720859  # 1 bohr in Angstrom

# Atom indices (0-based, order of li_bcc_110_big.xyz) of the coupled Li atoms.
COUPLING_PRESETS = {
    "center": [19, 18, 5, 13],
    "surface_center": [0, 4, 22, 16, 17, 10, 11],
    "edges": [
        1, 2, 3, 6, 7, 8, 9,
        12, 14, 15,
        20, 21, 23, 24, 25,
        26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    ],
}

# Lowest valence principal quantum number per element (Li: 2s, 2p).
_MIN_VALENCE_N = {"H": 1, "He": 1, "Li": 2, "Be": 2}


def read_xyz(path: Path = LI_XYZ):
    """Symbols and coordinates (Angstrom) from an xyz file."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    symbols, coords = [], []
    for line in lines[2:2 + n]:
        f = line.split()
        symbols.append(f[0])
        coords.append([float(f[1]), float(f[2]), float(f[3])])
    return symbols, np.asarray(coords, dtype=float)


def build_mol(basis: str = BASIS):
    """Li40 cluster (neutral, closed shell)."""
    symbols, coords = read_xyz(LI_XYZ)
    mol = gto.Mole()
    mol.atom = [(s, tuple(r)) for s, r in zip(symbols, coords)]
    mol.unit = "Angstrom"
    mol.charge = 0
    mol.spin = 0
    mol.basis = basis
    mol.verbose = 0
    mol.build()
    return mol, coords


def make_mf(mol):
    """RKS/PBE0 mean-field object (also used for the canonical reference)."""
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = GRID_LEVEL
    mf.max_cycle = 200
    mf.conv_tol = 1.0e-9
    return mf


def valence_ao_indices(mol, atom_ids) -> np.ndarray:
    """AO indices of the valence shells (n >= n_val) of the given atoms."""
    atom_ids = {int(i) for i in atom_ids}
    idxs = [
        i for i, lab in enumerate(mol.ao_labels(fmt=False))
        if int(lab[0]) in atom_ids
        and int(lab[2][0]) >= _MIN_VALENCE_N.get(lab[1].capitalize(), 1)
    ]
    return np.asarray(idxs, dtype=int)


def make_sigma(mol, scheme: str, eta: float, eps: float = 0.0):
    """Diagonal self-energy on the valence AOs of the atoms of a coupling scheme."""
    aos = valence_ao_indices(mol, COUPLING_PRESETS[scheme])
    Sigma_ao, _ = make_sigma_ao_diagonal(mol, idxs=aos, gamma_val=float(eta), eps_val=float(eps))
    return Sigma_ao, aos


def run_gc(mol, mf, Sigma_ao, *, mu, target_ne=None, dm_init=None, V_ext=None,
           max_cycle=300, conv_tol=1e-5, conv_tol_omega=1e-6, diis_space=3,
           method="ANDERSON", verbose=False) -> dict:
    """
    One GC-SCF run with the public gcec kernel.

    target_ne given -> mu is adjusted in every iteration (neutrality),
    otherwise mu is fixed.
    """
    out = scf_gc(
        mol, Sigma_ao,
        mu=float(mu),
        target_ne=target_ne,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_omega=conv_tol_omega,
        diis_space=diis_space,
        method=method,
        logger=None,
        verbose=verbose,
        V_ext=V_ext,
        mf=mf,
        dm_init=dm_init,
    )
    P, Ne, f, F, E_tot, converged, mo_energy, mo_occ_gc, S, C_ao, mu_out, P_raw, mo_gamma = out
    E_tot = float(np.real(E_tot))
    mu_out = float(mu_out)
    return {
        "P": np.array(np.real(P), copy=True),
        "Ne": float(Ne),
        "E_tot": E_tot,
        "Omega": E_tot - mu_out * float(Ne),
        "converged": bool(converged),
        "mo_energy": np.asarray(np.real(mo_energy), float),
        "mo_occ_gc": np.asarray(mo_occ_gc, float),
        "mu": mu_out,
    }


def canonical_reference(mol, mf, cache: Path | None = None) -> dict:
    """Canonical (Sigma = 0) RKS reference; cached as npz."""
    cache = cache or (RESULTS_DIR / "canonical.npz")
    if cache.exists():
        c = np.load(cache)
        return {k: c[k] for k in c.files}
    e_tot = float(mf.kernel())
    nocc = mol.nelectron // 2
    homo, lumo = float(mf.mo_energy[nocc - 1]), float(mf.mo_energy[nocc])
    ref = {
        "dm": np.asarray(mf.make_rdm1()),
        "e_tot": np.asarray(e_tot),
        "mo_energy": np.asarray(mf.mo_energy),
        "mo_coeff": np.asarray(mf.mo_coeff),
        "mo_occ": np.asarray(mf.mo_occ),
        "homo": np.asarray(homo),
        "lumo": np.asarray(lumo),
        "mu_midgap": np.asarray(0.5 * (homo + lumo)),
        "converged": np.asarray(bool(mf.converged)),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **ref)
    return ref


def git_commit() -> str | None:
    """Commit of the GC-SCF checkout this script runs from (None if unavailable)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=THIS_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def run_metadata(**extra) -> dict:
    """Provenance written next to every result."""
    import os
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "host": platform.node(),
        "platform": platform.platform(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "basis": BASIS,
        "xc": XC,
        "grid_level": GRID_LEVEL,
    }
    meta.update(extra)
    return meta


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
