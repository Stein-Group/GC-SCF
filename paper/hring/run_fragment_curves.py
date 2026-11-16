#!/usr/bin/env python3
"""
H-ring fragment curves at the published (optimized) coupling parameters
(paper Sec. IV.B, Fig. 3c for H6 and Fig. 4 for H6-H20; PBE0/MINAO).

For each fragment, the two outer H atoms (smallest and largest y) are coupled
to the reservoir with Sigma = eps - i*eta on their AOs (MINAO: one 1s
function each). mu is fixed at the value that makes the isolated fragment
neutral (from the optimization). A point charge q = +1 approaches along the
x-axis, at (x0 + r, 0, 0) with x0 = 4.701296 A and the 68 r values of the
H40 reference scan (r_grid.txt). Every point starts from the default SAD
guess (no warm start).

  Delta Omega(r) = Omega(q, r) - Omega(q = 0),   Omega = E_tot - mu * N_e

The coupling parameters were optimized beforehand (L-BFGS on the curve
objective against the H40 reference) and are not re-optimized here.

SCF settings: Anderson mixing (mixing 0.10, history 8),
max. 200 iterations, ||Delta P|| < 1e-5, |Delta Omega| < 1e-8; RKS/PBE0,
grid level 3.

Output: results/fragment_curves/<label>/curve.npz, curve.csv, metadata.json

Usage:
    python run_fragment_curves.py                 # all fragments
    python run_fragment_curves.py --fragment H6   # only H6 (Fig. 3c)
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, dft, qmmm

from gcec import make_sigma_ao_diagonal
from gcec.scf_loop import scf_gc

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results" / "fragment_curves"
HA_TO_KCAL = 627.5094740631

BASIS, XC, GRID_LEVEL = "minao", "pbe0", 3
X0 = 4.701296          # outermost atom plane of the ring (A)
POINT_CHARGE_Q = 1.0
SCF = dict(method="ANDERSON", max_cycle=200, conv_tol=1e-5, conv_tol_omega=1e-8,
           diis_space=8, anderson_mix=0.10)

# Published parameters (optimized for PBE0/MINAO against the H40 curve).
# label: (xyz file, eps [Ha], eta [Ha], mu [Ha]); mu from the neutrality condition.
FRAGMENTS = {
    "H6":        ("ring_3uc.xyz",  -0.06548206236300383, 0.20482076169506147, -0.2884005262008257),
    "H6_alt":    ("ring_3uc.xyz",  -0.06019334700853716, 0.3292233465731656,  -0.27672187270937254),
    "H8_a":      ("ring_4uc.xyz",  -0.05986069886294746, 0.19981679877763173, -0.2548590189263092),
    "H10":       ("ring_5uc.xyz",  -0.05171421231174585, 0.11842270849671054, -0.23599489204432736),
    "H12":       ("ring_6uc.xyz",  -0.058699264088564164, 0.1350863672353512, -0.2228429563836421),
    "H14":       ("ring_7uc.xyz",  -0.04024707935676431, 0.08738712127060468, -0.20640689385643426),
    "H16":       ("ring_8uc.xyz",  -0.045661371409268965, 0.08957864168571376, -0.1993074823452837),
    "H18":       ("ring_9uc.xyz",  -0.024405463066558584, 0.06022189602522503, -0.1862329236937191),
    "H20":       ("ring_10uc.xyz", -0.02571608392512921, 0.06413146596940882, -0.18051034525115808),
}


def read_r_grid() -> np.ndarray:
    return np.loadtxt(THIS_DIR / "r_grid.txt", comments="#")


def build_mol(xyz: str):
    lines = (THIS_DIR / "xyz" / xyz).read_text().splitlines()
    n = int(lines[0])
    atoms = [(f[0], (float(f[1]), float(f[2]), float(f[3])))
             for f in (l.split() for l in lines[2:2 + n])]
    mol = gto.Mole()
    mol.atom, mol.unit, mol.basis, mol.verbose = atoms, "Angstrom", BASIS, 0
    mol.build()
    return mol


def outer_h_aos(mol) -> np.ndarray:
    """All AOs of the two outermost H atoms along y (one per side)."""
    y = mol.atom_coords(unit="Ang")[:, 1]
    outer = {int(np.argmin(y)), int(np.argmax(y))}
    return np.array([i for i, lab in enumerate(mol.ao_labels(fmt=False)) if int(lab[0]) in outer], int)


def gc_point(mol, Sigma_ao, mu, x=None):
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level = XC, GRID_LEVEL
    if x is not None:
        mf = qmmm.mm_charge(mf, [(float(x), 0.0, 0.0)], [POINT_CHARGE_Q], unit="Ang")
    out = scf_gc(mol, Sigma_ao, mu=float(mu), logger=None, verbose=False, mf=mf, **SCF)
    E_tot, Ne, conv = float(np.real(out[4])), float(out[1]), bool(out[5])
    return {"E_tot": E_tot, "Ne": Ne, "Omega": E_tot - float(mu) * Ne, "converged": conv}


def metadata(**extra) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=THIS_DIR,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    import os
    return {"created": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
            "python": sys.version.split()[0], "pyscf": pyscf.__version__, "numpy": np.__version__,
            "host": platform.node(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "basis": BASIS, "xc": XC, "grid_level": GRID_LEVEL, "x0_ang": X0,
            "point_charge_q": POINT_CHARGE_Q, **SCF, **extra}


def run(label: str):
    xyz, eps, eta, mu = FRAGMENTS[label]
    mol = build_mol(xyz)
    aos = outer_h_aos(mol)
    Sigma_ao, _ = make_sigma_ao_diagonal(mol, idxs=aos, gamma_val=eta, eps_val=eps)
    r = read_r_grid()

    t0 = time.time()
    p0 = gc_point(mol, Sigma_ao, mu)
    print(f"{label}: q=0  Omega={p0['Omega']:+.10f}  Ne={p0['Ne']:.8f}  conv={p0['converged']}", flush=True)
    rows = []
    for ri in r:
        p = gc_point(mol, Sigma_ao, mu, x=X0 + ri)
        d = p["Omega"] - p0["Omega"]
        rows.append({"r_ang": ri, "x_ang": X0 + ri, "Omega": p["Omega"], "Ne": p["Ne"],
                     "DeltaOmega": d, "DeltaOmega_kcal": d * HA_TO_KCAL, "converged": p["converged"]})
    nconv = sum(r_["converged"] for r_ in rows)
    print(f"{label}: {nconv}/{len(rows)} points converged ({time.time()-t0:.0f} s)", flush=True)

    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "curve.npz", **{k: np.array([r_[k] for r_ in rows]) for k in rows[0]},
                        Omega_q0=np.array(p0["Omega"]), Ne_q0=np.array(p0["Ne"]),
                        q0_converged=np.array(p0["converged"]), mu=np.array(mu),
                        eps=np.array(eps), eta=np.array(eta), coupled_aos=aos)
    with (out / "curve.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "metadata.json").write_text(json.dumps(metadata(
        script="run_fragment_curves.py", fragment=label, xyz=xyz, eps=eps, eta=eta, mu=mu,
        coupled_aos=aos.tolist(), n_points=len(rows), n_converged=int(nconv),
        Omega_q0=p0["Omega"], Ne_q0=p0["Ne"]), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragment", choices=[*FRAGMENTS, "all"], default="all")
    a = ap.parse_args()
    for label in (FRAGMENTS if a.fragment == "all" else [a.fragment]):
        run(label)


if __name__ == "__main__":
    main()
