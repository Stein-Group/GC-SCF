#!/usr/bin/env python3
"""
Canonical reference curves for the H-ring study (paper Sec. IV.B, Figs. 3c and 4).

Plain pySCF (no grand-canonical coupling), same setup as run_fragment_curves.py:
RKS/PBE0, MINAO, grid level 3, point charge q = +1 at (x0 + r, 0, 0) with
x0 = 4.701296 A and the 68 r values in r_grid.txt.

  Delta E(r) = E_tot(q, r) - E_tot(q = 0)

Systems:
  H40          full ring (xyz/ring_full.xyz), the reference the fragments are fitted to
  H38          H40 without the two atoms opposite the point charge (xyz/ring_h38.xyz)
  H6 ... H20   the isolated fragments without coupling (xyz/ring_<n>uc.xyz)

Each point starts from the density of the previous point; if that does not
converge, the point is repeated from the SAD guess with damping.

Output: results/reference_curves/<label>/curve.npz, curve.csv, metadata.json

Usage:
    python run_reference_curves.py               # all systems
    python run_reference_curves.py --system H40  # only H40
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
from pyscf import dft, qmmm

from run_fragment_curves import (THIS_DIR, HA_TO_KCAL, XC, GRID_LEVEL, X0,
                                 POINT_CHARGE_Q, read_r_grid, build_mol, metadata)

RESULTS_DIR = THIS_DIR / "results" / "reference_curves"
MAX_CYCLE = 200
SYSTEMS = {"H40": "ring_full.xyz", "H38": "ring_h38.xyz", **{f"H{2 * n}": f"ring_{n}uc.xyz" for n in range(3, 11)}}


def scf_point(mol, x=None, dm0=None):
    def run(dm, damp):
        mf = dft.RKS(mol)
        mf.xc, mf.grids.level, mf.max_cycle, mf.verbose = XC, GRID_LEVEL, MAX_CYCLE, 0
        mf.damp = damp
        if x is not None:
            mf = qmmm.mm_charge(mf, [(float(x), 0.0, 0.0)], [POINT_CHARGE_Q], unit="Ang")
        mf.kernel(dm0=dm)
        return mf

    mf = run(dm0, 0.0)
    if not mf.converged:                       # fallback: SAD guess with damping
        mf = run(None, 0.5)
    return mf


def run(label: str):
    xyz = SYSTEMS[label]
    mol = build_mol(xyz)
    r = read_r_grid()

    t0 = time.time()
    mf0 = scf_point(mol)
    E0 = float(mf0.e_tot)
    print(f"{label}: q=0  E={E0:+.10f}  conv={mf0.converged}", flush=True)

    rows, dm = [], mf0.make_rdm1()
    for ri in r:
        mf = scf_point(mol, x=X0 + ri, dm0=dm)
        if mf.converged:
            dm = mf.make_rdm1()
        d = float(mf.e_tot) - E0
        rows.append({"r_ang": ri, "x_ang": X0 + ri, "E_tot": float(mf.e_tot),
                     "DeltaE": d, "DeltaE_kcal": d * HA_TO_KCAL, "converged": bool(mf.converged)})
    nconv = sum(r_["converged"] for r_ in rows)
    print(f"{label}: {nconv}/{len(rows)} points converged ({time.time()-t0:.0f} s)", flush=True)

    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "curve.npz", **{k: np.array([r_[k] for r_ in rows]) for k in rows[0]},
                        E_q0=np.array(E0), q0_converged=np.array(mf0.converged),
                        mo_energy_q0=mf0.mo_energy, mo_occ_q0=mf0.mo_occ)
    with (out / "curve.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = metadata(script="run_reference_curves.py", system=label, xyz=xyz, method="canonical RKS",
                  max_cycle=MAX_CYCLE, n_points=len(rows), n_converged=int(nconv), E_q0=E0)
    for k in ("conv_tol", "conv_tol_omega", "diis_space", "anderson_mix"):
        md.pop(k, None)                        # GC-SCF settings, not used here
    (out / "metadata.json").write_text(json.dumps(md, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=[*SYSTEMS, "all"], default="all")
    a = ap.parse_args()
    for label in (SYSTEMS if a.system == "all" else [a.system]):
        run(label)


if __name__ == "__main__":
    main()
