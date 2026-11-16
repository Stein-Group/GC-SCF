#!/usr/bin/env python3
"""
Canonical (uncoupled, N_e = 120) reference for the Li40 point-charge scan.

Standard RKS/PBE0/STO-3G with the point charge added to H_core, no
self-energy. Same charge (q = +0.4 e), approach atom and 33 distances as
run_charge_scan.py. Each distance starts from the density of the previous one.
max_cycle = 200; the pySCF convergence flag is stored for every distance.

Output: results/charge_scan/canonical_ref.npz (+ .csv, metadata.json)
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from pyscf import dft

from common import RESULTS_DIR, XC, GRID_LEVEL, build_mol, run_metadata, save_json
from run_charge_scan import (POINT_CHARGE_Q, APPROACH_ABOVE_ATOM, point_charge_v_ext,
                             nuclear_point_charge_energy)

from run_charge_scan import DISTANCES_ANG as _D
DISTANCES_ANG = sorted(_D)   # same 33-point grid as the coupled scan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-cycle", type=int, default=200)
    a = ap.parse_args()

    mol, coords = build_mol()
    mf0 = dft.RKS(mol); mf0.xc = XC; mf0.grids.level = GRID_LEVEL; mf0.grids.build()
    mf0.max_cycle = a.max_cycle
    mf0.kernel()
    dm_prev = mf0.make_rdm1()
    ref_pos = coords[APPROACH_ABOVE_ATOM].copy()

    recs = []
    for d in DISTANCES_ANG:
        pos = ref_pos.copy(); pos[2] -= d
        V_ext = point_charge_v_ext(mol, POINT_CHARGE_Q, pos)
        E_nuc_ext = nuclear_point_charge_energy(mol, POINT_CHARGE_Q, pos)
        mf = dft.RKS(mol); mf.xc = XC; mf.grids = mf0.grids; mf.max_cycle = a.max_cycle
        h1e = mf.get_hcore()
        mf.get_hcore = lambda *args, _h=h1e + V_ext, **kw: _h
        mf.kernel(dm_prev)
        dm_prev = mf.make_rdm1()
        recs.append({"distance_ang": d, "E_tot": float(mf.e_tot), "E_nuc_ext": E_nuc_ext,
                     "E_tot_full": float(mf.e_tot) + E_nuc_ext, "converged": bool(mf.converged)})
        print(f"d={d:4.1f} A  E_full={recs[-1]['E_tot_full']:+.10f}  conv={mf.converged}", flush=True)

    out = RESULTS_DIR / "charge_scan"
    out.mkdir(parents=True, exist_ok=True)
    keys = list(recs[0])
    np.savez_compressed(out / "canonical_ref.npz", **{k: np.array([r[k] for r in recs]) for k in keys},
                        point_charge_q=np.array(POINT_CHARGE_Q), approach_atom=np.array(APPROACH_ABOVE_ATOM))
    with (out / "canonical_ref.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(recs)
    save_json(run_metadata(script="run_charge_scan_canonical.py", max_cycle=a.max_cycle,
                           point_charge_q=POINT_CHARGE_Q, approach_atom=APPROACH_ABOVE_ATOM,
                           distances_ang=DISTANCES_ANG), out / "canonical_ref_metadata.json")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
