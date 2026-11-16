#!/usr/bin/env python3
"""
Li40 point-charge scan (paper Sec. IV.C, Fig. 8).

1. Neutral reference: center coupling, eta = 2 mHa, mu solved for N_e = 120
   (eta ramp 1e-3 -> 2e-3 Ha, starting from the eta = 2e-3 density of the
   broadening sweep if available, otherwise from the canonical density).
2. Scan: a point charge q approaches along -z above atom 16 (top surface,
   z = 12 A) from d = 10 A to 1 A. mu is kept FIXED at the neutral value, so
   charge exchange with the reservoir shows up as Delta N_e. Each distance
   starts from the converged density of the previous (farther) one.

Settings (paper Fig. 8): q = +0.4 e, eta = 2e-3 Ha, approach atom 16, d = 1.0 ... 8.0 A in 0.25 A steps
plus 8.5 ... 10.0 A (33 points), STO-3G, Anderson, history 4,
||Delta P|| < 1e-5, |Delta Omega| < 1e-6. Energies in the figure are relative
to d = 10 A.

Output: results/charge_scan/center/charge_scan.npz (+ .csv, metadata.json)
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from common import (RESULTS_DIR, COUPLING_PRESETS, BOHR_ANG, build_mol, make_mf, make_sigma,
                    run_gc, canonical_reference, run_metadata, save_json)

SCHEME = "center"
ETA = 2.0e-3
PRERAMP_ETAS = [1.0e-3, 2.0e-3]
POINT_CHARGE_Q = 0.4
APPROACH_ABOVE_ATOM = 16
# 1.0 ... 8.0 A in 0.25 A steps, then 8.5 ... 10.0 A (33 points), scanned far -> close
DISTANCES_ANG = ([round(1.0 + 0.25 * k, 2) for k in range(29)] + [8.5, 9.0, 9.5, 10.0])[::-1]

SCF = dict(max_cycle=300, conv_tol=1e-5, conv_tol_omega=1e-6, diis_space=4, method="ANDERSON")


def point_charge_v_ext(mol, q, pos_ang):
    """AO matrix of the electron-point-charge attraction (added to H_core)."""
    with mol.with_rinv_origin(np.asarray(pos_ang) / BOHR_ANG):
        v = mol.intor("int1e_rinv")
    return -q * v


def nuclear_point_charge_energy(mol, q, pos_ang):
    """Repulsion between the cluster nuclei and the point charge (Ha)."""
    R = mol.atom_coords()
    Z = np.array([mol.atom_charge(i) for i in range(mol.natm)], float)
    d = np.linalg.norm(R - (np.asarray(pos_ang) / BOHR_ANG)[None, :], axis=1)
    return float(q * np.sum(Z / d))


def mulliken_charges(P, S, mol):
    PS = P @ S
    pop = np.zeros(mol.natm)
    for i, lab in enumerate(mol.ao_labels(fmt=False)):
        pop[int(lab[0])] += PS[i, i]
    Z = np.array([mol.atom_charge(i) for i in range(mol.natm)], float)
    return Z - pop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-points", type=int, default=None, help="only the first N distances (test)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    mol, coords = build_mol()
    mf = make_mf(mol)
    can = canonical_reference(mol, mf)
    S = mol.intor("int1e_ovlp")
    target_ne = float(mol.nelectron)

    # --- 1. neutral reference -------------------------------------------------
    dm, mu = np.array(can["dm"], copy=True), float(can["mu_midgap"])
    sweep_file = RESULTS_DIR / "broadening" / SCHEME / "sweep.npz"
    if sweep_file.exists():
        sw = np.load(sweep_file)
        k = int(np.argmin(np.abs(sw["eta"] - ETA)))
        if sw["converged"][k]:
            dm, mu = np.array(sw["P"][k]), float(sw["mu"][k])
            print(f"start from broadening sweep at eta = {sw['eta'][k]:.1e}")
    for eta in PRERAMP_ETAS:
        Sigma_ao, _ = make_sigma(mol, SCHEME, eta)
        res = run_gc(mol, mf, Sigma_ao, mu=mu, target_ne=target_ne, dm_init=dm, verbose=a.verbose, **SCF)
        mu = res["mu"]
        dm = res["P"] if res["converged"] else np.array(can["dm"], copy=True)
        print(f"preramp eta={eta:.1e}  mu={mu:+.10f}  Ne={res['Ne']:.8f}  conv={res['converged']}")
    mu_neutral, dm_ref = mu, dm
    q_ref = mulliken_charges(dm_ref, S, mol)
    print(f"fixed mu for the scan: {mu_neutral:+.10f} Ha")

    # --- 2. distance scan at fixed mu -----------------------------------------
    Sigma_ao, aos = make_sigma(mol, SCHEME, ETA)
    ref_pos = coords[APPROACH_ABOVE_ATOM].copy()
    dists = DISTANCES_ANG[:a.max_points] if a.max_points else DISTANCES_ANG
    dm_chain, recs = dm_ref, []
    for d in dists:
        t0 = time.time()
        pos = ref_pos.copy(); pos[2] -= d
        V_ext = point_charge_v_ext(mol, POINT_CHARGE_Q, pos)
        E_nuc_ext = nuclear_point_charge_energy(mol, POINT_CHARGE_Q, pos)
        res = run_gc(mol, mf, Sigma_ao, mu=mu_neutral, target_ne=None, dm_init=dm_chain,
                     V_ext=V_ext, verbose=a.verbose, **SCF)
        if res["converged"]:
            dm_chain = res["P"]
        q = mulliken_charges(res["P"], S, mol)
        recs.append({"distance_ang": d, "charge_z_ang": float(pos[2]), "Ne": res["Ne"],
                     "dNe": res["Ne"] - target_ne, "E_tot": res["E_tot"],
                     "E_nuc_ext": E_nuc_ext, "E_tot_full": res["E_tot"] + E_nuc_ext,
                     "Omega": res["Omega"], "converged": res["converged"],
                     "mulliken_q": q, "mulliken_dq": q - q_ref})
        print(f"d={d:4.1f} A  Ne={res['Ne']:.8f}  dNe={res['Ne']-target_ne:+.6f}  "
              f"E_full={res['E_tot']+E_nuc_ext:+.10f}  conv={res['converged']}  ({time.time()-t0:.0f} s)",
              flush=True)

    recs.sort(key=lambda r: r["distance_ang"])
    out = RESULTS_DIR / "charge_scan" / SCHEME
    out.mkdir(parents=True, exist_ok=True)
    scal = ["distance_ang", "charge_z_ang", "Ne", "dNe", "E_tot", "E_nuc_ext", "E_tot_full", "Omega", "converged"]
    np.savez_compressed(
        out / "charge_scan.npz",
        **{k: np.array([r[k] for r in recs]) for k in scal},
        mulliken_q=np.array([r["mulliken_q"] for r in recs]),
        mulliken_dq=np.array([r["mulliken_dq"] for r in recs]),
        mu_neutral=np.array(mu_neutral), dm_ref=dm_ref, mulliken_q_ref=q_ref,
        gamma=np.array(ETA), point_charge_q=np.array(POINT_CHARGE_Q),
        approach_atom=np.array(APPROACH_ABOVE_ATOM), coupled_atoms=np.array(COUPLING_PRESETS[SCHEME]),
    )
    with (out / "charge_scan.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=scal); w.writeheader()
        w.writerows([{k: r[k] for k in scal} for r in recs])
    save_json(run_metadata(script="run_charge_scan.py", scheme=SCHEME, eta=ETA,
                           preramp_etas=PRERAMP_ETAS, point_charge_q=POINT_CHARGE_Q,
                           approach_atom=APPROACH_ABOVE_ATOM, distances_ang=list(dists),
                           mu_neutral=mu_neutral, n_coupled_aos=int(len(aos)), **SCF),
              out / "metadata.json")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
