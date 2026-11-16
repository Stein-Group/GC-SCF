#!/usr/bin/env python3
"""
Li40 broadening sweep (paper Fig. 6, SI Fig. S16).

For each coupling scheme, eta is increased over 15 values from 1e-5 to 0.5 Ha.
At every eta, mu is adjusted self-consistently so that the cluster stays
neutral (N_e = 120). Each point starts from the converged density of the
previous eta ("chain"); after a non-converged point the chain restarts from
the canonical density.

SCF settings: Anderson mixing, history 3,
max. 300 iterations, ||Delta P|| < 1e-5, |Delta Omega| < 1e-6.

Output: results/broadening/<scheme>/
    sweep.npz      eta, mu, Ne, E_tot, Omega, converged, mo_energy, mo_occ_gc, P
    sweep.csv      scalar columns
    metadata.json  code version, pySCF/Python versions, settings

Usage:
    python run_broadening_sweep.py                    # all three schemes
    python run_broadening_sweep.py --scheme center
    python run_broadening_sweep.py --scheme center --max-points 3   # quick test
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from common import (RESULTS_DIR, COUPLING_PRESETS, build_mol, make_mf, make_sigma,
                    run_gc, canonical_reference, run_metadata, save_json)

ETA_GRID = [
    1.0e-5, 2.0e-5, 5.0e-5,
    1.0e-4, 2.0e-4, 5.0e-4,
    1.0e-3, 2.0e-3, 5.0e-3,
    1.0e-2, 2.0e-2, 5.0e-2,
    1.0e-1, 2.0e-1, 5.0e-1,
]

SCF = dict(max_cycle=300, conv_tol=1e-5, conv_tol_omega=1e-6, diis_space=3, method="ANDERSON")


def sweep(scheme: str, max_points: int | None = None, verbose: bool = False):
    mol, _ = build_mol()
    mf = make_mf(mol)
    can = canonical_reference(mol, mf)
    target_ne = float(mol.nelectron)
    print(f"canonical: E = {float(can['e_tot']):+.10f} Ha, HOMO = {float(can['homo']):+.6f}, "
          f"LUMO = {float(can['lumo']):+.6f}, gap = {1e3*(float(can['lumo'])-float(can['homo'])):.3f} mHa")

    etas = ETA_GRID[:max_points] if max_points else ETA_GRID
    mu = float(can["mu_midgap"])
    dm_chain = np.array(can["dm"], copy=True)
    rows, Ps, mo_e, mo_occ = [], [], [], []

    for i, eta in enumerate(etas, 1):
        t0 = time.time()
        Sigma_ao, aos = make_sigma(mol, scheme, eta)
        res = run_gc(mol, mf, Sigma_ao, mu=mu, target_ne=target_ne, dm_init=dm_chain,
                     verbose=verbose, **SCF)
        mu = res["mu"]
        dm_chain = res["P"] if res["converged"] else np.array(can["dm"], copy=True)
        rows.append({"eta": eta, "mu": res["mu"], "Ne": res["Ne"], "E_tot": res["E_tot"],
                     "Omega": res["Omega"], "converged": res["converged"]})
        Ps.append(res["P"]); mo_e.append(res["mo_energy"]); mo_occ.append(res["mo_occ_gc"])
        print(f"{scheme:15s} {i:2d}/{len(etas)}  eta={eta:.1e}  mu={res['mu']:+.8f}  "
              f"Ne={res['Ne']:.8f}  Omega={res['Omega']:+.10f}  conv={res['converged']}  "
              f"({time.time()-t0:.0f} s)", flush=True)

    out = RESULTS_DIR / "broadening" / scheme
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "sweep.npz",
        eta=np.array([r["eta"] for r in rows]),
        mu=np.array([r["mu"] for r in rows]),
        Ne=np.array([r["Ne"] for r in rows]),
        E_tot=np.array([r["E_tot"] for r in rows]),
        Omega=np.array([r["Omega"] for r in rows]),
        converged=np.array([r["converged"] for r in rows]),
        mo_energy=np.array(mo_e), mo_occ_gc=np.array(mo_occ), P=np.array(Ps),
    )
    with (out / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    save_json(run_metadata(script="run_broadening_sweep.py", scheme=scheme,
                           coupled_atoms=COUPLING_PRESETS[scheme], n_coupled_aos=int(len(aos)),
                           eps=0.0, eta_grid=list(etas), target_ne=target_ne,
                           canonical_converged=bool(can["converged"]), **SCF),
              out / "metadata.json")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", choices=[*COUPLING_PRESETS, "all"], default="all")
    ap.add_argument("--max-points", type=int, default=None, help="only the first N eta values (test)")
    ap.add_argument("--verbose", action="store_true", help="print every SCF iteration")
    a = ap.parse_args()
    schemes = list(COUPLING_PRESETS) if a.scheme == "all" else [a.scheme]
    for s in schemes:
        sweep(s, a.max_points, a.verbose)


if __name__ == "__main__":
    main()
