#!/usr/bin/env python3
"""
H6 grand-canonical levels as a function of the chemical potential mu
(paper Fig. 3b): eps = 0, eta = 0.1 Ha on the two outer H atoms, no point charge.

mu runs over 50 points from (HOMO - 0.02) to (LUMO + 0.02) Ha. As in the
published figure, HOMO and LUMO (mu range and dashed lines) come from a
canonical RHF calculation of H6, while the GC-SCF itself is RKS/PBE0. Each point starts from the
density of the previous one. For every mu the level energies e_k = Re Z_k,
broadenings gamma_k = -Im Z_k and grand-canonical occupations f_k (Appendix C)
of the effective Hamiltonian are stored.

Output: results/mu_sweep/H6_eps0_eta0.1.npz (+ .csv with Ne, Omega, converged)
Plot with plot_mu_sweep.py.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from pyscf import dft, scf

from gcec import make_sigma_ao_diagonal
from gcec.scf_loop import scf_gc
from run_fragment_curves import THIS_DIR, XC, GRID_LEVEL, SCF, build_mol, outer_h_aos, metadata

OUT = THIS_DIR / "results" / "mu_sweep" / "H6_eps0_eta0.1.npz"
XYZ, EPS, ETA = "ring_3uc.xyz", 0.0, 0.1
N_MU, MU_PAD = 50, 0.02


def main():
    mol = build_mol(XYZ)
    mf0 = scf.RHF(mol); mf0.verbose = 0          # reference HOMO/LUMO (see docstring)
    mf0.kernel()
    nocc = mol.nelectron // 2
    homo, lumo = float(mf0.mo_energy[nocc - 1]), float(mf0.mo_energy[nocc])
    mu_grid = np.linspace(homo - MU_PAD, lumo + MU_PAD, N_MU)

    Sigma_ao, _ = make_sigma_ao_diagonal(mol, idxs=outer_h_aos(mol), gamma_val=ETA, eps_val=EPS)
    e_all, g_all, f_all, rows, dm = [], [], [], [], None
    for mu in mu_grid:
        mf = dft.RKS(mol); mf.xc, mf.grids.level = XC, GRID_LEVEL
        out = scf_gc(mol, Sigma_ao, mu=float(mu), logger=None, verbose=False, mf=mf, dm_init=dm, **SCF)
        P, Ne, E, conv = np.real(out[0]), float(out[1]), float(np.real(out[4])), bool(out[5])
        e_all.append(np.asarray(out[6], float)); f_all.append(np.asarray(out[7], float))
        g_all.append(np.asarray(out[12], float))
        rows.append({"mu": mu, "Ne": Ne, "Omega": E - mu * Ne, "converged": conv})
        dm = P
        print(f"mu={mu:+.5f}  Ne={Ne:.5f}  conv={conv}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, mu_grid=mu_grid, mo_energy=np.array(e_all), mo_gamma=np.array(g_all),
                        mo_occ_gc=np.array(f_all), Ne=np.array([r["Ne"] for r in rows]),
                        converged=np.array([r["converged"] for r in rows]),
                        homo=homo, lumo=lumo, eps=EPS, eta=ETA)
    with OUT.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    OUT.with_suffix(".json").write_text(json.dumps(metadata(
        script="run_mu_sweep.py", xyz=XYZ, eps=EPS, eta=ETA, n_mu=N_MU, mu_padding=MU_PAD,
        reference_homo_lumo_from="canonical RHF", canonical_homo=homo, canonical_lumo=lumo,
        n_converged=int(sum(r["converged"] for r in rows))), indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
