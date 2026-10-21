#!/usr/bin/env python3
"""
GC-SCF example: H6 fragment of the H40 ring with a positive point charge.

The two outer H atoms of the H6 fragment are coupled to a reservoir with
Sigma = eps - i*eta, using the optimized PBE0/MINAO parameters of Fig. 3c
of the paper (eps = -0.065482 Ha, eta = 0.204821 Ha, mu = -0.2884 Ha from
the neutrality condition). A point charge q = +1 approaches the fragment
along the x-axis; r is measured from the outermost atom plane.

The script prints N_e of the fragment without the point charge (~6, i.e.
neutral) and the adsorption energy Delta Omega = Omega(r) - Omega(q = 0)
at the minimum of the H40 reference curve, r = 1.22 A.

Run with:
    python examples/hydrogen_ring.py
"""
import numpy as np
from pyscf import gto, dft, qmmm

from gcec import GCEC, SCFSettings, make_sigma_ao_diagonal

HARTREE_TO_KCAL = 627.509474

# ---- H6 fragment (six atoms of the H40 ring, bond length 0.74 A) ----
atoms = """
H 4.356862 -1.804671 0.000000
H 4.585534 -1.100889 0.000000
H 4.701296 -0.370000 0.000000
H 4.701296  0.370000 0.000000
H 4.585534  1.100889 0.000000
H 4.356862  1.804671 0.000000
"""
mol = gto.M(atom=atoms, basis="minao")
x0 = mol.atom_coords(unit="Ang")[:, 0].max()   # outermost atom plane

# ---- Self-energy on the 1s functions of the two outer H atoms ----
EPS, ETA, MU = -0.065482, 0.204821, -0.2884
Sigma_ao, _ = make_sigma_ao_diagonal(mol, idxs=[0, 5], gamma_val=ETA, eps_val=EPS)

settings = SCFSettings(method="ANDERSON", max_cycle=200, conv_tol=1e-5, verbose=False)


def gc_point(r=None, q=1.0, dm_init=None):
    """GC-SCF of the fragment, with a point charge q at distance r (A) or without."""
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 3
    if r is not None:
        mf = qmmm.mm_charge(mf, [(x0 + r, 0.0, 0.0)], [q], unit="Ang")
    res = GCEC(mol, Sigma_ao=Sigma_ao, mu=MU, mf=mf, settings=settings).run_scf(dm_init=dm_init)
    if not res["converged"]:
        raise RuntimeError(f"GC-SCF not converged (r = {r})")
    return res


ref = gc_point()
ads = gc_point(r=1.22, dm_init=ref["P"])
dOmega = (ads["Omega"] - ref["Omega"]) * HARTREE_TO_KCAL

print(f"N_e without point charge : {ref['Ne']:.4f}  (neutral: 6)")
print(f"N_e at r = 1.22 A        : {ads['Ne']:.4f}")
print(f"Delta Omega(r = 1.22 A)  : {dOmega:.2f} kcal/mol")
