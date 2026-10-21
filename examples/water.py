#!/usr/bin/env python3
"""
Minimal GC-SCF example: water coupled to a wide-band reservoir.

Couples the valence AOs (O 2s/2p, H 1s) of water to a reservoir with
Sigma = -i*eta (eta = 0.1 Ha) at a fixed chemical potential mu, and runs
the GC-SCF at the RHF/STO-3G level.

Reproduces the EDIIS(Omega), SAD-guess entry of Table II of the paper:
Omega = -75.496810 Ha, N_e = 10.1389.

Run with:
    python examples/water.py
"""
from pyscf import gto, scf

from gcec import GCEC, SCFSettings, make_sigma_ao_diagonal


# ---- Molecule ----
mol = gto.M(
    atom="""
        O  0.000000  0.000000  0.000000
        H  0.000000 -0.757000  0.587000
        H  0.000000  0.757000  0.587000
    """,
    basis="sto-3g",
)

# ---- Chemical potential: HOMO/LUMO midgap of the isolated molecule ----
mf0 = scf.RHF(mol).run()
nocc = mol.nelectron // 2
mu = 0.5 * (mf0.mo_energy[nocc - 1] + mf0.mo_energy[nocc])

# ---- Self-energy on the valence AOs ----
idxs = [
    i for i, lab in enumerate(mol.ao_labels())
    if (" O " in lab and (" 2s" in lab or " 2p" in lab)) or (" H " in lab and " 1s" in lab)
]
Sigma_ao, _ = make_sigma_ao_diagonal(mol, idxs, gamma_val=0.1)

# ---- GC-SCF ----
settings = SCFSettings(theory="RHF", method="EDIIS_OMEGA", max_cycle=200, diis_space=12, verbose=False)
res = GCEC(mol, Sigma_ao=Sigma_ao, mu=mu, settings=settings).run_scf()

print(f"mu (midgap)          : {mu:.6f} Ha")
print(f"canonical E (Sigma=0): {mf0.e_tot:.6f} Ha")
print(f"GC-SCF converged     : {res['converged']}")
print(f"Omega                : {res['Omega']:.6f} Ha")
print(f"N_e                  : {res['Ne']:.4f}")
print(f"GC occupations       : {res['mo_occ_gc'].round(3)}")
