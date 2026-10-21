"""Small helpers for the water calculations (reference mu)."""
from pyscf import scf

from gcec import make_sigma_ao_diagonal  # noqa: F401  (re-exported for the scripts)


def mu_from_midgap(mol):
    """
    Reference chemical potential mu = (e_HOMO + e_LUMO)/2 from a canonical RHF.

    The RHF gap is used for all levels of theory (as in Tables S1-S4).

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Closed-shell molecule.

    Returns
    -------
    mu : float
        Midgap chemical potential in Ha.
    """
    mf = scf.RHF(mol)
    mf.kernel()
    nocc = mol.nelectron // 2
    return 0.5 * float(mf.mo_energy[nocc - 1] + mf.mo_energy[nocc])
