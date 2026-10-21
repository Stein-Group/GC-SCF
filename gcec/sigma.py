"""Self-energy construction in the AO basis."""
import numpy as np


def make_sigma_ao_diagonal(mol, idxs, gamma_val, eps_val=0.0):
    """
    Diagonal AO self-energy Sigma = diag(eps) - i diag(gamma) on selected AOs.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule.
    idxs : sequence of int
        AO indices coupled to the reservoir.
    gamma_val : float
        Broadening eta in Ha.
    eps_val : float, optional
        Real shift epsilon in Ha.

    Returns
    -------
    Sigma_ao : (nao, nao) complex ndarray
        Self-energy in the AO basis.
    idxs : ndarray
        Coupled AO indices.
    """
    nao = mol.nao_nr()
    idxs = np.asarray(idxs, dtype=int)
    if idxs.size == 0 or idxs.min() < 0 or idxs.max() >= nao:
        raise ValueError(f"Invalid AO indices for nao={nao}: {idxs}")
    diag = np.zeros(nao, dtype=complex)
    diag[idxs] = float(eps_val) - 1j * float(gamma_val)
    return np.diag(diag), idxs
