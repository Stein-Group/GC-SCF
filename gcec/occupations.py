import numpy as np
from scipy.linalg import logm

def f_gc(Z, mu, eps=1e-10):
    """
    Grand-canonical occupations per spin for complex eigenvalues.

    For Z = E - iG, f = 1/2 + arctan((mu - E)/G)/pi, i.e. f_k/2 with f_k
    from Appendix C. Levels with G < eps get step-function occupations.

    Parameters
    ----------
    Z : (n,) complex ndarray
        Eigenvalues of Heff.
    mu : float
        Chemical potential.
    eps : float, optional
        Broadening below which a level is treated as sharp.

    Returns
    -------
    f : (n,) ndarray
        Occupations per spin, between 0 and 1.
    """  
    if mu is None:
        raise ValueError("mu must be provided for grand-canonical occupations.")
    Z = np.asarray(Z)
    E = Z.real
    G = np.abs(Z.imag)

    f = np.empty_like(E, dtype=float)
    step = G < eps
    f[step] = (E[step] < mu).astype(float)

    mask = ~step
    if np.any(mask):
        f[mask] = 0.5 - (1.0/np.pi) * np.arctan((E[mask] - mu)/G[mask])
    return f


def build_density_gc(Heff_orth, X, S, mu):
    """
    Grand-canonical density matrix from the logarithmic expression.

    P = (i/pi) [log(Heff - mu I) - log(Heff^dagger - mu I)] (Appendix B),
    evaluated with scipy.linalg.logm in the orthonormal basis, without
    diagonalizing Heff.

    Parameters
    ----------
    Heff_orth : (n, n) complex ndarray
        Effective Hamiltonian F + Sigma in the orthonormal basis.
    X : (nao, n) ndarray
        Orthogonalizer, AO -> orthonormal basis.
    S : (nao, nao) ndarray
        AO overlap matrix.
    mu : float
        Chemical potential.

    Returns
    -------
    P : (nao, nao) ndarray
        AO density matrix (Hermitian).
    Ne : float
        Electron number Tr[P S].

    References
    ----------
    .. [1] A. Arnold, F. Weigend, F. Evers, J. Chem. Phys. 126, 174101 (2007).
    """
    n = Heff_orth.shape[0]
    I = np.eye(n, dtype=complex)

    log1 = logm(Heff_orth - mu * I)
    log2 = logm(Heff_orth.conj().T - mu * I)

    P_orth = (1j / np.pi) * (log1 - log2)        # includes spin factor 2
    P_orth = 0.5 * (P_orth + P_orth.conj().T)    # symmetrize numerically

    P = X @ P_orth @ X.conj().T
    Ne = float(np.real(np.trace(P @ S)))
    return P, Ne