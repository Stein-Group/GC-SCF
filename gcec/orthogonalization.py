import numpy as np
from scipy.linalg import eigh

def lowdin_orthogonalizer(S: np.ndarray) -> np.ndarray:
    """
    Symmetric (Löwdin) orthogonalizer X = S^(-1/2).

    Parameters
    ----------
    S : (nao, nao) ndarray
        AO overlap matrix.

    Returns
    -------
    X : (nao, nao) ndarray
        Orthogonalizer, AO -> orthonormal basis.
    """
    w, U = eigh(S)
    return U @ np.diag(w**-0.5) @ U.T
