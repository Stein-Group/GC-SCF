from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import root_scalar
from .occupations import build_density_gc

@dataclass
class MuNeutralityConfig:
    """Settings for the neutrality-constrained mu solve: initial bracket and tolerance."""
    bracket: Tuple[float, float] = (-2.0, 2.0)
    xtol: float = 1e-7

def solve_mu(
    Heff_orth: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    target_ne: float,
    mu_init: float,
    bracket_half: float = 0.5,
    xtol: float = 1e-9,
    maxiter: int = 100,
) -> float:
    """
    Find mu such that Tr[P(mu) S] = target_ne for a fixed Heff.

    Only the density is re-evaluated for each trial mu (no Fock builds).
    The root is found with Brent's method.

    Parameters
    ----------
    Heff_orth : (n, n) complex ndarray
        Effective Hamiltonian F + Sigma in the orthonormal basis.
    X : (nao, n) ndarray
        Orthogonalizer, AO -> orthonormal basis.
    S : (nao, nao) ndarray
        AO overlap matrix.
    target_ne : float
        Target electron number.
    mu_init : float
        Starting guess; the search bracket is centred here.
    bracket_half : float, optional
        Initial half-width of the bracket in Ha, doubled up to 12 times.
    xtol : float, optional
        Absolute tolerance on mu.
    maxiter : int, optional
        Maximum number of Brent iterations.

    Returns
    -------
    mu : float
        Chemical potential that reproduces target_ne.

    Raises
    ------
    RuntimeError
        If no bracket is found.
    """
    def ne_of_mu(mu: float) -> float:
        """Electron number Tr[P(mu) S] at fixed Heff."""
        _, Ne = build_density_gc(Heff_orth, X, S, float(mu))
        return Ne

    a = float(mu_init) - bracket_half
    b = float(mu_init) + bracket_half

    bracket = None
    for _ in range(12):
        ne_a = ne_of_mu(a)
        ne_b = ne_of_mu(b)
        if (ne_a - target_ne) * (ne_b - target_ne) <= 0:
            bracket = (a, b)
            break
        bracket_half *= 2.0
        a = float(mu_init) - bracket_half
        b = float(mu_init) + bracket_half

    if bracket is None:
        raise RuntimeError(
            f"solve_mu: no bracket found for target_ne={target_ne} "
            f"(mu_init={mu_init}, last Ne(a)={ne_a:.4f}, Ne(b)={ne_b:.4f})."
        )

    sol = root_scalar(
        lambda mu: ne_of_mu(mu) - target_ne,
        bracket=bracket,
        method="brentq",
        xtol=xtol,
        maxiter=maxiter,
    )
    return float(sol.root) 