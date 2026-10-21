# gcec/main.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from pyscf import scf as pyscf_scf
from pyscf import dft as pyscf_dft

from .orthogonalization import lowdin_orthogonalizer
from .scf_loop import scf_gc


@dataclass
class SCFSettings:
    """
    Settings for a GC-SCF run.

    Attributes
    ----------
    max_cycle : int
        Maximum number of SCF iterations.
    conv_tol : float
        Convergence threshold on ||Delta P||.
    conv_tol_omega : float
        Convergence threshold on |Delta Omega|.
    diis_space : int
        Size of the mixing history.
    method : str
        Mixing scheme passed to scf_gc.
    guess : str
        PySCF initial guess.
    theory : str
        'RHF' or 'RKS'.
    xc : str
        Exchange-correlation functional (RKS only).
    use_df : bool
        Use density fitting.
    pcm, pcm_method, pcm_eps
        Optional PySCF PCM solvation (RKS only).
    verbose : bool
        Print one line per SCF iteration.
    """
    max_cycle: int = 50
    conv_tol: float = 1e-6
    conv_tol_omega: float = 1e-8
    diis_space: int = 8
    method: str = "EDIIS"
    guess: str = "sad"

    # choose theory here:
    theory: str = "RHF"   # "RHF" or "RKS"
    xc: str = "pbe"       # only used for RKS

    use_df: bool = False  # optional: density fitting
    pcm: bool = False      # optional: PySCF PCM for RKS
    pcm_method: str = "IEF-PCM"
    pcm_eps: float = 78.3553
    verbose: bool = True


class GCEC:
    """
    Main entry point for GC-SCF calculations.

    Creates a PySCF mean-field object (RHF or RKS) unless one is supplied,
    caches overlap/orthogonalizer, and calls the GC-SCF kernel scf_gc().

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule.
    Sigma_ao : (nao, nao) complex ndarray
        Self-energy in the AO basis (eta > 0 required).
    mu : float
        Chemical potential.
    V_ext : (nao, nao) ndarray, optional
        External potential added to the core Hamiltonian.
    mf : pyscf mean-field object, optional
        If omitted, built from settings (RHF or RKS).
    settings : SCFSettings, optional
        SCF settings; defaults to SCFSettings().
    """
    def __init__(
        self,
        mol,
        *,
        Sigma_ao: np.ndarray,
        mu: float,
        V_ext: Optional[np.ndarray] = None,
        mf: Optional[pyscf_scf.hf.SCF] = None,
        settings: Optional[SCFSettings] = None,
    ):
        self.mol = mol
        self.Sigma_ao = np.asarray(Sigma_ao, dtype=complex)
        self.mu = mu
        self.V_ext = V_ext
        self.settings = settings or SCFSettings()
        self.guess=self.settings.guess

        # ---- Build / accept PySCF mean-field object ----
        if mf is not None:
            self.mf = mf
        else:
            th = (self.settings.theory or "RHF").upper()
            if th == "RKS":
                if self.settings.pcm:
                    self.mf = pyscf_dft.RKS(mol).PCM()
                    self.mf.with_solvent.method = self.settings.pcm_method
                    self.mf.with_solvent.eps = float(self.settings.pcm_eps)
                else:
                    self.mf = pyscf_dft.RKS(mol)
                self.mf.xc = self.settings.xc
            elif th == "RHF":
                if self.settings.pcm:
                    raise ValueError("PCM is only supported for RKS in this interface.")
                self.mf = pyscf_scf.RHF(mol)
            else:
                raise ValueError(f"Unknown theory={self.settings.theory!r}. Use 'RHF' or 'RKS'.")

        # optional density fitting
        if bool(self.settings.use_df):
            # PySCF supports density_fit() for both RHF and RKS objects
            self.mf = self.mf.density_fit()

        # ---- Cached basics (often useful for diagnostics/plots) ----
        self.S = self.mol.intor("int1e_ovlp")
        self.X = lowdin_orthogonalizer(self.S)
        self.Enuc = self.mol.energy_nuc()


    def run_scf(
        self,
        *,
        logger: Optional[Dict[str, Any]] = None,
        dm_init: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run the GC-SCF with the stored settings.

        Parameters
        ----------
        logger : dict, optional
            Log dictionary from scf_log.init_log.
        dm_init : (nao, nao) ndarray, optional
            Initial density matrix.

        Returns
        -------
        result : dict
            Keys: P, Ne, f, F, E_tot, Omega, converged, mo_energy, mo_occ_gc,
            mo_gamma, mo_coeff, S, mu, log.
        """
        (P, Ne, f, F, E_tot, converged, mo_energy, mo_occ_gc,
         S, C_ao, mu, P_raw, mo_gamma) = scf_gc(
            self.mol,
            Sigma_ao=self.Sigma_ao,
            mu=self.mu,
            mf=self.mf,
            max_cycle=self.settings.max_cycle,
            conv_tol=self.settings.conv_tol,
            conv_tol_omega=self.settings.conv_tol_omega,
            diis_space=self.settings.diis_space,
            method=self.settings.method,
            logger=logger,
            verbose=self.settings.verbose,
            V_ext=self.V_ext,
            guess=self.guess,
            dm_init=dm_init,
        )

        return {
            "P": np.asarray(P),
            "Ne": float(Ne),
            "f": np.asarray(f),
            "F": np.asarray(F),
            "E_tot": float(E_tot),
            "Omega": float(E_tot) - float(mu) * float(Ne),
            "converged": bool(converged),
            "mo_energy": np.asarray(mo_energy),
            "mo_occ_gc": np.asarray(mo_occ_gc),
            "mo_gamma": np.asarray(mo_gamma),
            "mo_coeff": np.asarray(C_ao),
            "S": np.asarray(S),
            "mu": float(mu),
            "log": logger,
        }
