import numpy as np
from scipy.linalg import eig

from .occupations import build_density_gc, f_gc
from .orthogonalization import lowdin_orthogonalizer

from .diis import EDIIS, CDIIS, AndersonDM, HybridEDIISAnderson, herm_real
from .mu_solvers import solve_mu


def one_shot(F_in, Sigma_ao, X, S, mu, target_ne=None, target_ne_tol=1e-9):
    """
    One GC-SCF step: build Heff, (optionally) solve for mu, build the density.

    The density is obtained directly from Heff via the logarithmic expression.
    The eigendecomposition of Heff is only used for the grand-canonical
    occupations and orbital energies (analysis quantities).

    Parameters
    ----------
    F_in : (nao, nao) ndarray
        Fock matrix in the AO basis.
    Sigma_ao : (nao, nao) complex ndarray
        Self-energy in the AO basis (must have a non-zero imaginary part).
    X : (nao, n) ndarray
        Orthogonalizer, AO -> orthonormal basis.
    S : (nao, nao) ndarray
        AO overlap matrix.
    mu : float
        Chemical potential (starting guess if target_ne is given).
    target_ne : float, optional
        If given, mu is adjusted so that Tr[P S] = target_ne.
    target_ne_tol : float, optional
        Absolute tolerance on mu for the target_ne solve.

    Returns
    -------
    P, Ne, f, Z, C_ao, mu, P_raw
        Density, electron number, occupations per spin, eigenvalues of Heff,
        right eigenvectors in the AO basis, chemical potential, and P again
        (kept for interface compatibility).
    """
    if np.allclose(Sigma_ao.imag, 0.0):
        raise ValueError("Sigma has no imaginary part (eta = 0); the "
                         "logarithmic density requires eta > 0.")

    Heff_orth = X.conj().T @ (F_in + Sigma_ao) @ X

    if target_ne is not None:
        mu = solve_mu(Heff_orth, X, S, float(target_ne),
                      mu_init=float(mu), xtol=float(target_ne_tol))

    P, Ne = build_density_gc(Heff_orth, X, S, mu)

    Z, Bp = eig(Heff_orth)
    C_ao = X @ Bp
    f = f_gc(Z, mu)
    
    ne_occ = 2.0 * float(np.sum(f))
    if abs(Ne - ne_occ) > 1e-6:
        raise RuntimeError(
            f"one_shot: Tr[PS] = {Ne:.8f} but 2*sum(f) = {ne_occ:.8f}; "
            "possible branch issue for a decoupled level."
        )
    return P, Ne, f, Z, C_ao, mu, P

def homo_lumo_from_eigs(Z, mol):
    """
    HOMO and LUMO energies from the real parts of the GC eigenvalues.

    Uses the neutral occupation nocc = N_electron / 2 of the molecule.

    Parameters
    ----------
    Z : (n,) complex ndarray
        Eigenvalues of Heff.
    mol : pyscf.gto.Mole
        Molecule (for the electron count).

    Returns
    -------
    homo, lumo : float
        Level energies (nan if undefined).
    """
    z = np.sort(np.real(np.asarray(Z)))
    nocc_ref = int(round(float(mol.nelectron) / 2.0))

    homo = np.nan
    lumo = np.nan

    if 0 < nocc_ref < len(z):
        homo = float(z[nocc_ref - 1])
        lumo = float(z[nocc_ref])

    return homo, lumo

def scf_gc(
    mol,
    Sigma_ao,
    mu=None,
    target_ne=None,
    target_ne_tol=1e-9,
    max_cycle=50,
    conv_tol=1e-6,
    conv_tol_omega=1e-8,
    diis_space=8,
    method='EDIIS',
    logger=None,
    verbose=True,
    V_ext=None,
    mf=None,
    guess="sad",
    dm_init=None,
    early_stop_dP=None,
    checkpoint_path=None,   
    checkpoint_every=10,   
    anderson_mix=0.10,     
):
    """
    Grand-canonical SCF at fixed chemical potential.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule.
    Sigma_ao : (nao, nao) complex ndarray
        Self-energy in the AO basis (eta > 0 required).
    mu : float
        Chemical potential (starting guess if target_ne is given).
    target_ne : float, optional
        If given, mu is adjusted in every iteration so that Tr[P S] = target_ne.
    target_ne_tol : float, optional
        Absolute tolerance on mu for the target_ne solve.
    max_cycle : int, optional
        Maximum number of SCF iterations.
    conv_tol : float, optional
        Convergence threshold on ||Delta P||.
    conv_tol_omega : float, optional
        Convergence threshold on |Delta Omega|.
    diis_space : int, optional
        Size of the mixing history.
    method : str, optional
        Mixing scheme: 'CDIIS', 'EDIIS_E', 'EDIIS_OMEGA' (or 'EDIIS'),
        'ANDERSON', or 'HYBRID'.
    logger : dict, optional
        Log dictionary from scf_log.init_log.
    verbose : bool, optional
        Print one line per iteration.
    V_ext : (nao, nao) ndarray, optional
        External potential added to the core Hamiltonian.
    mf : pyscf mean-field object, optional
        Defaults to RHF(mol); pass an RKS object for DFT.
    guess : str, optional
        PySCF initial guess, used if dm_init is None.
    dm_init : (nao, nao) ndarray, optional
        Initial density matrix.
    early_stop_dP : float, optional
        Abort if ||Delta P|| exceeds this for 5 consecutive iterations.
    checkpoint_path : str, optional
        Save the accepted density to this .npz file.
    checkpoint_every : int, optional
        Checkpoint interval in iterations.
    anderson_mix : float, optional
        Damping factor for standalone Anderson mixing.

    Returns
    -------
    tuple
        (P, Ne, f, F, E_tot, converged, mo_energy, mo_occ_gc, S, C_ao, mu,
        P_raw, mo_gamma): density, electron number, occupations per spin,
        Fock matrix, total energy, convergence flag, level energies e_k,
        occupations 2 f_k, overlap, right eigenvectors (AO), chemical
        potential, P again, and level broadenings gamma_k.
    """
    from pyscf import scf as pyscf_scf
    if mu is None:
        raise ValueError("mu must be set.")
    if mf is None:
        mf = pyscf_scf.RHF(mol)

    S = mol.intor('int1e_ovlp')
    X = lowdin_orthogonalizer(S)

    Hcore = mf.get_hcore()
    if V_ext is not None:
        Hcore = Hcore + V_ext

    if dm_init is not None:
        P = herm_real(np.asarray(dm_init), name="initial density matrix", strict=True)
    else:
        guess = (guess or "sad").lower()
        P = herm_real(mf.get_init_guess(mol, key=guess), name="initial guess density matrix", strict=True)
 
    # normalize method name
    method = (method or "").upper()

    diis = None
    mixer_dm = None
    hybrid = None

    if method == "CDIIS":
        diis = CDIIS(max_diis=diis_space)

    elif method == "EDIIS_E":
        diis = EDIIS(max_diis=diis_space)

    elif method in ("EDIIS_OMEGA", "EDIIS"):
        diis = EDIIS(max_diis=diis_space)
        method = "EDIIS_OMEGA"

    elif method == "ANDERSON":
        mixer_dm = AndersonDM(
            max_diis=diis_space,
            mix=float(anderson_mix),
            reg=1e-10,
            start_iter=2,
            max_coeff_norm=8.0,
        )

    elif method in ("HYBRID", "EDIIS_ANDERSON"):
        # EDIIS(Ω) for stabilization, then Anderson on density
        ediis = EDIIS(max_diis=diis_space)
        anderson = AndersonDM(
            max_diis=4,
            mix=0.3,
            reg=1e-8,
            start_iter=12,
            max_coeff_norm=10.0,
        )
        hybrid = HybridEDIISAnderson(
            ediis=ediis,
            anderson=anderson,
            n_pre_ediis=12,
            switch_dP=1e-2,
        )
        hybrid.reset()

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Use CDIIS, EDIIS_E, EDIIS_OMEGA (or EDIIS), ANDERSON, HYBRID (or EDIIS_ANDERSON)."
        )

    E_nuc = mf.energy_nuc()
    vhf = mf.get_veff(mol, P)
    F = Hcore + vhf

    P0_hr = herm_real(P, name="initial accepted density matrix", strict=True)
    E_tot = float(np.real_if_close(mf.energy_elec(P0_hr, Hcore, vhf)[0])) + E_nuc
    Ne = float(np.einsum("ij,ij->", P0_hr, S).real)
    Omega0 = E_tot - mu * Ne
    
    if logger is not None:
        from .scf_log import log_entry
        log_entry(
            logger, 0,
            E_tot=E_tot, E_elec=np.nan,
            Omega=Omega0, N_e=Ne,
            P=None, dP=np.nan, score=np.nan, dOmega=np.nan,
            homo=np.nan, lumo=np.nan, mu=mu,
        )


    converged = False
    dP_trial = None  # used by hybrid to decide when to switch
    _n_diverge = 0   # consecutive iterations above early_stop_dP
    Omega_prev = Omega0

    for it in range(1, max_cycle + 1):
        # vhf is carried forward from the previous iteration (or pre-loop)
        P_old_hr = herm_real(P, name="accepted density matrix", strict=True)
        F = Hcore + vhf

        Ne_old = float(np.einsum("ij,ij->", P_old_hr, S).real)
        E_old = float(np.real_if_close(mf.energy_elec(P_old_hr, Hcore, vhf)[0])) + E_nuc
        Omega_old = E_old - float(mu) * Ne_old
        
        # --- choose Fock to diagonalize ---
        if hybrid is not None:
            F_use, score, phase = hybrid.mix_fock(
                S, P_old_hr, F, mf, Hcore, vhf, mu=mu, dP_trial=dP_trial
            )
        elif diis is not None:
            mu_for_mixer = None if method in ("EDIIS_E",) else mu
            F_use, score = diis.update(S, P_old_hr, F, mf, Hcore, vhf, mu=mu_for_mixer)
            phase = method
        else:
            # no Fock mixing (Anderson-only mode typically)
            F_use, score, phase = herm_real(F, name="current Fock matrix"), np.nan, "ANDERSON"

        # --- diagonalize -> trial density ---
        P_trial, Ne_trial, f, Z, C_ao, mu, _ = one_shot(F_use, Sigma_ao, X, S, mu,
                                                         target_ne=target_ne,
                                                         target_ne_tol=target_ne_tol)
        homo, lumo = homo_lumo_from_eigs(Z, mol)
        P_trial_hr = herm_real(P_trial, name="trial density matrix", strict=True)
        dP_trial = float(np.linalg.norm(P_trial_hr - P_old_hr))

        # Early exit: if dP_trial stays large for several consecutive iterations,
        # the SCF is diverging — abort instead of running max_cycle wasted steps.
        if early_stop_dP is not None:
            if dP_trial > float(early_stop_dP):
                _n_diverge += 1
                if _n_diverge >= 5:
                    break
            else:
                _n_diverge = 0

        # --- compute vhf and Omega for TRIAL step (needed for rejection) ---
        vhf_trial = mf.get_veff(mol, P_trial_hr)
        Ne_trial = float(np.einsum("ij,ij->", P_trial_hr, S).real)
        E_trial = float(np.real_if_close(mf.energy_elec(P_trial_hr, Hcore, vhf_trial)[0])) + E_nuc
        Omega_trial = E_trial - mu * Ne_trial
        
        # --- accept / mix density for next iteration ---
        if hybrid is not None:
            P_new_hr, info = hybrid.mix_density(P_old_hr, P_trial_hr, dP_trial=dP_trial)
        elif mixer_dm is not None:
            P_new_hr, info = mixer_dm.update(P_old=P_old_hr, P_new=P_trial_hr)
            score = np.nan
        else:
            P_new_hr = P_trial_hr
            info = {"mode": "trial"}

        # reuse vhf_trial when mixer didn't change the density, else rebuild once
        if P_new_hr is P_trial_hr:
            vhf = vhf_trial
        else:
            vhf = mf.get_veff(mol, P_new_hr)

        Ne_acc = float(np.einsum("ij,ij->", P_new_hr, S).real)
        E_acc = float(np.real_if_close(mf.energy_elec(P_new_hr, Hcore, vhf)[0])) + E_nuc
        Omega_acc = E_acc - mu * Ne_acc

        # --- Ω-based rejection (prevents Anderson overshoot) ---
        if np.isfinite(Omega_trial) and np.isfinite(Omega_acc) and (Omega_acc > Omega_trial):            
            P_new_hr = P_trial_hr
            vhf = vhf_trial
            Ne_acc = Ne_trial
            E_acc = E_trial
            Omega_acc = Omega_trial
            info = {"mode": "reject->trial(Omega)"}
            # reset Anderson history after rejection
            if hybrid is not None:
                hybrid.anderson.reset()
            elif mixer_dm is not None:
                mixer_dm.reset()

        dP_acc = float(np.linalg.norm(P_new_hr - P_old_hr))

        # --- diagnostics on accepted density ---
        Ne_new = Ne_acc
        E_tot = E_acc
        Omega = Omega_acc
        dP = dP_acc

        if checkpoint_path is not None and it % max(1, int(checkpoint_every)) == 0:
            np.savez_compressed(
                checkpoint_path, P=P_new_hr, mu=float(mu),
                it=it, Ne=float(Ne_new), E_tot=float(np.real(E_tot)), dP=float(dP),
            )

        # F for next iteration (vhf already set for P_new_hr above)
        F = Hcore + vhf

        # dOmega for logging and convergence
        dOmega = abs(Omega - Omega_prev)
        Omega_prev = Omega

        if logger is not None:
            from .scf_log import log_entry
            log_entry(
            logger, it,
            E_tot=E_tot, E_elec=np.nan,
            Omega=Omega, N_e=Ne_new,
            dP=dP, score=score, P=None, dOmega=dOmega,
            homo=homo, lumo=lumo, mu=mu,
        )

        if verbose:
            tag = phase
            mode = info.get("mode", "n/a") if isinstance(info, dict) else "n/a"

            print(
                    f"it {it:03d} [{tag}/{mode}] "
                    f"dP_trial={dP_trial:.3e} dP_acc={dP_acc:.3e} "
                    f"mu={float(mu):+.6f} HOMO={homo:+.6f} LUMO={lumo:+.6f}"
                )
        # convergence
        converged = (dP < conv_tol)
        if np.isfinite(dOmega):
            converged = converged and (dOmega < conv_tol_omega)

        # accept step (vhf carries forward to next iteration)
        P = P_new_hr
        Ne = Ne_new

        mo_energy = np.real(Z)
        mo_gamma = -np.imag(Z)   # level broadening gamma_k >= 0 (Appendix C)
        mo_occ_gc = 2.0 * f

        if converged:
            break


    P_raw = P  
    if target_ne is not None:
        P, Ne, f, Z, C_ao, mu, P_raw = one_shot(F, Sigma_ao, X, S, mu,
                                                  target_ne=target_ne,
                                                  target_ne_tol=target_ne_tol)
        P = herm_real(P, name="final neutral density", strict=True)
        Ne = float(np.einsum("ij,ij->", P, S).real)
        mo_energy = np.real(Z)
        mo_gamma = -np.imag(Z)   # level broadening gamma_k >= 0 (Appendix C)
        mo_occ_gc = 2.0 * f

    return P, Ne, f, F, E_tot, converged, mo_energy, mo_occ_gc, S, C_ao, mu, P_raw, mo_gamma
