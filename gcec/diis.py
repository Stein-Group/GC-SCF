# Copyright 2026 L. Hetzel, M. Head-Gordon, C. J. Stein
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# EDIIS._ediis_minimize is adapted from pyscf/scf/diis.py
# (PySCF 2.12.0, function ediis_minimize):
#   Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#   Licensed under the Apache License, Version 2.0.
# Modifications: real Hermitian inputs, explicit regularization of the
# quadratic form, fallback for degenerate solutions.


import numpy as np
import scipy.optimize
import warnings


def herm_real(A, *, name="matrix", strict=False):
    """
    Return the real Hermitian part of a matrix.

    Parameters
    ----------
    A : (n, n) ndarray
        Matrix to check.
    name : str, optional
        Label used in messages.
    strict : bool, optional
        If True, raise instead of repairing a non-real-Hermitian matrix
        (used for densities).

    Returns
    -------
    A_hr : (n, n) ndarray
        Real Hermitian matrix.

    Raises
    ------
    ValueError
        If strict is True and A is not real Hermitian within tolerance.
    """
    A = np.asarray(A)
    herm_dev = float(np.max(np.abs(A - A.conj().T)))
    imag_max = float(np.max(np.abs(A.imag))) if np.iscomplexobj(A) else 0.0

    if herm_dev < 1e-8 and imag_max < 1e-10:
        return A.real

    msg = (f"Non-real-Hermitian {name} (max |A - A^H| = {herm_dev:.3e}, "
           f"max |Im A| = {imag_max:.3e}).")
    if strict:
        raise ValueError(msg)

    warnings.warn(f"{msg} Symmetrizing and taking real part.", RuntimeWarning, stacklevel=2)
    return 0.5 * (A + A.conj().T).real

class DIIS:
    """Base class for the SCF mixing schemes."""
    def __init__(self, max_diis=6):
        self.max_diis = int(max_diis)


class AndersonDM(DIIS):
    """
    Anderson acceleration on the density matrix with damping and linear-mixing fallback.

    References
    ----------
    .. [1] D. G. Anderson, J. ACM 12, 547 (1965).

    Parameters
    ----------
    max_diis : int, optional
        Number of stored iterations.
    mix : float, optional
        Damping factor toward the accelerated density.
    reg : float, optional
        Tikhonov regularization of the least-squares problem.
    start_iter : int, optional
        Number of initial iterations with plain linear mixing.
    max_coeff_norm : float, optional
        Fall back to linear mixing if the coefficient norm exceeds this value.
    """
    def __init__(self, max_diis=6, mix=0.03, reg=1e-10, start_iter=2, max_coeff_norm=8.0):
        super().__init__(max_diis)
        self.space = int(max_diis)
        self.mix = float(mix)               # linear mixing fraction toward accelerated target
        self.reg = float(reg)
        self.start_iter = int(start_iter)   # delay Anderson until a few iterations
        self.max_coeff_norm = float(max_coeff_norm)

        self._x_hist = []   # P_k (flattened)
        self._f_hist = []   # f_k = G(P_k)-P_k (flattened)
        self._k = 0

    def reset(self):
        """Clear the stored history."""
        self._x_hist.clear()
        self._f_hist.clear()
        self._k = 0

    def update(self, P_old, P_new):
        """
        Parameters
        ----------
        P_old : density used to build the current Fock (x_k)
        P_new : density obtained after diagonalization (g_k)

        Returns
        -------
        P_mix : mixed density to use for next vhf/F build
        info  : dict with diagnostic numbers
        """
        self._k += 1

        xk = herm_real(P_old, name="old density matrix", strict=True)
        gk = herm_real(P_new, name="new density matrix", strict=True)
        fk = (gk - xk) #difference (residual)

        xk_flat = xk.real.ravel().astype(float, copy=True)
        fk_flat = fk.real.ravel().astype(float, copy=True)

        # store histories
        self._x_hist.append(xk_flat)
        self._f_hist.append(fk_flat)
        if len(self._x_hist) > self.space + 1: #if we have more than max_diis history, pop the oldest
            self._x_hist.pop(0)
            self._f_hist.pop(0)

        # Not enough history or too early: do simple linear mixing
        if self._k < self.start_iter or len(self._f_hist) < 3:
            P_lin = (1.0 - self.mix) * xk + self.mix * gk
            return herm_real(P_lin, name="linearly mixed density matrix", strict=True), {"mode": "linear", "fnorm": float(np.linalg.norm(fk_flat))}

        # Build ΔF and ΔX from the window (n residuals -> n-1 diffs)
        f_list = self._f_hist
        x_list = self._x_hist
        n = len(f_list)

        dF = np.column_stack([f_list[i+1] - f_list[i] for i in range(n-1)])  # (dim, n-1)
        dX = np.column_stack([x_list[i+1] - x_list[i] for i in range(n-1)])  # (dim, n-1)

        # Solve min || f_k - dF * c || via regularized normal eq
        A = dF.T @ dF + self.reg * np.eye(dF.shape[1])
        b = dF.T @ fk_flat
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c, *_ = np.linalg.lstsq(dF, fk_flat, rcond=None)

        # guard against insane coefficients
        cnorm = float(np.linalg.norm(c))
        if not np.isfinite(cnorm) or cnorm > self.max_coeff_norm:
            P_lin = (1.0 - self.mix) * xk + self.mix * gk
            return herm_real(P_lin, name="fallback mixed density matrix", strict=True), {"mode": "linear(fallback_coeffs)", "fnorm": float(np.linalg.norm(fk_flat)), "cnorm": cnorm}

        # Anderson target: g_k - dX*c   (type-I form for fixed point)
        target_flat = gk.real.ravel().astype(float, copy=False) - dX @ c
        target = target_flat.reshape(xk.shape)

        # Final damped step toward target (keeps it stable)
        P_mix = (1.0 - self.mix) * xk + self.mix * target
        return herm_real(P_mix, name="Anderson mixed density matrix", strict=True), {"mode": "anderson", "fnorm": float(np.linalg.norm(fk_flat)), "cnorm": cnorm}

class EDIIS(DIIS):
    """
    Energy DIIS on the Fock matrix; minimizes E, or Omega = E - mu N if mu is given.

    References
    ----------
    .. [1] K. N. Kudin, G. E. Scuseria, E. Cancès, J. Chem. Phys. 116, 8255 (2002).

    Parameters
    ----------
    max_diis : int, optional
        Number of stored iterations.
    """
    def __init__(self, max_diis=6):
        super().__init__(max_diis)
        self._buffer = {}
        self._head = 0
        self.space = max_diis

    def reset(self):
        """Clear the stored history."""
        self._buffer.clear()
        self._head = 0

    def update(self, S, P, F, mf, Hcore, vhf, mu=None):
        """
        Parameters
        ----------
        S : (nao,nao) overlap (AO)
        P : (nao,nao) current density (complex allowed)
        F : (nao,nao) current Fock (complex allowed)
        mf, Hcore, vhf : PySCF objects/matrices for energy evaluation
        mu : float or None    # when given, minimize grand potential
        """
        # Hermitian-real snapshots for the mixer & RHF energy
        P_hr = herm_real(P, name="EDIIS density matrix", strict=True)
        F_hr = herm_real(F, name="EDIIS Fock matrix")

        # Initialize buffers on first call (real storage is sufficient)
        if not self._buffer:
            shape = (self.space,) + F.shape
            self._buffer['dm']    = np.zeros(shape, dtype=float)
            self._buffer['fock']  = np.zeros(shape, dtype=float)
            self._buffer['score'] = np.full(self.space, np.inf, dtype=float)

        i = self._head % self.space
        self._buffer['dm'][i]   = P_hr
        self._buffer['fock'][i] = F_hr

        # Electronic energy from PySCF (real), add Enuc and optional -mu*N
        e_elec = float(np.real_if_close(mf.energy_elec(P_hr, Hcore, vhf)[0]))
        e_tot  = e_elec + mf.energy_nuc()
        if mu is None:
            score = e_tot
        else:
            n_el  = float(np.einsum('ij,ij->', P_hr, S))
            score = e_tot - float(mu) * n_el
        self._buffer['score'][i] = float(score)

        self._head += 1
        n = min(self._head, self.space)
        if n < 2:
            # Not enough history: return current Fock (Hermitian-real)
            return F_hr, float(score)

        Ps = self._buffer['dm'][:n]     # (n,*,nao,nao), real
        Fs = self._buffer['fock'][:n]   # (n,*,nao,nao), real
        Es = self._buffer['score'][:n]  # (n,),          real

        score_min, coeffs = self._ediis_minimize(Es, Ps, Fs)
        F_mixed = np.einsum('i,i...pq->...pq', coeffs, Fs)  # real Hermitian
        return F_mixed, score_min

    # Adapted from pyscf.scf.diis.ediis_minimize (Apache 2.0, see file header).
    @staticmethod
    def _ediis_minimize(Es, Ps, Fs):
        """
        Minimize the EDIIS model over convex combinations of stored iterates.

        Minimizes f(c) = sum_i c_i E_i - c^T M c with
        M_ij = Tr[(P_i - P_j)(F_i - F_j)] / 2, where the convexity constraint
        is imposed through c_i = x_i^2 / sum_j x_j^2.

        Parameters
        ----------
        Es : (m,) ndarray
            Energies (or grand potentials) of the stored iterates.
        Ps, Fs : (m, nao, nao) ndarray
            Stored density and Fock matrices.

        Returns
        -------
        f_min : float
            Minimum of the model function.
        c : (m,) ndarray
            Mixing coefficients (non-negative, summing to one).
        """
        # Ensure real quantities for optimizer
        Es = np.asarray(Es, dtype=float)
        nx = Es.size
        nao = Ps.shape[-1]

        Ps = Ps.reshape(nx, -1, nao, nao)   # already real/Hermitian
        Fs = Fs.reshape(nx, -1, nao, nao)

        # Build df and the symmetric M (use real parts only)
        df = np.einsum('inpq,jnqp->ij', Ps, Fs)    # real
        df = df.astype(float)
        diag = df.diagonal().copy()
        M = diag[:, None] + diag[None, :] - df - df.T
        if nx > 1:
            M += 1e-14 * np.eye(nx)

        def costf(x):
            """Model function f(x)."""
            x = np.asarray(x, dtype=float)
            x2 = x * x
            s  = x2.sum()
            if s <= 0.0:
                return np.inf
            c  = x2 / s
            return float(np.dot(c, Es) - c @ M @ c)

        def grad(x):
            """Gradient of the model function with respect to x."""
            x = np.asarray(x, dtype=float)
            x2 = x * x
            s  = x2.sum()
            if s <= 0.0:
                return np.ones_like(x, dtype=float)
            c  = x2 / s
            fc = Es - 2.0 * (c @ M)
            cx = (np.diag(x) * s - np.outer(x2, x)) * (2.0 / (s * s))
            g  = cx @ fc
            return np.asarray(g, dtype=float)

        res = scipy.optimize.minimize(
            costf, np.ones(nx, dtype=float), method='BFGS', jac=grad, tol=1e-9
        )
        x = np.asarray(res.x, dtype=float)
        x2 = x * x
        s  = x2.sum()
        if s <= 0.0:
            k = int(np.argmin(Es))
            c = np.zeros(nx, dtype=float); c[k] = 1.0
            return float(Es[k]), c
        c = x2 / s
        return float(res.fun), c


class CDIIS(DIIS):
    """
    Commutator DIIS (Pulay) on the Fock matrix with residual FPS - SPF.

    References
    ----------
    .. [1] P. Pulay, Chem. Phys. Lett. 73, 393 (1980).
    .. [2] P. Pulay, J. Comput. Chem. 3, 556 (1982).

    Parameters
    ----------
    max_diis : int, optional
        Number of stored iterations.
    """
    def __init__(self, max_diis=6):
        super().__init__(max_diis)
        self.space = max_diis
        self._F_hist = []
        self._R_hist = []

    def reset(self):
        """Clear the stored history."""
        self._F_hist.clear()
        self._R_hist.clear()

    def update(self, S, P, F, *args, **kwargs):
        """
        Add the current iterate and return the extrapolated Fock matrix.

        Parameters
        ----------
        S : (nao, nao) ndarray
            AO overlap matrix.
        P : (nao, nao) ndarray
            Current density matrix.
        F : (nao, nao) ndarray
            Current Fock matrix.
        *args
            Optional (mf, Hcore, vhf), only used for the diagnostic Omega.
        **kwargs
            Optional mf, Hcore, vhf, mu (same purpose).

        Returns
        -------
        F_mix : (nao, nao) ndarray
            Extrapolated Fock matrix.
        omega : float
            Grand potential of the current iterate (nan if context is missing).
        """
        mf    = args[0] if len(args) > 0 else kwargs.get('mf', None)
        Hcore = args[1] if len(args) > 1 else kwargs.get('Hcore', None)
        vhf   = args[2] if len(args) > 2 else kwargs.get('vhf', None)
        mu    = kwargs.get('mu', None)

        # use Hermitian–real for metric and energy eval
        P_hr = herm_real(P, name="CDIIS density matrix", strict=True)
        F_hr = herm_real(F, name="CDIIS Fock matrix")

        # compute a diagnostic Ω from the *current* snapshot (not used for mixing)
        omega = float('nan')
        if (mf is not None) and (Hcore is not None) and (vhf is not None) and (mu is not None):
            e_elec = float(np.real_if_close(mf.energy_elec(P_hr, Hcore, vhf)[0]))
            e_tot  = e_elec + mf.energy_nuc()
            n_el   = float(np.einsum('ij,ij->', P_hr, S))
            omega  = float(np.real_if_close(e_tot - float(mu) * n_el))

        # standard CDIIS residual
        R = F_hr @ P_hr @ S - S @ P_hr @ F_hr

        self._F_hist.append(F_hr.copy())
        self._R_hist.append(R.ravel().astype(float, copy=True))
        if len(self._F_hist) > self.space:
            self._F_hist.pop(0)
            self._R_hist.pop(0)

        m = len(self._R_hist)
        if m < 2:
            # return current Fock and the diagnostic Ω
            return F_hr, omega

        # Pulay B-matrix
        B = np.empty((m + 1, m + 1), dtype=float)
        B[-1, :] = 1.0
        B[:, -1] = 1.0
        B[-1, -1] = 0.0
        for i in range(m):
            ei = self._R_hist[i]
            for j in range(m):
                ej = self._R_hist[j]
                B[i, j] = float(np.vdot(ei, ej).real)

        rhs = np.zeros(m + 1, dtype=float)
        rhs[-1] = 1.0
        try:
            sol = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            B[:m, :m] += 1e-12 * np.eye(m)
            sol = np.linalg.solve(B, rhs)

        coeffs = sol[:m]
        dtype = np.result_type(*self._F_hist)
        F_mix = np.zeros(self._F_hist[0].shape, dtype=dtype)
        for ci, Fi in zip(coeffs, self._F_hist):
            F_mix += ci * Fi

        # return mixed Fock + diagnostic Ω (still from current snapshot)
        return F_mix, omega
    
class HybridEDIISAnderson(DIIS):
    """
    EDIIS(Omega) for the first iterations, then Anderson on the density.

    References
    ----------
    .. [1] K. N. Kudin, G. E. Scuseria, E. Cancès, J. Chem. Phys. 116, 8255 (2002).
    .. [2] D. G. Anderson, J. ACM 12, 547 (1965).

    Parameters
    ----------
    ediis : EDIIS
        Mixer for the first phase.
    anderson : AndersonDM
        Mixer for the second phase.
    n_pre_ediis : int, optional
        Minimum number of EDIIS iterations.
    switch_dP : float, optional
        Switch to Anderson once ||Delta P|| falls below this value
        (after n_pre_ediis iterations).
    """
    def __init__(
        self,
        *,
        ediis: EDIIS,
        anderson: AndersonDM,
        n_pre_ediis: int = 12,
        switch_dP: float = 1e-2,
    ):
        super().__init__(max_diis=max(ediis.space, anderson.space))
        self.ediis = ediis
        self.anderson = anderson

        self.n_pre_ediis = int(n_pre_ediis)    # always use EDIIS for first N iterations
        self.switch_dP = float(switch_dP)      # or switch once ||ΔP|| is below this

        self._it = 0
        self._phase = "ediis"

    def reset(self):
        """Clear both histories and return to the EDIIS phase."""
        self._it = 0
        self._phase = "ediis"
        self.ediis.reset()
        self.anderson.reset()

    @property
    def phase(self):
        """Current phase, 'ediis' or 'anderson'."""
        return self._phase

    def mix_fock(self, S, P, F, mf, Hcore, vhf, mu=None, dP_trial=None):
        """
        Return the Fock matrix to diagonalize.

        In the EDIIS phase the EDIIS(Omega)-mixed Fock matrix is returned,
        in the Anderson phase the current Fock matrix.

        Parameters
        ----------
        S, P, F : (nao, nao) ndarray
            Overlap, current density and current Fock matrix.
        mf, Hcore, vhf
            Mean-field object, core Hamiltonian and two-electron potential,
            used for the energy evaluation in EDIIS.
        mu : float, optional
            Chemical potential; if given, EDIIS minimizes Omega.
        dP_trial : float, optional
            ||Delta P|| of the previous step, used for the phase switch.

        Returns
        -------
        F_use : (nao, nao) ndarray
            Fock matrix to diagonalize.
        score : float
            EDIIS model value (nan in the Anderson phase).
        phase : str
            'ediis' or 'anderson'.
        """
        self._it += 1

        # decide phase
        if self._phase == "ediis":
            if (self._it > self.n_pre_ediis) and (dP_trial is not None) and (float(dP_trial) < self.switch_dP):
                self._phase = "anderson"

        if self._phase == "ediis":
            F_use, score = self.ediis.update(S, P, F, mf, Hcore, vhf, mu=mu)
            return F_use, score, "ediis"
        else:
            # In the local phase we typically do not need Fock mixing.
            return herm_real(F, name="Anderson Fock matrix"), float("nan"), "anderson"

    def mix_density(self, P_old, P_new, dP_trial=None):
        """
        Return the density for the next Fock build.

        In the EDIIS phase the trial density is accepted unchanged, in the
        Anderson phase it is mixed with the same update as AndersonDM.

        Parameters
        ----------
        P_old, P_new : (nao, nao) ndarray
            Density used to build the current Fock matrix and trial density.
        dP_trial : float, optional
            Unused; kept for interface compatibility.

        Returns
        -------
        P_use : (nao, nao) ndarray
            Density for the next iteration.
        info : dict
            Diagnostics ('mode', 'phase').
        """
        P_old_hr = herm_real(P_old, name="old hybrid density matrix", strict=True)
        P_new_hr = herm_real(P_new, name="new hybrid density matrix", strict=True)

        if self._phase == "ediis":
            P_use = P_new_hr
            info = {"mode": "accept", "phase": "ediis"}
            return P_use, info

        # Anderson phase: same update as standalone AndersonDM.
        P_acc, info = self.anderson.update(P_old_hr, P_new_hr)
        info["phase"] = "anderson"
        return P_acc, info
