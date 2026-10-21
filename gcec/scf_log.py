import numpy as np

def init_log():
    """
    Create an empty SCF log.

    Returns
    -------
    log : dict
        Lists for iter, E_tot, E_elec, Omega, N_e, P, dP, score, dOmega,
        homo, lumo and mu, filled by log_entry.
    """
    return {
        "iter": [], "E_tot": [], "E_elec": [], "Omega": [], "N_e": [],
        "P": [], "dP": [], "score": [], "dOmega": [],
        "homo": [], "lumo": [], "mu": [],
    }

def log_entry(log, it, *, E_tot=None, E_elec=None, Omega=None, N_e=None,
              P=None, dP=None, score=None, dOmega=None, homo=None, lumo=None, mu=None):
    """
    Append one SCF iteration to the log; missing values are stored as nan.

    Parameters
    ----------
    log : dict
        Log from init_log.
    it : int
        Iteration number (0 = initial guess).
    E_tot, E_elec, Omega, N_e, P, dP, score, dOmega, homo, lumo, mu : optional
        Quantities of this iteration.
    """
    log["iter"].append(int(it))
    log["E_tot"].append(float(np.real(E_tot)) if E_tot is not None else np.nan)
    log["E_elec"].append(float(np.real(E_elec)) if E_elec is not None else np.nan)
    log["Omega"].append(float(np.real(Omega)) if Omega is not None else np.nan)
    log["N_e"].append(float(np.real(N_e)) if N_e is not None else np.nan)
    log["P"].append(np.array(P) if P is not None else None)
    log["dP"].append(float(np.real(dP)) if dP is not None else np.nan)
    log["score"].append(float(np.real(score)) if score is not None else np.nan)
    log["dOmega"].append(float(np.real(dOmega)) if dOmega is not None else np.nan)
    log["homo"].append(float(homo) if homo is not None else np.nan)
    log["lumo"].append(float(lumo) if lumo is not None else np.nan)
    log["mu"].append(float(mu) if mu is not None else np.nan)