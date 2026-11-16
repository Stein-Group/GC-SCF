#!/usr/bin/env python3
"""
Electronic-structure analysis of the optimized H-ring fragments
(paper Fig. 4b, SI Table S6 and SI Fig. S13).

For each fragment H6-H20 (published parameters) and the references H38, H40:
  - canonical HOMO, LUMO and gap Delta E_HL (RKS/PBE0), without the point
    charge and with q = +1 at r_min = 1.2181 A
  - fragments only: grand-canonical levels e_k = Re Z_k and occupations f_k
    (Appendix C) of the effective Hamiltonian, without and with the charge;
    from these
      Delta_mu  = (lowest level above mu) - (highest level below mu)
      T_eff     from a least-squares fit of ln(2/f_k - 1) = beta (e_k - mu)
                over levels with 1e-3 < f_k < 2 - 1e-3 and |e_k - mu| <= 0.3 Ha;
                k_B T_eff = 1/beta, standard error from the fit

Output: results/electronic_structure/gaps_teff.csv (all numbers),
        gaps_teff_table.tex (SI Table S6), metadata.json
Plot with plot_electronic_structure.py.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from pyscf import dft, qmmm

from gcec import make_sigma_ao_diagonal
from gcec.scf_loop import scf_gc
from run_fragment_curves import (THIS_DIR, XC, GRID_LEVEL, X0, POINT_CHARGE_Q, SCF, FRAGMENTS,
                                 build_mol, outer_h_aos, metadata)
from run_density_comparison import R_MIN

OUT_DIR = THIS_DIR / "results" / "electronic_structure"
SYSTEMS = ["H6", "H8_a", "H10", "H12", "H14", "H16", "H18", "H20"]   # Fig. 4 / Table S6
REFERENCES = {"H38": "ring_h38.xyz", "H40": "ring_full.xyz"}
KB_HA_PER_K = 3.166811563e-6
OCC_MIN, OCC_MAX, EWIN = 1e-3, 2.0 - 1e-3, 0.30
CANONICAL_SCF = dict(conv_tol=1e-10, max_cycle=800)


def with_charge(mf, on: bool):
    return qmmm.mm_charge(mf, [(X0 + R_MIN, 0.0, 0.0)], [POINT_CHARGE_Q], unit="Ang") if on else mf


def canonical_gap(xyz: str, charge: bool):
    mol = build_mol(xyz)
    mf = dft.RKS(mol); mf.xc, mf.grids.level, mf.verbose = XC, GRID_LEVEL, 0
    for k, v in CANONICAL_SCF.items():
        setattr(mf, k, v)
    mf = with_charge(mf, charge)
    mf.kernel()
    nocc = mol.nelectron // 2
    homo, lumo = float(mf.mo_energy[nocc - 1]), float(mf.mo_energy[nocc])
    return homo, lumo, lumo - homo, bool(mf.converged)


def gc_levels(label: str, charge: bool):
    xyz, eps, eta, mu = FRAGMENTS[label]
    mol = build_mol(xyz)
    Sigma_ao, _ = make_sigma_ao_diagonal(mol, idxs=outer_h_aos(mol), gamma_val=eta, eps_val=eps)
    mf = dft.RKS(mol); mf.xc, mf.grids.level = XC, GRID_LEVEL
    out = scf_gc(mol, Sigma_ao, mu=float(mu), logger=None, verbose=False, mf=with_charge(mf, charge), **SCF)
    return np.asarray(out[6], float), np.asarray(out[7], float), bool(out[5])


def spacing(e, mu):
    below, above = e[e < mu], e[e > mu]
    if not len(below) or not len(above):
        return np.nan, np.nan, np.nan
    return float(below.max()), float(above.min()), float(above.min() - below.max())


def teff_fit(e, f, mu):
    m = (f > OCC_MIN) & (f < OCC_MAX) & (np.abs(e - mu) <= EWIN)
    n = int(m.sum())
    if n < 2:
        return np.nan, np.nan, n
    x, y = e[m] - mu, np.log(2.0 / f[m] - 1.0)
    A = np.vstack([x, np.ones_like(x)]).T
    (beta, b0), *_ = np.linalg.lstsq(A, y, rcond=None)
    if beta <= 0:
        return np.nan, np.nan, n
    kbt = 1.0 / beta
    se = np.nan
    if n > 2:                                   # standard error of the slope, propagated to 1/beta
        s2 = float(np.sum((y - A @ [beta, b0]) ** 2)) / (n - 2)
        se = kbt ** 2 * float(np.sqrt(max((s2 * np.linalg.inv(A.T @ A))[0, 0], 0.0)))
    return float(kbt), float(se), n


def row(label: str) -> dict:
    is_ref = label in REFERENCES
    xyz = REFERENCES[label] if is_ref else FRAGMENTS[label][0]
    r = {"system": label}
    for tag, ch in (("q0", False), ("q", True)):
        homo, lumo, gap, conv = canonical_gap(xyz, ch)
        r |= {f"homo_{tag}": homo, f"lumo_{tag}": lumo, f"gap_{tag}": gap, f"canonical_conv_{tag}": conv}
    if is_ref:
        return r
    _, eps, eta, mu = FRAGMENTS[label]
    r |= {"eps": eps, "eta": eta, "mu": mu}
    for tag, ch in (("q0", False), ("q", True)):
        e, f, conv = gc_levels(label, ch)
        lo, hi, dmu = spacing(e, mu)
        kbt, se, n = teff_fit(e, f, mu)
        r |= {f"gc_below_mu_{tag}": lo, f"gc_above_mu_{tag}": hi, f"delta_mu_{tag}": dmu,
              f"kbt_{tag}": kbt, f"kbt_se_{tag}": se, f"teff_K_{tag}": kbt / KB_HA_PER_K,
              f"nfit_{tag}": n, f"gc_conv_{tag}": conv}
    r["kbt_over_delta_mu_q0"] = r["kbt_q0"] / r["delta_mu_q0"]
    return r


def tex_table(rows) -> str:
    def f(x, nd=6):
        return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    def k(x, n):
        if x is None or not np.isfinite(x):
            return "--"
        return f"{int(round(x)):,}".replace(",", r"\,") + (r"$^\dagger$" if n == 2 else "")
    lines = [r"\begin{tabular}{lccccccc}", r"\hline",
             r"System & $\Delta E_\mathrm{HL}$ & $\Delta E_\mathrm{HL}(q@r_\mathrm{min})$ & $\Delta_\mu$ & "
             r"$\Delta_\mu(q@r_\mathrm{min})$ & $T_\mathrm{eff}$ & $T_\mathrm{eff}(q@r_\mathrm{min})$ & "
             r"$k_\mathrm{B}T_\mathrm{eff}/\Delta_\mu$ \\",
             r" & (Ha) & (Ha) & (Ha) & (Ha) & (K) & (K) & \\", r"\hline"]
    for r in rows:
        name = rf"H$_{{{r['system'].split('_')[0][1:]}}}$"
        lines.append(f"{name} & {f(r['gap_q0'])} & {f(r['gap_q'])} & {f(r.get('delta_mu_q0'))} & "
                     f"{f(r.get('delta_mu_q'))} & {k(r.get('teff_K_q0'), r.get('nfit_q0'))} & "
                     f"{k(r.get('teff_K_q'), r.get('nfit_q'))} & {f(r.get('kbt_over_delta_mu_q0'), 3)} \\\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    rows = []
    for label in SYSTEMS + list(REFERENCES):
        rows.append(row(label))
        print(f"{label}: gap = {rows[-1]['gap_q0']:.6f} Ha"
              + (f", Delta_mu = {rows[-1]['delta_mu_q0']:.6f}, T_eff = {rows[-1]['teff_K_q0']:.0f} K"
                 if label not in REFERENCES else ""), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT_DIR / "gaps_teff.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (OUT_DIR / "gaps_teff_table.tex").write_text(tex_table(rows))
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata(
        script="run_electronic_structure.py", systems=SYSTEMS + list(REFERENCES), r_min_ang=R_MIN,
        teff_fit=dict(model="f = 2/(1+exp(beta (e - mu)))", occ_min=OCC_MIN, occ_max=OCC_MAX,
                      energy_window_ha=EWIN, kb_ha_per_k=KB_HA_PER_K),
        canonical_scf=CANONICAL_SCF), indent=2))
    print(f"saved {OUT_DIR}")


if __name__ == "__main__":
    main()
