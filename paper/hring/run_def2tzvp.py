#!/usr/bin/env python3
"""
H6 in the def2-TZVP basis (paper Fig. 5, SI Sec. S3.5). RKS/PBE0, grid level 3,
point charge q = +1 at (x0 + r, 0, 0) on the 68 r values of r_grid.txt.

Parts (run all, or one with --part):
  refs     canonical Delta E(r) of H40, H38 and the uncoupled H6 (plain pySCF)
  gc       GC-SCF Delta Omega(r) and N_e(r) of H6 with the published parameters
             s   : Sigma on the intermediate s function (exponent 0.327) of the
                   two outer H atoms
             s+p : additionally on their in-plane p_x, p_y functions
  density  density matrices at the density-analysis points (H40 canonical and
           both coupled H6) for plot_def2tzvp.py

The coupling parameters were optimized beforehand (two-stage protocol, SI S3.5)
and are not re-optimized here. Following the optimization runs, the GC curves
are solved from r = 100 A inwards, each point starting from the density of the
previous one (Anderson, mixing 0.03, <= 600 iterations); a point that does not
converge is repeated from the q = 0 density, then from SAD, then from SAD with
mixing 0.10.

Output: results/def2tzvp/<system>/curve.npz, curve.csv, metadata.json and
        results/def2tzvp/density/dm_<system>__r<r>.npy
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
from pyscf import dft, qmmm

from gcec.scf_loop import scf_gc
from run_fragment_curves import THIS_DIR, HA_TO_KCAL, XC, GRID_LEVEL, X0, POINT_CHARGE_Q, read_r_grid, metadata
from run_fragment_curves import build_mol as build_mol_minao
import run_fragment_curves as rfc

BASIS = "def2-tzvp"
OUT = THIS_DIR / "results" / "def2tzvp"
REFS = {"H40": "ring_full.xyz", "H38": "ring_h38.xyz", "H6": "ring_3uc.xyz"}
FRAG_XYZ = "ring_3uc.xyz"

# published parameters (Ha), Fig. 5b
COUPLED = {
    "H6_s":  dict(eps_s=-0.021256572991842568, eta_s=0.08285206795411573,
                  mu=-0.3129462120430777),
    "H6_sp": dict(eps_s=-0.02589248688506986, eta_s=0.10073108775442383,
                  eps_p=1.0007011807928377, eta_p=0.045576091792799506,
                  mu=-0.30839925751251224),
}
R_MIN_TZVP = 0.6747765515032179      # minimum of the def2-TZVP H40 curve (quadratic interpolation)
R_MIN_MINAO = 1.2181190329978901     # minimum of the MINAO H40 curve
DENSITY_R = (R_MIN_TZVP, R_MIN_MINAO)

GC_SCF = dict(method="ANDERSON", max_cycle=600, conv_tol=1e-5, conv_tol_omega=1e-8,
              diis_space=8, anderson_mix=0.03)
RETRY = [("q0", {}), ("sad", {}), ("sad", {"anderson_mix": 0.10})]
REF_SCF = dict(conv_tol=1e-10, max_cycle=800, diis_space=20, level_shift=0.05, damp=0.20)


def build_mol(xyz):
    old = rfc.BASIS
    rfc.BASIS = BASIS
    try:
        return build_mol_minao(xyz)
    finally:
        rfc.BASIS = old


def with_charge(mf, r):
    return mf if r is None else qmmm.mm_charge(mf, [(X0 + float(r), 0.0, 0.0)], [POINT_CHARGE_Q], unit="Ang")


def canonical(mol, r=None, dm0=None):
    mf = dft.RKS(mol); mf.xc, mf.grids.level, mf.verbose = XC, GRID_LEVEL, 0
    for k, v in REF_SCF.items():
        setattr(mf, k, v)
    mf = with_charge(mf, r)
    mf.kernel(dm0=dm0)
    return mf


def coupling_sigma(mol, p: dict):
    """Diagonal AO self-energy: intermediate s (and optionally in-plane p) of the two outer H atoms."""
    y = mol.atom_coords(unit="Ang")[:, 1]
    outer = {int(np.argmin(y)), int(np.argmax(y))}
    diag = np.zeros(mol.nao_nr(), complex)
    for i, (atom, _, shell, ml) in enumerate(mol.ao_labels(fmt=False)):
        if atom not in outer:
            continue
        if shell == "2s":                               # exponent 0.327 in def2-TZVP
            diag[i] = p["eps_s"] - 1j * p["eta_s"]
        elif "eps_p" in p and shell == "2p" and ml in ("x", "y"):
            diag[i] = p["eps_p"] - 1j * p["eta_p"]
    if not np.any(diag):
        raise RuntimeError("no coupled AOs found")
    return np.diag(diag), np.flatnonzero(diag)


def gc_point(mol, Sigma, mu, r, dm0, q0_dm):
    def once(seed, extra):
        mf = dft.RKS(mol); mf.xc, mf.grids.level = XC, GRID_LEVEL
        out = scf_gc(mol, Sigma, mu=float(mu), logger=None, verbose=False, mf=with_charge(mf, r),
                     dm_init=seed, **(GC_SCF | extra))
        E, Ne = float(np.real(out[4])), float(out[1])
        return {"P": np.real(out[0]), "Ne": Ne, "Omega": E - mu * Ne, "converged": bool(out[5])}
    res, tries = once(dm0, {}), 0
    for seed, extra in RETRY:
        if res["converged"]:
            break
        tries += 1
        res = once(q0_dm if seed == "q0" else None, extra)
    res["retries"] = tries
    return res


def md(**extra):
    m = metadata(**extra)
    for k in ("conv_tol", "conv_tol_omega", "diis_space", "anderson_mix"):   # MINAO GC settings, not used here
        if k not in extra:
            m.pop(k, None)
    m["basis"] = BASIS
    return m


def save_curve(name, rows, extra):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "curve.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    with (d / "curve.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (d / "metadata.json").write_text(json.dumps(md(**extra), indent=2))


def run_refs(names):
    for name in names:
        mol, t0 = build_mol(REFS[name]), time.time()
        mf0 = canonical(mol)
        E0, dm, rows = float(mf0.e_tot), mf0.make_rdm1(), []
        for r in read_r_grid():
            mf = canonical(mol, r, dm)
            dm = mf.make_rdm1()
            rows.append({"r_ang": r, "E_tot": float(mf.e_tot), "DeltaE": float(mf.e_tot) - E0,
                         "DeltaE_kcal": (float(mf.e_tot) - E0) * HA_TO_KCAL, "converged": bool(mf.converged)})
        n = sum(x["converged"] for x in rows)
        print(f"{name} (canonical): {n}/{len(rows)} converged ({time.time() - t0:.0f} s)", flush=True)
        save_curve(name, rows, dict(script="run_def2tzvp.py --part refs", system=name, xyz=REFS[name],
                                    method="canonical RKS", scf=REF_SCF, E_q0=E0, q0_converged=bool(mf0.converged),
                                    n_converged=n, warm_start="previous point, increasing r"))


def run_gc(names):
    mol = build_mol(FRAG_XYZ)
    r_grid = read_r_grid()
    for name in names:
        p, t0 = COUPLED[name], time.time()
        Sigma, aos = coupling_sigma(mol, p)
        q0 = gc_point(mol, Sigma, p["mu"], None, None, None)
        rows, dm = [None] * len(r_grid), q0["P"]
        for i in range(len(r_grid) - 1, -1, -1):            # from r = 100 A inwards
            res = gc_point(mol, Sigma, p["mu"], r_grid[i], dm, q0["P"])
            dm = res["P"]
            d = res["Omega"] - q0["Omega"]
            rows[i] = {"r_ang": r_grid[i], "Omega": res["Omega"], "Ne": res["Ne"], "DeltaOmega": d,
                       "DeltaOmega_kcal": d * HA_TO_KCAL, "converged": res["converged"], "retries": res["retries"]}
        n = sum(x["converged"] for x in rows)
        print(f"{name}: q0 Ne={q0['Ne']:.6f}; {n}/{len(rows)} converged ({time.time() - t0:.0f} s)", flush=True)
        save_curve(name, rows, dict(script="run_def2tzvp.py --part gc", system=name, xyz=FRAG_XYZ, **p,
                                    coupled_aos=aos.tolist(), gc_scf=GC_SCF, retry=RETRY,
                                    Omega_q0=q0["Omega"], Ne_q0=q0["Ne"], q0_converged=q0["converged"],
                                    n_converged=n, warm_start="previous point, decreasing r"))


def run_density():
    d = OUT / "density"; d.mkdir(parents=True, exist_ok=True)
    ring, frag = build_mol(REFS["H40"]), build_mol(FRAG_XYZ)
    info = {}
    for r in DENSITY_R:
        tag = f"r{r:.4f}"
        mf = canonical(ring, r)
        np.save(d / f"dm_H40__{tag}.npy", mf.make_rdm1()); info[f"H40__{tag}"] = bool(mf.converged)
        for name, p in COUPLED.items():
            Sigma, _ = coupling_sigma(frag, p)
            q0 = gc_point(frag, Sigma, p["mu"], None, None, None)
            res = gc_point(frag, Sigma, p["mu"], r, q0["P"], q0["P"])
            np.save(d / f"dm_{name}__{tag}.npy", res["P"]); info[f"{name}__{tag}"] = res["converged"]
        print(f"density r = {r:.4f} A: {info}", flush=True)
    (d / "metadata.json").write_text(json.dumps(md(
        script="run_def2tzvp.py --part density", r_values=list(DENSITY_R), converged=info), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["refs", "gc", "density", "all"], default="all")
    ap.add_argument("--system", help="only this reference (H40/H38/H6) or coupled run (H6_s/H6_sp)")
    a = ap.parse_args()
    if a.part in ("refs", "all"):
        run_refs([a.system] if a.system in REFS else list(REFS))
    if a.part in ("gc", "all"):
        run_gc([a.system] if a.system in COUPLED else list(COUPLED))
    if a.part in ("density", "all"):
        run_density()


if __name__ == "__main__":
    main()
