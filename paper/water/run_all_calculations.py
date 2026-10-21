#!/usr/bin/env python3
"""
Run all water GC-SCF configs written by gen_configs.py (Tables S1-S4,
Figs. S2-S5). Results go to results/calculations/<calc_id>/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from pyscf import gto, scf, dft

from gcec import GCEC, SCFSettings
from gcec.scf_log import init_log

from helpers import make_sigma_ao_diagonal, mu_from_midgap


# ============================================================
# Paths / user settings
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

CONFIG_DIR = THIS_DIR / "configs"

OVERWRITE = False


# ============================================================
# Helpers
# ============================================================

def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_json(obj, path: Path):
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=THIS_DIR,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def read_xyz_atoms(path: Path) -> str:
    lines = path.read_text().splitlines()
    natoms = int(lines[0])
    atom_lines = lines[2:2 + natoms]
    return "\n".join(atom_lines)


def build_mol(cfg: dict):
    system = cfg["system"]
    geom = system["geometry"]

    if geom["type"] != "xyz":
        raise ValueError(f"Only geometry.type='xyz' supported, got {geom['type']}")

    xyz_path = THIS_DIR / geom["file"]   # relative to paper/water/

    mol = gto.Mole()
    mol.atom = read_xyz_atoms(xyz_path)
    mol.unit = geom.get("units", "angstrom")
    mol.basis = system["basis"]
    mol.charge = int(system["charge"])
    mol.spin = int(system["spin"])
    mol.build()

    return mol


def make_pyscf_mf(mol, theory: str, xc: str | None):
    theory = theory.upper()

    if theory == "RHF":
        return scf.RHF(mol)

    if theory == "RKS":
        mf = dft.RKS(mol)
        if xc is None:
            raise ValueError("RKS requires xc in config.")
        mf.xc = xc
        return mf

    raise ValueError(f"Unknown theory: {theory}")


def build_canonical_dm(mol, cfg: dict):
    dm_init_cfg = cfg["scf"]["dm_init"]

    if dm_init_cfg is None:
        return None

    mode = dm_init_cfg["mode"]

    if mode != "canonical_same_theory":
        raise ValueError(f"Unknown dm_init mode: {mode}")

    theory = cfg["electronic_structure"]["theory"]
    xc = cfg["electronic_structure"]["xc"]

    mf = make_pyscf_mf(mol, theory, xc)
    mf.kernel()

    if not mf.converged:
        raise RuntimeError("Canonical reference calculation did not converge.")

    return mf.make_rdm1()


def water_sigma_idxs(mol):
    labels = mol.ao_labels()
    idxs = []

    for i, lab in enumerate(labels):
        if " O " in lab and (" 2s" in lab or " 2p" in lab):
            idxs.append(i)
        if " H " in lab and " 1s" in lab:
            idxs.append(i)

    return sorted(set(idxs)), labels


def fps_residual(mol, F, Sigma_ao, P):
    S = mol.intor("int1e_ovlp")
    Heff = np.asarray(F, complex) + np.asarray(Sigma_ao, complex)
    P = np.asarray(P, complex)

    R = Heff @ P @ S - S @ P @ Heff

    return {
        "fro": float(np.linalg.norm(R)),
        "max_abs": float(np.max(np.abs(R))),
    }


def result_to_npz_dict(res: dict):
    out = {}

    for key in [
        "P",
        "F",
        "mo_energy",
        "mo_occ",
        "mo_occ_gc",
        "mo_gamma",      # new: level broadening gamma_k = -Im Z_kk
        "mo_coeff",
    ]:
        if key in res:
            out[key] = np.asarray(res[key])

    for key in [
        "E_tot",
        "Omega",
        "Ne",
        "converged",
    ]:
        if key in res:
            out[key] = np.asarray(res[key])

    return out


def log_to_serializable(log: dict):
    clean = {}

    for key, value in log.items():
        arr = np.asarray(value)

        if arr.ndim == 0:
            clean[key] = arr.item()
        else:
            clean[key] = arr.tolist()

    return clean


# ============================================================
# One calculation
# ============================================================

def run_one_config(config_path: Path):
    cfg = load_yaml(config_path)

    calc_id = cfg["calculation"]["id"]
    out_root = THIS_DIR / cfg["outputs"]["root"]   # relative to paper/water/
    out_dir = out_root / calc_id

    if out_dir.exists() and not OVERWRITE:
        print(f"skip existing: {calc_id}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(calc_id)
    print("=" * 80)

    mol = build_mol(cfg)

    theory = cfg["electronic_structure"]["theory"]
    xc = cfg["electronic_structure"]["xc"]

    mf = make_pyscf_mf(mol, theory, xc)

    mu_mode = cfg["electronic_structure"]["mu"]["mode"]
    if mu_mode != "midgap":
        raise ValueError(f"Only mu.mode='midgap' supported, got {mu_mode}")

    mu_ref = mu_from_midgap(mol)

    sigma_cfg = cfg["grand_canonical"]["sigma"]
    gamma = float(sigma_cfg["gamma"])

    if sigma_cfg["coupled_aos"]["mode"] == "water_valence":
        idxs, labels = water_sigma_idxs(mol)
    else:
        raise ValueError(f"Unknown coupled AO mode: {sigma_cfg['coupled_aos']['mode']}")

    Sigma_ao, idxs_used = make_sigma_ao_diagonal(mol, idxs, gamma)

    dm_init = build_canonical_dm(mol, cfg)

    scf_cfg = cfg["scf"]

    settings = SCFSettings(
        method=scf_cfg["method"],
        max_cycle=int(scf_cfg["max_cycle"]),
        diis_space=int(scf_cfg["diis_space"]),
        theory=theory,
        verbose=bool(scf_cfg["verbose"]),
        guess=scf_cfg["guess"],
    )

    engine = GCEC(
        mol,
        Sigma_ao=Sigma_ao,
        mu=mu_ref,
        settings=settings,
    )

    log = init_log()

    res = engine.run_scf(
        logger=log,
        dm_init=dm_init,
    )

    fps = None
    if cfg["analysis"].get("compute_fps_residual", False):
        if bool(res.get("converged", False)):
            fps = fps_residual(mol, res["F"], Sigma_ao, res["P"])

    # ----------------------------
    # Save full result
    # ----------------------------
    npz_dict = result_to_npz_dict(res)
    npz_dict["Sigma_ao"] = np.asarray(Sigma_ao)
    npz_dict["sigma_idxs"] = np.asarray(idxs_used, dtype=int)
    npz_dict["mu_ref"] = np.asarray(mu_ref)
    npz_dict["gamma"] = np.asarray(gamma)

    np.savez_compressed(out_dir / "result.npz", **npz_dict)

    # ----------------------------
    # Save summary
    # ----------------------------
    summary = {
        "calculation_id": calc_id,
        "config_file": config_path.name,
        "basis": cfg["system"]["basis"],
        "theory": theory,
        "xc": xc,
        "gamma": gamma,
        "method": scf_cfg["method"],
        "start": cfg["scf"]["dm_init"]["mode"] if cfg["scf"]["dm_init"] else "sad",
        "mu_ref": mu_ref,
        "converged": bool(res.get("converged", False)),
        "E_tot": float(res.get("E_tot", np.nan)),
        "Omega": float(res.get("Omega", np.nan)),
        "Ne": float(res.get("Ne", np.nan)),
        "fps_residual": fps,
        "n_sigma_ao": int(len(idxs_used)),
        "sigma_ao_labels": [labels[i] for i in idxs_used],
    }

    save_json(summary, out_dir / "summary.json")

    # ----------------------------
    # Save log
    # ----------------------------
    save_json(log_to_serializable(res["log"]), out_dir / "log.json")

    # ----------------------------
    # Save config and metadata
    # ----------------------------
    shutil.copy2(config_path, out_dir / "resolved_config.yaml")

    metadata = {
        "created": datetime.now().isoformat(),
        "python": sys.version,
        "git_commit": get_git_commit(),
        "script": Path(__file__).name,
        "config": config_path.name,
    }

    save_json(metadata, out_dir / "metadata.json")

    print(
        f"done: converged={summary['converged']}  "
        f"Ne={summary['Ne']:.6f}  "
        f"Omega={summary['Omega']:.8f}"
    )


# ============================================================
# Main
# ============================================================

def main():
    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))

    if not config_paths:
        raise FileNotFoundError(f"No YAML files found in {CONFIG_DIR}")

    print(f"Found {len(config_paths)} configs.")

    for path in config_paths:
        run_one_config(path)

    print()
    print("All calculations finished.")


if __name__ == "__main__":
    main()