#!/usr/bin/env python3
"""
Write the YAML configs for the water GC-SCF calculations (Tables S1-S4,
Figs. S2-S5): all combinations of basis, level of theory, broadening,
mixing scheme and initial guess.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = THIS_DIR / "configs"
RESULTS_ROOT = THIS_DIR / "results" / "calculations"

XYZ_FILE = THIS_DIR / "water.xyz"


# ============================================================
# User settings
# ============================================================

OVERWRITE = False

GEOMETRY_UNITS = "angstrom"

BASIS_SETS = [
    "minao",
    "sto-3g",
]

LEVELS_OF_THEORY = [
    {
        "name": "RHF",
        "theory": "RHF",
        "xc": None,
    },

    {
        "name": "RKS_PBE",
        "theory": "RKS",
        "xc": "pbe",
    },

]

GAMMAS = [1.0e-6, 5.0e-3, 1.0e-1]

SCF_METHODS = [
    "EDIIS_E",
    "CDIIS",
    "ANDERSON",
    "EDIIS_OMEGA",
    "HYBRID",
]

START_MODES = [
    {
        "name": "sad",
        "guess": "sad",
        "dm_init": None,
    },

    {
        "name": "canonical_same_theory",
        "guess": "sad",

        "dm_init": {
            "mode": "canonical_same_theory",
        },
    },
]

SCF_MAX_CYCLE = 200
SCF_DIIS_SPACE = 12

SCF_CONV_TOL = 1.0e-6
SCF_CONV_TOL_OMEGA = 1.0e-8

SCF_VERBOSE = False


# ============================================================
# Helpers
# ============================================================

def gamma_tag(gamma: float) -> str:
    return f"{gamma:.0e}".replace("+", "").replace(".", "p")


def make_config(
    *,
    basis,
    level,
    gamma,
    method,
    start_mode,
):
    theory_name = level["name"]
    theory = level["theory"]
    xc = level["xc"]

    start_name = start_mode["name"]

    calc_id = (
        f"water__{theory_name.lower()}"
        f"__basis-{basis}"
        f"__gamma-{gamma_tag(gamma)}"
        f"__method-{method.lower()}"
        f"__start-{start_name}"
    )

    description = (
        f"Water GC-SCF calculation: "
        f"{theory_name}, "
        f"basis={basis}, "
        f"gamma={gamma:.6e} Ha, "
        f"method={method}, "
        f"start={start_name}."
    )

    scf_cfg = {
        "method": method,

        "max_cycle": SCF_MAX_CYCLE,
        "diis_space": SCF_DIIS_SPACE,

        "conv_tol": SCF_CONV_TOL,
        "conv_tol_omega": SCF_CONV_TOL_OMEGA,

        "verbose": SCF_VERBOSE,

        "guess": start_mode["guess"],

        "dm_init": start_mode["dm_init"],
    }

    cfg = {
        "calculation": {
            "id": calc_id,
            "description": description,
        },

        "system": {
            "name": "water",

            "charge": 0,
            "spin": 0,

            "geometry": {
                "type": "xyz",
                "file": XYZ_FILE.name,  # relative to paper/water/
                "units": GEOMETRY_UNITS,
            },

            "basis": basis,
        },

        "electronic_structure": {
            "package": "pyscf",

            "theory": theory,
            "xc": xc,

            "mu": {
                "mode": "midgap",
            },
        },

        "grand_canonical": {
            "sigma": {
                "type": "ao_diagonal",

                "gamma": float(gamma),

                "coupled_aos": {
                    "mode": "water_valence",

                    "oxygen_orbitals": [
                        "2s",
                        "2p",
                    ],

                    "hydrogen_orbitals": [
                        "1s",
                    ],
                },
            },
        },

        "scf": scf_cfg,

        "analysis": {
            "compute_fps_residual": True,

            "save_log": True,

            "save_density_matrix": True,
            "save_fock_matrix": True,

            "save_mo_coeff": True,
            "save_mo_energy": True,
            "save_mo_occupation": True,
        },

        "outputs": {
            "root": str(RESULTS_ROOT.relative_to(THIS_DIR)),  # relative to paper/water/

            "save_result_npz": True,
            "save_summary_json": True,
            "save_log_json": True,
            "save_metadata_json": True,
            "save_resolved_config": True,
        },
    }

    return cfg


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_written = 0
    n_skipped = 0

    for (
        basis,
        level,
        gamma,
        method,
        start_mode,
    ) in product(
        BASIS_SETS,
        LEVELS_OF_THEORY,
        GAMMAS,
        SCF_METHODS,
        START_MODES,
    ):
        cfg = make_config(
            basis=basis,
            level=level,
            gamma=gamma,
            method=method,
            start_mode=start_mode,
        )

        calc_id = cfg["calculation"]["id"]

        outpath = OUTPUT_DIR / f"{calc_id}.yaml"

        if outpath.exists() and not OVERWRITE:
            print(f"skip existing: {outpath}")
            n_skipped += 1
            continue

        with outpath.open("w") as f:
            yaml.safe_dump(
                cfg,
                f,
                sort_keys=False,
            )

        print(f"wrote: {outpath}")

        n_written += 1

    print()
    print(f"Written : {n_written}")
    print(f"Skipped : {n_skipped}")
    print(f"Output  : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
