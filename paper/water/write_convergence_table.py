#!/usr/bin/env python3
"""
Write the LaTeX table of final GC-SCF results for water (Tables S1-S4):
Omega, N_e, number of iterations and FPS residual. One table is written
per (level of theory, basis), as LaTeX (.tex) and as plain numbers (.csv).

The iteration count is the length of the SCF log, which includes the
initial guess (iteration 0).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# User settings
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = THIS_DIR / "results" / "calculations"
OUTPUT_DIR = THIS_DIR / "results" / "tables"
GAMMAS = [1.0e-6, 5.0e-3, 1.0e-1]
TABLE_LABEL = "tab:water_scf_results"

BASIS = "sto-3g"       # minao, sto-3g
THEORY_NAME = "rhf"    # rhf, rks_pbe

METHODS = [
    "EDIIS_E",
    "CDIIS",
    "ANDERSON",
    "EDIIS_OMEGA",
    "HYBRID",
]

LABELS = {
    "EDIIS_E": "EDIIS(E)",
    "CDIIS": "CDIIS",
    "ANDERSON": "Anderson",
    "EDIIS_OMEGA": r"EDIIS($\Omega$)",
    "HYBRID": r"EDIIS($\Omega$)+A",
}

START_COLUMNS = [
    {
        "name": "sad",
        "title": "SAD guess",
    },
    {
        "name": "canonical_same_theory",
        "title": r"$\boldsymbol{\Sigma}=0$ start",
    },
]


# ============================================================
# Helpers
# ============================================================

def gamma_tag(gamma: float) -> str:
    return f"{gamma:.0e}".replace("+", "").replace(".", "p")


def method_tag(method: str) -> str:
    return method.lower()


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def find_calc_dir(*, gamma: float, method: str, start: str) -> Path:
    calc_id = (
        f"water__{THEORY_NAME}"
        f"__basis-{BASIS}"
        f"__gamma-{gamma_tag(gamma)}"
        f"__method-{method_tag(method)}"
        f"__start-{start}"
    )

    path = RESULTS_ROOT / calc_id

    if not path.exists():
        raise FileNotFoundError(f"Missing result directory:\n{path}")

    return path


def load_summary(*, gamma: float, method: str, start: str) -> dict:
    calc_dir = find_calc_dir(gamma=gamma, method=method, start=start)
    path = calc_dir / "summary.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing summary file:\n{path}")

    return load_json(path)


def load_n_iter(*, gamma: float, method: str, start: str) -> int | None:
    try:
        calc_dir = find_calc_dir(gamma=gamma, method=method, start=start)
        log = load_json(calc_dir / "log.json")
        return len(log["iter"])
    except Exception:
        return None


def fmt_eta(eta: float) -> str:
    if np.isclose(eta, 1.0e-10):
        return r"1\times10^{-10}"
    if np.isclose(eta, 1.0e-6):
        return r"1\times10^{-6}"
    if np.isclose(eta, 5.0e-3):
        return r"5\times10^{-3}"
    if np.isclose(eta, 1.0e-1):
        return r"1\times10^{-1}"

    exp = int(np.floor(np.log10(abs(eta))))
    mant = eta / 10**exp

    return rf"{mant:.1f}\times10^{{{exp}}}"


def fmt_number(x, ndigits: int = 5) -> str:
    try:
        x = float(x)
    except Exception:
        return "--"

    if not np.isfinite(x):
        return "--"

    return f"{x:.{ndigits}f}"


def fmt_resid(summary: dict) -> str:
    resid = summary.get("fps_residual", None)

    if resid is None:
        return "--"

    x = resid.get("fro", np.nan)

    try:
        x = float(x)
    except Exception:
        return "--"

    if not np.isfinite(x) or x == 0.0:
        return "--"

    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10**exp

    return rf"${mant:.2f}\times10^{{{exp}}}$"


def theory_title(theory_name: str) -> str:
    mapping = {
        "rhf": "RHF",
        "rks_pbe": "RKS/PBE",
        "rks_pbe0": "RKS/PBE0",
    }
    return mapping.get(theory_name, theory_name.upper())


def convergence_mark(summary: dict) -> str:
    return "" if summary.get("converged", False) else r"$^\dagger$"


# ============================================================
# Table writer
# ============================================================

def write_table(*, out_tex: Path):
    summaries = {}
    n_iters = {}

    for gamma in GAMMAS:
        for method in METHODS:
            for start_col in START_COLUMNS:
                start = start_col["name"]
                try:
                    summaries[(gamma, method, start)] = load_summary(
                        gamma=gamma, method=method, start=start,
                    )
                    n_iters[(gamma, method, start)] = load_n_iter(
                        gamma=gamma, method=method, start=start,
                    )
                except FileNotFoundError:
                    summaries[(gamma, method, start)] = None
                    n_iters[(gamma, method, start)] = None

    first_summary = next((s for s in summaries.values() if s is not None), None)
    if first_summary is None:
        print(f"  SKIPPING {THEORY_NAME}/{BASIS}: no data found.")
        return
    mu_ref = first_summary.get("mu_ref", np.nan)

    ncols = 1 + 4 * len(START_COLUMNS)

    lines = []

    lines.append(r"\begin{table}")
    lines.append(
        rf"\caption{{Final grand-canonical SCF results for the water molecule "
        rf"using the \textbf{{{BASIS.upper()}}} basis at \textbf{{{theory_title(THEORY_NAME)}}} level. "
        r"Reported are the grand potential $\Omega$, the number of electrons $N_e$, "
        r"and the Frobenius norm of the FPS residual $\|\mathbf{R}\|_F$. "
        r"Residuals are reported only for converged calculations; "
        r"$\dagger$ marks a run that did not meet the convergence criterion. "
        rf"The reference chemical potential is $\mu_{{\mathrm{{ref}}}} = {float(mu_ref):.5f}~\mathrm{{Ha}}$.}}"
    )
    lines.append(rf"\label{{{TABLE_LABEL}}}")
    lines.append(r"\centering")

    colspec = "l" + "|cccc" * len(START_COLUMNS)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\hline")

    header = ""
    for i, start_col in enumerate(START_COLUMNS):
        suffix = "|" if i < len(START_COLUMNS) - 1 else ""
        header += rf"& \multicolumn{{4}}{{c{suffix}}}{{{start_col['title']}}} "

    lines.append(header + r"\\")
    lines.append(
        r"SCF scheme "
        + "".join(
            r"& $\Omega$ (Ha) & $N_e$ & iter & $\|\mathbf{R}\|_F$ "
            for _ in START_COLUMNS
        )
        + r"\\"
    )
    lines.append(r"\hline")

    for gamma in GAMMAS:
        lines.append(
            rf"\multicolumn{{{ncols}}}{{c}}{{$\eta = {fmt_eta(gamma)}$ Ha}} \\"
        )
        lines.append(r"\hline")

        for method in METHODS:
            row = LABELS.get(method, method)

            for start_col in START_COLUMNS:
                start = start_col["name"]
                summary = summaries[(gamma, method, start)]
                n_iter = n_iters[(gamma, method, start)]

                if summary is None:
                    row += r" & -- & -- & -- & --"
                    continue

                iter_str = str(n_iter) if n_iter is not None else "--"
                mark = convergence_mark(summary)

                row += (
                    rf" & {fmt_number(summary.get('Omega', np.nan))}{mark}"
                    rf" & {fmt_number(summary.get('Ne', np.nan))}"
                    rf" & {iter_str}"
                    rf" & {fmt_resid(summary)}"
                )

            row += r" \\"
            lines.append(row)

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_tex.write_text("\n".join(lines) + "\n")

    # same numbers as plain CSV (full precision)
    with out_tex.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["theory", "basis", "eta_Ha", "method", "start", "Omega_Ha", "Ne",
                    "n_iter", "converged", "fps_residual_fro"])
        for (gamma, method, start), summary in summaries.items():
            if summary is None:
                continue
            resid = (summary.get("fps_residual") or {}).get("fro")
            w.writerow([THEORY_NAME, BASIS, gamma, method, start, summary.get("Omega"),
                        summary.get("Ne"), n_iters[(gamma, method, start)],
                        bool(summary.get("converged", False)), resid])


# ============================================================
# Main
# ============================================================

def main():
    global BASIS, THEORY_NAME

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for theory_name in ["rhf", "rks_pbe"]:
        for basis in ["minao", "sto-3g"]:
            THEORY_NAME, BASIS = theory_name, basis

            out_tex = (
                OUTPUT_DIR
                / f"water_table__{THEORY_NAME}__basis-{BASIS}.tex"
            )
            write_table(out_tex=out_tex)
            print(f"Saved: {out_tex} (+ .csv)")


if __name__ == "__main__":
    main()
