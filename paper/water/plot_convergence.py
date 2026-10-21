#!/usr/bin/env python3
"""
Plot the SCF convergence of the water calculations (Figs. S2-S5):
|Delta Omega|, ||Delta P|| and N_e vs. iteration for each mixing scheme
and broadening.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# plotting_utils.py-style font fallback: Arial isn't installed (no sudo in
# this environment), Liberation Sans is metric-compatible and installed.
plt.rcParams.update({"font.sans-serif": ["Arial", "Liberation Sans"]})


# ============================================================
# User settings
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = THIS_DIR / "results" / "calculations"
OUTPUT_DIR = THIS_DIR / "results" / "figures" / "convergence"
GAMMAS = [1e-6, 5e-3, 1e-1]

THEORIES = ["rhf", "rks_pbe"]
BASES = ["minao", "sto-3g"]
STARTS = ["sad", "canonical_same_theory"]

METHODS = [
    "CDIIS",
    "ANDERSON",
    "EDIIS_OMEGA",
    "HYBRID",
]

LABELS = {
    "CDIIS": "CDIIS",
    "ANDERSON": "Anderson",
    "EDIIS_OMEGA": r"EDIIS($\Omega$)",
    "HYBRID": r"EDIIS($\Omega$)+A",
}

THRESH_DOMEGA = 1.0e-8
THRESH_DP = 1.0e-6
# Convergence thresholds (SCFSettings defaults, used for all water runs).
MAX_ITER = 200
SET_XLIM = False

# ============================================================
# Plot defaults
# ============================================================

cm = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "figure.dpi": 300,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

styles = {
    "CDIIS": dict(color="#015895", ls="-", lw=1.6),
    "ANDERSON": dict(color="#ff8400", ls="-", lw=1.8),
    "EDIIS_OMEGA": dict(color="#00af00", ls="-", lw=1.8),
    "HYBRID": dict(color="#e82626", ls="--", lw=2.0),
}

# ============================================================
# Helpers
# ============================================================

def gamma_tag(gamma: float) -> str:
    return f"{gamma:.0e}".replace("+", "").replace(".", "p")


def method_tag(method: str) -> str:
    return method.lower()


def load_json(path: Path, *, retries: int = 6, delay: float = 5.0):
    # Occasionally hits a transient timeout reading from this directory
    # shortly after startup; retry with backoff rather than fail outright.
    last_exc = None
    for attempt in range(retries):
        try:
            with path.open("r") as f:
                return json.load(f)
        except (TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_exc


def find_calc_dir(*, theory_name: str, basis: str, start: str, gamma: float, method: str) -> Path | None:
    calc_id = (
        f"water__{theory_name}"
        f"__basis-{basis}"
        f"__gamma-{gamma_tag(gamma)}"
        f"__method-{method_tag(method)}"
        f"__start-{start}"
    )

    path = RESULTS_ROOT / calc_id

    if not path.exists():
        print(f"WARNING: missing result directory: {path}")
        return None

    return path


def _finite_xy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _as_float_array(x):
    return np.asarray(x, float)


def _delta_abs(arr):
    a = _as_float_array(arr)
    if a.size == 0:
        return a
    out = np.full_like(a, np.nan, dtype=float)
    out[1:] = np.abs(a[1:] - a[:-1])
    return out


def _rowwise_ylim(values_list, yscale, pad_frac=0.05):
    vals = []
    for v in values_list:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)

    if not vals:
        return None

    allv = np.concatenate(vals)
    vmin = float(np.nanmin(allv))
    vmax = float(np.nanmax(allv))

    if yscale == "log":
        vmin = max(vmin, 1.0e-16)
        vmax = max(vmax, vmin * 1.01)
        lo = vmin / (1.0 + pad_frac)
        hi = vmax * (1.0 + pad_frac)
        return lo, hi

    span = vmax - vmin
    if span < 1.0e-14:
        span = 1.0

    lo = vmin - pad_frac * span
    hi = vmax + pad_frac * span
    return lo, hi


def _sci_tex(x: float) -> str:
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0 ** exp)
    if np.isclose(mant, 1.0):
        return rf"10^{{{exp:d}}}"
    return rf"{mant:.1f}\times 10^{{{exp:d}}}"


def plot_convergence_grid(
    *,
    logs,
    summaries,
    gammas,
    methods,
    labels,
    out_png,
    thresh_domega=1.0e-8,
    thresh_dp=1.0e-6,
):
    if len(gammas) != 3:
        raise ValueError(f"Expected exactly 3 gammas, got {len(gammas)}")

    rows = [
        (r"$|\Delta \Omega|$ (Ha)", "Omega", _delta_abs, "log"),
        (r"$\|\Delta \mathbf{P}\|$", "dP", _as_float_array, "log"),
        (r"$N_e$", "N_e", _as_float_array, "linear"),
    ]

    fig_w = 11.33 * cm
    fig_h = 11.33 * cm

    fig, axs = plt.subplots(
        nrows=len(rows),
        ncols=3,
        sharex="col",
        figsize=(fig_w, fig_h),
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
        dpi=300,
    )

    axs = np.atleast_2d(axs)

    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.12, top=0.93)

    transformed = {}
    row_vals = {i: [] for i in range(len(rows))}

    for gamma in gammas:
        for method in methods:
            log = logs.get((gamma, method))
            if log is None:
                continue

            it = np.asarray(log.get("iter", []), int)

            for i, (_ylabel, key, transform, yscale) in enumerate(rows):
                y = transform(log.get(key, []))

                if yscale == "log":
                    y = np.asarray(y, float)
                    y = np.where(np.isfinite(y), y, np.nan)
                    y = np.maximum(y, 1.0e-16)

                transformed[(i, gamma, method)] = (it, y)
                row_vals[i].append(y)

    row_ylims = []
    for i, (_ylabel, _key, _transform, yscale) in enumerate(rows):
        row_ylims.append(_rowwise_ylim(row_vals[i], yscale))

    for j, gamma in enumerate(gammas):
        axs[0, j].set_title(rf"$\eta = {_sci_tex(gamma)}$ Ha")

        for i, (ylabel, keyname, _transform, yscale) in enumerate(rows):
            ax = axs[i, j]

            ax.grid(ls=":", alpha=0.5)
            ax.set_yscale(yscale)

            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            if i == len(rows) - 1:
                ax.set_xlabel("SCF iteration")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if row_ylims[i] is not None:
                ax.set_ylim(*row_ylims[i])

            y0, _y1 = ax.get_ylim()

            if keyname == "Omega":
                thr = max(float(thresh_domega), 1.0e-16)
                if y0 < thr:
                    ax.axhspan(y0, thr, facecolor="0.9", alpha=0.85, zorder=0)
            elif keyname == "dP":
                thr = max(float(thresh_dp), 1.0e-16)
                if y0 < thr:
                    ax.axhspan(y0, thr, facecolor="0.9", alpha=0.85, zorder=0)

            for method in methods:
                key = (i, gamma, method)
                if key not in transformed:
                    continue

                it, y = transformed[key]
                x, yy = _finite_xy(it, y)
                if x.size == 0:
                    continue

                summary = summaries.get((gamma, method), {})
                converged = bool(summary.get("converged", False))

                style = styles.get(method, {}).copy()
                color = style.get("color", "k")
                alpha = 1.0 if converged else 0.35

                ax.plot(x, yy, label=labels.get(method, method), alpha=alpha, **style)

                marker = "o" if converged else "x"
                ax.plot(
                    x[-1], yy[-1],
                    marker=marker,
                    markersize=2.5 if converged else 5.5,
                    markeredgewidth=1.4,
                    color=color,
                    linestyle="None",
                    zorder=10,
                )

                if SET_XLIM:
                    ax.set_xlim(0, MAX_ITER)

    handles, leglabels = axs[-1, 0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles, leglabels,
            loc="lower left", ncol=2, frameon=True, fancybox=False,
            framealpha=1.0, borderpad=0.3, handlelength=2.2,
            columnspacing=1.2, handletextpad=0.6, fontsize=8,
            bbox_to_anchor=(0.16, 0.16),
        )
        fig.subplots_adjust(bottom=0.16)

    fig.set_size_inches(fig_w, fig_h, forward=True)
    fig.savefig(out_png, dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def make_one(theory_name: str, basis: str, start: str):
    logs = {}
    summaries = {}

    for gamma in GAMMAS:
        for method in METHODS:
            calc_dir = find_calc_dir(
                theory_name=theory_name, basis=basis, start=start,
                gamma=gamma, method=method,
            )
            if calc_dir is None:
                print(f"  WARNING: skipping missing calculation: gamma={gamma:.1e}, method={method}")
                continue

            log_path = calc_dir / "log.json"
            summary_path = calc_dir / "summary.json"

            if not log_path.exists():
                print(f"  WARNING: missing log file: {log_path}")
                continue

            if not summary_path.exists():
                print(f"  WARNING: missing summary file: {summary_path}")
                summary = {}
            else:
                summary = load_json(summary_path)

            if summary and not bool(summary.get("converged", False)):
                print(f"  WARNING: calculation did not converge: gamma={gamma:.1e}, method={method}, "
                      f"basis={basis}, theory={theory_name}, start={start}")

            log = load_json(log_path)

            if len(log.get("iter", [])) == 0:
                print(f"  WARNING: empty iteration log: gamma={gamma:.1e}, method={method}")

            logs[(gamma, method)] = log
            summaries[(gamma, method)] = summary

    if not logs:
        print(f"  SKIPPING {theory_name}/{basis}/{start}: no data found.")
        return

    out_png = OUTPUT_DIR / f"water_convergence__{theory_name}__basis-{basis}__start-{start}.png"

    plot_convergence_grid(
        logs=logs, summaries=summaries, gammas=GAMMAS, methods=METHODS,
        labels=LABELS, out_png=out_png,
        thresh_domega=THRESH_DOMEGA, thresh_dp=THRESH_DP,
    )
    print(f"Saved: {out_png}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for theory_name in THEORIES:
        for basis in BASES:
            for start in STARTS:
                print(f"=== {theory_name} / {basis} / {start} ===")
                make_one(theory_name, basis, start)


if __name__ == "__main__":
    main()
