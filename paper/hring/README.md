# Hydrogen ring (paper Sec. IV.B, Figs. 3–5; SI Sec. S3)

PBE0/MINAO, grid level 3. Reference: H40 ring (bond length 0.74 Å,
`xyz/ring_full.xyz`). Fragments H6–H20 (`xyz/ring_<n>uc.xyz`, n = 3…10) are
cut from the same ring; the two outer H atoms (smallest/largest y) are coupled
to the reservoir with Σ = ε − iη. A point charge q = +1 approaches along the
x-axis at (x0 + r, 0, 0), x0 = 4.701296 Å, on the 68 r values of the H40
reference scan (`r_grid.txt`).

## Scripts

| Script | What it does | Output |
|---|---|---|
| `run_fragment_curves.py` | ΔΩ(r) and N_e(r) of each fragment at the **published** (ε, η, μ) | `results/fragment_curves/<fragment>/` |
| `run_density_comparison.py` | densities of fragment and local H40 at r_min = 1.218 Å, and their difference (z = 0 plane) | `results/density_comparison/<system>/` |
| `plot_density_comparison.py` | Fig. 3d (H6), SI Figs. S10b (H6_alt) and S12 | `figures/density_comparison_<system>.png` |
| `plot_curves.py` | Fig. 3c (H6), SI Fig. S10a (`H6_alt`), SI Fig. S11 (`--overlay`) | `figures/curve_<label>.png`, `figures/curves_overlay.png` |
| `run_mu_sweep.py`, `plot_mu_sweep.py` | Fig. 3b: H6 levels and occupations vs μ (ε = 0, η = 0.1 Ha) | `results/mu_sweep/`, `figures/mu_sweep_H6.png` |
| `run_electronic_structure.py` | canonical gaps, Δ_μ, T_eff with and without the point charge (SI Table S6) | `results/electronic_structure/gaps_teff.csv`, `gaps_teff_table.tex` |
| `plot_electronic_structure.py` | Fig. 4a, Fig. 4b, SI Fig. S13 | `figures/fig4a_parameters.png`, `fig4b_energy_scales.png`, `figS13_mu_alignment.png` |
| `run_def2tzvp.py`, `plot_def2tzvp.py` | Fig. 5: H40/H38/H6 references and H6 with s and s+p coupling in def2-TZVP; Hirshfeld density differences | `results/def2tzvp/`, `figures/fig5*.png` |
| `run_reference_curves.py` | canonical ΔE(r) of the H40 and H38 rings and of the uncoupled fragments H6–H20 (plain pySCF) | `results/reference_curves/<system>/` |

```bash
cd paper/hring
python run_fragment_curves.py --fragment H6    # Fig. 3c (~8 min on 4 cores)
python run_fragment_curves.py                  # all fragments (Fig. 4)
python run_reference_curves.py                 # H40, H38 + uncoupled fragments (~25 min)
python run_density_comparison.py && python plot_density_comparison.py   # Fig. 3d, S10b, S12
python plot_curves.py && python plot_curves.py H6_alt && python plot_curves.py --overlay   # Fig. 3c, S10a, S11
python run_mu_sweep.py && python plot_mu_sweep.py                       # Fig. 3b (~5 min)
python run_electronic_structure.py && python plot_electronic_structure.py   # Fig. 4, Table S6, S13 (~15 min)
python run_def2tzvp.py && python plot_def2tzvp.py                       # Fig. 5 (def2-TZVP, a few hours)
```

## Notes

- The coupling parameters were optimized beforehand (L-BFGS on the
  curve objective against the H40 reference, μ from neutrality); they are
  listed in `run_fragment_curves.py` and are not re-optimized here.
- `H6` is the parameter set of Fig. 3c (ε = −0.065482, η = 0.204821,
  μ = −0.2884 Ha); `H6_alt` is the second local minimum shown in Fig. S10
  (ε = −0.060, η = 0.329 Ha). `H8_a` is the H8 parameter set of Fig. 4.
- Fragment curves: every point starts from the SAD guess (no warm start).
  SCF: Anderson (mixing 0.10, history 8), ≤ 200 iterations, ‖ΔP‖ < 1e-5,
  |ΔΩ| < 1e-8.
- Reference curves (H40, H38, uncoupled fragments): canonical pySCF
  (no GC coupling), default pySCF SCF settings, ≤ 200 iterations; each point
  starts from the density of the previous point (SAD + damping as fallback).
- Fig. 3b: as in the published figure, the μ range and the dashed HOMO/LUMO
  lines come from a canonical RHF calculation of H6 (the GC-SCF is RKS/PBE0).
