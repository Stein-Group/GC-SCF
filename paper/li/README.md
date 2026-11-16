# Li40 cluster (paper Sec. IV.C, Figs. 6–7, SI Section S4)

RKS/PBE0/STO-3G, grid level 3, Li40 cut from a Li BCC(110) surface
(`li_bcc_110_big.xyz`, lattice constant 3.51 Å). The self-energy
Σ = ε − iη (ε = 0) acts on the valence AOs (2s, 2p) of the coupled Li atoms;
three coupling topologies are defined in `common.py` (`center`,
`surface_center`, `edges`, atom indices 0-based).

## Scripts

| Script | What it does | Output |
|---|---|---|
| `run_broadening_sweep.py` | η = 1e-5 … 0.5 Ha (15 values) at charge neutrality, for each coupling | `results/broadening/<scheme>/` |
| `run_charge_scan.py` | point charge q = +0.4 e above atom 16, d = 10 → 1 Å (33 points), μ fixed at the neutral value (center coupling, η = 2 mHa) — Fig. 8 | `results/charge_scan/center/` |
| `run_charge_scan_canonical.py` | same scan without coupling (canonical RKS, N_e = 120) | `results/charge_scan/canonical_ref.npz` |
| `plot_occupation_profile.py` | MO energies and projected occupations vs η (Fig. 6, SI Fig. S16) | `figures/` (png + csv) |
| `plot_charge_scan.py` | ΔE(d) coupled vs uncoupled and ΔN_e(d) | `figures/` (png + csv) |
| `plot_density_maps.py` | Δρ = ρ^GC − ρ^can at η = 5 mHa, side and top view, three couplings (Fig. 7) | `figures/` (png + npz) |
| `plot_density_scaling.py` | ∫\|Δρ\| d³r at η = 5 mHa vs number of coupled atoms (SI Fig. S17) | `figures/` (png + csv) |

## Run

```bash
cd paper/li
export OMP_NUM_THREADS=1        # optional; converged values do not depend on threads
python run_broadening_sweep.py --scheme center --max-points 2   # quick test (~minutes)
python run_broadening_sweep.py                                  # all schemes (hours)
python run_charge_scan.py                                       # uses the center sweep as start, if present
python run_charge_scan_canonical.py
python plot_occupation_profile.py
python plot_charge_scan.py
python plot_density_maps.py
python plot_density_scaling.py
```

Every result folder contains `metadata.json` with the git commit, Python,
pySCF and NumPy versions, host, thread setting and all SCF settings.

## Notes

- SCF settings: Anderson mixing,
  history 3 (sweep) / 4 (charge scan), at most 300 iterations,
  ‖ΔP‖ < 1e-5 and |ΔΩ| < 1e-6.
- The canonical reference of the charge scan uses max. 200 pySCF cycles and
  stores a convergence flag per distance.
- Fig. 8 energies are relative to d = 10 Å.
