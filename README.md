# GC-SCF

Grand-canonical self-consistent field (GC-SCF) calculations for finite
clusters coupled to an electron reservoir, built on
[PySCF](https://github.com/pyscf/pyscf).

The coupling enters as a wide-band-limit self-energy
Σ = ε − iη on selected atomic orbitals. The SCF is converged at fixed
chemical potential μ, with the density matrix obtained directly from
the non-Hermitian effective Hamiltonian, so the electron
number follows from μ and the coupling instead of being fixed.

This code accompanies:

> L. Hetzel, M. Head-Gordon, C. J. Stein,
> *An embedding method with constant potential boundary conditions*,
> arXiv:2608.25154 (2026).

## Installation

Requires Python ≥ 3.10.

```bash
git clone <repository-url>
cd GC-SCF
pip install -e .            # core: numpy, scipy, pyscf
pip install -e ".[paper]"   # plus pyyaml, matplotlib for paper/
```

## Quick start

```python
from pyscf import gto
from gcec import GCEC, SCFSettings, make_sigma_ao_diagonal

mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="sto-3g")

# couple AOs 1-6 (O 2s/2p, H 1s) with eta = 0.1 Ha
Sigma_ao, _ = make_sigma_ao_diagonal(mol, idxs=range(1, 7), gamma_val=0.1)

res = GCEC(mol, Sigma_ao=Sigma_ao, mu=0.107,
           settings=SCFSettings(method="EDIIS_OMEGA", max_cycle=200)).run_scf()
print(res["Omega"], res["Ne"])
```

`run_scf()` returns a dict with the grand potential `Omega`, the electron
number `Ne`, the density matrix `P`, the GC orbital energies, occupations
and broadenings (`mo_energy`, `mo_occ_gc`, `mo_gamma`), and the SCF log.

Mixing schemes (`SCFSettings.method`): `CDIIS`, `EDIIS_E`,
`EDIIS_OMEGA`, `ANDERSON`, `HYBRID` (EDIIS(Ω) + Anderson). See the paper
and SI Section S1 for details.

## Examples

| Script | System | Reproduces |
|---|---|---|
| `examples/water.py` | H₂O, RHF/STO-3G, η = 0.1 Ha | Table II (EDIIS(Ω), SAD guess) |
| `examples/hydrogen_ring.py` | H₆ fragment of the H₄₀ ring with a point charge, PBE0/MINAO | Fig. 3c at the minimum r = 1.22 Å |

The lithium-cluster results (Figs. 6, 7) were obtained with the same code
and the settings given in the paper and SI Section S4.

## Paper scripts

`paper/water/` regenerates the water results (Fig. 2, Figs. S2–S5;
the data behind Tables S1–S4 is written to `results/calculations/`):

```bash
cd paper/water
python gen_configs.py              # write the 120 YAML configs
python run_all_calculations.py     # run them -> results/calculations/
python plot_convergence.py         # Fig. 2a, Figs. S2-S5
python plot_orbitals.py            # Fig. 2b
python plot_density.py             # Fig. 2c
```

Each calculation writes `summary.json` (Ω, N_e, convergence) and
`log.json` (per-iteration log). Iteration counts in the SI tables are the
length of this log, i.e. they include the initial guess (iteration 0).

## Package layout

| Module | Content |
|---|---|
| `gcec/main.py` | `GCEC` driver and `SCFSettings` |
| `gcec/scf_loop.py` | GC-SCF kernel `scf_gc` |
| `gcec/occupations.py` | GC density matrix and occupations |
| `gcec/mu_solvers.py` | μ from a target electron number (Brent) |
| `gcec/diis.py` | CDIIS, EDIIS, Anderson mixing |
| `gcec/sigma.py` | Self-energy construction |
| `gcec/orthogonalization.py` | Löwdin orthogonalization |
| `gcec/scf_log.py` | SCF logging |

## License

Apache License 2.0, see `LICENSE`. `gcec/diis.py` contains a function
adapted from PySCF (Apache 2.0); see `NOTICE` and the header of that file.

## Citation

If you use this code, please cite:

```bibtex
@article{hetzel2026embedding,
  title={An embedding method with constant potential boundary conditions},
  author={Hetzel, Lisa and Head-Gordon, Martin and Stein, Christopher J},
  journal={arXiv preprint arXiv:2608.25154},
  year={2026}
}
```
