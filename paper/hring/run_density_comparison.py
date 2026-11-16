#!/usr/bin/env python3
"""
Electron-density comparison fragment vs H40 at the adsorption minimum
(paper Fig. 3d for H6, SI Fig. S10b for H6_alt, SI Fig. S12 for H6-H20).

The point charge q = +1 sits at x0 + r_min, with r_min = 1.2181 A the
minimum of the canonical H40 curve (three-point quadratic interpolation
around the discrete minimum at r = 1.25 A).

  fragment : GC-SCF density at the published (eps, eta, mu), same settings as
             run_fragment_curves.py (SAD start)
  H40      : canonical RKS density of the full ring with the same point charge;
             only the AO block of the ring atoms at the fragment positions is
             kept (the "local" H40 density)
  diff     : rho_fragment - rho_H40,local

All densities are evaluated in the ring plane (z = 0) on a rectangular grid
centred on the fragment atoms.

Output: results/density_comparison/<label>/density_comparison.npz, metadata.json
Plot with plot_density_comparison.py.

Usage:
    python run_density_comparison.py              # all fragments + H6_alt
    python run_density_comparison.py --system H6  # only H6 (Fig. 3d)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from pyscf import dft, gto, qmmm

from gcec import make_sigma_ao_diagonal
from gcec.scf_loop import scf_gc
from run_fragment_curves import (THIS_DIR, BASIS, XC, GRID_LEVEL, X0, POINT_CHARGE_Q, SCF,
                                 FRAGMENTS, build_mol, outer_h_aos, metadata)

RESULTS_DIR = THIS_DIR / "results" / "density_comparison"
R_MIN = 1.2181190329978901     # A, minimum of the H40 reference curve
ANG2BOHR = 1.8897259886
NX, NY = 241, 401

# plane size (x, y) in A per fragment size
SPAN = {6: (3.0, 6.0), 8: (4.0, 7.0), 10: (5.0, 8.0), 12: (6.0, 9.0), 14: (7.0, 10.0),
        16: (8.0, 11.0), 18: (9.0, 12.0), 20: (10.0, 13.0)}

# canonical H40 SCF (tight, damped: the ring is nearly metallic)
RING_SCF = dict(conv_tol=1e-10, max_cycle=800, diis_space=20, level_shift=0.05, damp=0.20)


def ring_atom_ids(n: int) -> list[int]:
    """H40 atoms (0-based) at the positions of the n-atom fragment (centred on the +x bond)."""
    return list(range(20 - n // 2, 20 + n // 2))


def rho_on_plane(mol, dm, center, span):
    gx = np.linspace(-span[0] / 2, span[0] / 2, NX)
    gy = np.linspace(-span[1] / 2, span[1] / 2, NY)
    A, B = np.meshgrid(gx, gy, indexing="xy")
    pts = np.stack([center[0] + A, center[1] + B, np.zeros_like(A)], -1).reshape(-1, 3)
    ao = mol.eval_gto("GTOval_sph", pts * ANG2BOHR)
    rho = np.einsum("pi,ij,pj->p", ao, dm, ao, optimize=True).reshape(NY, NX)
    return rho, np.array([gx[0], gx[-1], gy[0], gy[-1]])


def h40_density(x_charge):
    mol = build_mol("ring_full.xyz")
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level, mf.verbose = XC, GRID_LEVEL, 0
    for k, v in RING_SCF.items():
        setattr(mf, k, v)
    mf = qmmm.mm_charge(mf, [(x_charge, 0.0, 0.0)], [POINT_CHARGE_Q], unit="Ang")
    mf.kernel()
    return mol, mf.make_rdm1(), float(mf.e_tot), bool(mf.converged)


def run(label: str, ring_cache: dict):
    xyz, eps, eta, mu = FRAGMENTS[label]
    x_charge = X0 + R_MIN

    # fragment, GC-SCF
    mol_f = build_mol(xyz)
    n = mol_f.natm
    Sigma_ao, _ = make_sigma_ao_diagonal(mol_f, idxs=outer_h_aos(mol_f), gamma_val=eta, eps_val=eps)
    mf = dft.RKS(mol_f)
    mf.xc, mf.grids.level = XC, GRID_LEVEL
    mf = qmmm.mm_charge(mf, [(x_charge, 0.0, 0.0)], [POINT_CHARGE_Q], unit="Ang")
    out = scf_gc(mol_f, Sigma_ao, mu=float(mu), logger=None, verbose=False, mf=mf, **SCF)
    dm_f, Ne_f, conv_f = np.real(out[0]), float(out[1]), bool(out[5])

    # H40, canonical (computed once, shared by all fragments)
    if "dm" not in ring_cache:
        ring_cache["mol"], ring_cache["dm"], ring_cache["e"], ring_cache["conv"] = h40_density(x_charge)
    mol_r, dm_r = ring_cache["mol"], ring_cache["dm"]
    ids = ring_atom_ids(n)
    sl = mol_r.aoslice_by_atom()
    ao_idx = np.concatenate([np.arange(sl[a, 2], sl[a, 3]) for a in ids])
    coords = mol_r.atom_coords(unit="Ang")
    mol_loc = gto.M(atom=[("H", tuple(coords[a])) for a in ids], unit="Ang", basis=BASIS, verbose=0)
    dm_loc = dm_r[np.ix_(ao_idx, ao_idx)]

    # sanity check: fragment atoms sit at the chosen ring positions
    assert np.allclose(mol_f.atom_coords(unit="Ang"), coords[ids], atol=1e-5), "atom mapping mismatch"

    center = coords[ids].mean(0); center[2] = 0.0
    span = SPAN[n]
    rho_f, extent = rho_on_plane(mol_f, dm_f, center, span)
    rho_r, _ = rho_on_plane(mol_loc, dm_loc, center, span)
    diff = rho_f - rho_r
    print(f"{label}: fragment conv={conv_f} Ne={Ne_f:.6f}  H40 conv={ring_cache['conv']}  "
          f"max|diff|={np.abs(diff).max():.3e}", flush=True)

    out_dir = RESULTS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "density_comparison.npz", rho_frag=rho_f, rho_ring=rho_r, rho_diff=diff,
                        extent=extent, center=center, x_charge=x_charge, r_min=R_MIN,
                        ring_atom_ids=np.array(ids), fragment=label)
    (out_dir / "metadata.json").write_text(json.dumps(metadata(
        script="run_density_comparison.py", system=label, xyz=xyz, eps=eps, eta=eta, mu=mu,
        r_min_ang=R_MIN, x_charge_ang=x_charge, ring_atom_ids=ids, plane="xy (z=0)",
        span_ang=list(span), nx=NX, ny=NY, fragment_converged=conv_f, fragment_Ne=Ne_f,
        ring_scf=RING_SCF, ring_converged=ring_cache["conv"], ring_E_tot=ring_cache["e"],
        max_abs_diff=float(np.abs(diff).max())), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=[*FRAGMENTS, "all"], default="all")
    a = ap.parse_args()
    cache: dict = {}
    for label in (FRAGMENTS if a.system == "all" else [a.system]):
        run(label, cache)


if __name__ == "__main__":
    main()
