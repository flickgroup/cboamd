# no-polar mlip comparison

Compares the CO2 dynamics of two machine-learned interatomic potentials, **NEP**
and **DeepMD**, against an **ab initio (DFT) reference**, without polarizability,
for cavity coupling `lambda = 0, 0.1, 0.3` over 1000 steps
(`eps_tilde = lambda*omega`). Shared settings are in `params.json`.

NEP, DeepMD and the ab initio reference are *different* models of the same
surface, so the tests are sanity-level: each method's mode sits in the C=O
stretch region, NEP/DeepMD track the reference at early times, and they land
near the DFT peak. The plots are the main deliverable.

## Cached references

Every method is read from `data/` (as `<method>_lam<lambda>.dat`, dipole_x(t) in
a.u.), so the tests load them and need only numpy/scipy/matplotlib:

- **NEP** (`nep_lam*.dat`): `generate_nep.py` (ase + calorine). `plot.collect()`
  falls back to recomputing NEP if its cache is missing.
- **DeepMD** (`deepmd_lam*.dat`): `generate_deepmd.py` (needs ase + deepmd-kit in
  one environment).
- **ab initio** (`aimd_lam*.dat`): not shipped with this repository. Two ways to
  get it: regenerate it locally with `generate_aimd.py`, which runs the same
  trajectories with the pyscf driver (needs the `pyscf` extra:
  `pip install -e .[pyscf]`; slow, a DFT energy and gradient every step), or take
  the cached DFT dipole trajectories from the data repository published with the
  li2026cboamd paper (no compute needed). Either way, place
  `aimd_lam{0.0,0.1,0.3}.dat` under `data/` to activate the comparison column.
  Without them the tests skip.

## Models

`models/` holds the model files: `nep-{energy,dipole,polar}.txt` (committed,
small) and the DeepMD graphs `dp-energy.pb`, `dp-dipole-aimd-ref.pb`. The DeepMD
graphs are not tracked in this repository; they are distributed through the
data repository published with the li2026cboamd paper and must be placed
under `models/` locally before running the DeepMD-dependent parts of this
suite.

## Tests and plotting

`plot.py` holds the shared logic (recompute NEP, load references, render figures);
the tests import it. Figures: `../../figures/nopolar_mlip_{dipole,spectrum}.png`.

```bash
cd tests/e2e/no-polar/mlip
PYTHONPATH=<repo>/src:<repo> python plot.py            # render figures
PYTHONPATH=<repo>/src:<repo> python generate_deepmd.py # cache DeepMD references
PYTHONPATH=<repo>/src:<repo> python generate_aimd.py   # cache the ab initio reference
```
