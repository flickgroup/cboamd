# polar mlip comparison

Compares the CO2 dynamics of two machine-learned interatomic potentials, **NEP**
and **DeepMD**, against an **ab initio (DFT) reference**, **with polarizability**
(`polar=true`, `polar_force=true`), for cavity coupling `lambda = 0, 0.1, 0.3`
over 1000 steps (`eps_tilde = lambda*omega`). Same structure as
`../../no-polar/mlip`, but the cavity coupling is screened by the molecular
polarizability (`denom = 1 + lambda^2 eps.chi.eps`) and the
polarizability-gradient force is included. Shared settings are in `params.json`.

NEP, DeepMD and the ab initio reference are *different* models of the same
surface (energy, dipole **and** polarizability), so the tests are sanity-level:
each method's mode sits in the C=O stretch region, and they land near the DFT
peak at `lambda=0`. The plots are the main deliverable.

## Polarizability models: NEP matches the reference, DeepMD is an outlier

The cavity coupling here is screened by the molecular polarizability
(`denom = 1 + lambda^2 eps.chi.eps`), so the polariton positions depend on `chi`.
At the CO2 geometry the three models give (atomic units, Bohr^3):

| method | chi_xx | chi_yy | chi_zz |
|--------|-------:|-------:|-------:|
| ab initio  |  21.8  |  7.77  |  7.77  |
| **NEP**    |  **21.83** |  **7.77**  |  **7.77**  |
| **DeepMD** |  **12.57** |  **0**     |  **0**     |

**NEP reproduces the reference tensor exactly. The DeepMD `dp-polar.pb` model is
the outlier**: it returns only `chi_xx ~ 12.57` (close to the *mean*
polarizability `(21.8+7.77+7.77)/3 = 12.45`) with `chi_yy = chi_zz = 0`. Since
the x-polarized cavity coupling uses `chi_xx`, DeepMD under-screens (12.57 vs
21.8) and its polariton shifts less than NEP or the reference. This is a
limitation of the supplied polarizability model, **not** the driver: the code
calls `eval_full(atomic=False)` on the graph (the same path as the working DeepMD
dipole model) and the model itself returns these values. The polar tests
therefore hold only NEP to the ab initio reference; DeepMD is plotted and
sanity-checked but flagged as the outlier.

## Confirming it is the model: DeepMD with a constant polarizability

To show the outlier is the model and not the glue, a fourth series **DeepMD
(const chi)** runs the DeepMD energy/dipole models but swaps the broken
`dp-polar.pb` for `dp-polar-const.npz`, a constant initial-state polarizability
`diag(21.83, 7.77, 7.77)` served by `ConstantPolar` (no TensorFlow / trained
graph; selected automatically because `dmd_pol` ends in `.npz`). With the correct
chi value (and a zero polarizability gradient, since chi is constant)
**DeepMD-const-chi lands exactly on NEP and the reference** (polariton peaks
2099/2499 at lambda=0.1 and 2499 at lambda=0.3, matching NEP), while the real
`dp-polar.pb` sits at 2566 / 2033. The test holds NEP **and** DeepMD-const-chi to
the ab initio reference; the real DeepMD remains the sanity-only outlier.

## Cached references

Every method is read from `data/` (as `<method>_lam<lambda>.dat`, dipole_x(t) in
a.u.), so the tests load them and need only numpy/scipy/matplotlib:

- **NEP** (`nep_lam*.dat`): `generate_nep.py` (ase + calorine). `plot.collect()`
  falls back to recomputing NEP if its cache is missing.
- **DeepMD** (`deepmd_lam*.dat`, `deepmd_const_lam*.dat`): `generate_deepmd.py`
  (needs ase + deepmd-kit in one environment).
- **ab initio** (`aimd_lam*.dat`): not shipped with this repository. Two ways to
  get it: regenerate it locally with `generate_aimd.py`, which runs the same
  polar=true trajectories with the pyscf driver (needs the `pyscf` extra:
  `pip install -e .[pyscf]`; slow, a DFT energy and gradient plus the response
  properties every step), or take the cached DFT dipole trajectories from the
  data repository published with the li2026cboamd paper (no compute needed).
  Either way, place `aimd_lam{0.0,0.1,0.3}.dat` under `data/` to activate the
  comparison column. Without them the tests skip.

## Models

Shared with `../../no-polar/mlip/models/`: `nep-{energy,dipole,polar}.txt` and the
DeepMD graphs `dp-energy.pb`, `dp-dipole-aimd-ref.pb`, `dp-polar.pb`. The DeepMD
graphs are not tracked in this repository; they are distributed through the
data repository published with the li2026cboamd paper and must be placed
under `models/` locally before running the DeepMD-dependent parts of this
suite. The polar run additionally uses the polarizability models
(`nep-polar.txt`, `dp-polar.pb`).

## Tests and plotting

`plot.py` holds the shared logic (recompute NEP, load references, render figures);
the tests import it. Figures: `../../figures/polar_mlip_{dipole,spectrum}.png`.

```bash
cd tests/e2e/polar/mlip
PYTHONPATH=<repo>/src:<repo> python plot.py             # render figures
PYTHONPATH=<repo>/src:<repo> python generate_deepmd.py  # cache DeepMD references
PYTHONPATH=<repo>/src:<repo> python generate_aimd.py    # cache the ab initio reference
```
