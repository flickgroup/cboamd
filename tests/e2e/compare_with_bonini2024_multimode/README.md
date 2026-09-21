# compare with Bonini (multi-mode: 7-harmonic CO2 Rabi splitting)

`data/CO2_Fig3_panel*.dat` are the Fig. 3 reference branches from Bonini,
Ahmadabadi, and Flick, J. Chem. Phys. 161, 154104 (2024), DOI
10.1063/5.0230983; the `aimd_*mode_lam*.dat` traces are the authors' own ab
initio MD runs.

Multi-mode counterpart of `../compare_with_bonini2024`: the CO2 vibro-polariton
frequencies versus cavity coupling `lambda` for a ladder of seven harmonic cavity
modes, compared against the ab initio Bonini 2024 Fig. 3B reference.

Setup (Fig. 3 panel B, chi included): `omega_a = (a/3) * omega_res`,
`lambda_a = (a/3) * lambda_res`, `a = 1..7`, so mode `a = 3` is resonant with the
CO2 asymmetric stretch. The resonant lower/upper polariton pair is extracted the
same way from both sides (the two IR-brightest branches in the resonant band), so
the comparison is like-for-like.

## Data

- `data/CO2_Fig3_panel{A,B,C,D}.dat`: the ab initio linear-response reference
  panels, `(Nlambda, 1 + 3*nb)` each (`lambda`, frequencies in meV, photon
  character, IR intensity). Panel B is the one compared here.
- `data/nep_lam*.dat`: NEP MD `dipole_x(t)` traces, one file per `lambda` on a fine
  0.01 grid. Regenerate with `python generate_branches.py nep` (seconds/lambda).
- `data/aimd_qmode_lam*.dat`, `data/aimd_emode_lam*.dat`: ab initio MD
  `dipole_x(t)` traces on the coarser `params.json` grid, shipped pre-generated.
  They need a self-consistent electronic-structure driver, which is not part of
  this package (it lives with the ab initio extension package), and cost hours per
  lambda.

Spectra, peak-picking and LP/UP extraction are all post-processing done at plot
time from these dipole traces, so changing the extraction never requires re-running
any MD.

## Tests and plotting

`plot.py` holds the shared logic (MD run, spectra, peak extraction, loading,
plotting); the tests import it. The figure is
`../figures/compare_with_bonini2024_multimode_rabi.png`.

The q-mode vs e-mode spectral cross-check (the two cavity Born-Oppenheimer
frameworks must give the same Rabi splitting) runs live self-consistent MD, so it
ships with the ab initio extension package rather than here.

```bash
cd tests/e2e/compare_with_bonini2024_multimode
PYTHONPATH=<repo>/src:<repo> python generate_branches.py nep   # cache NEP traces
PYTHONPATH=<repo>/src:<repo> pytest                            # run the comparison
```
