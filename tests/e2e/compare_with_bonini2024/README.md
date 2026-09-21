# compare with Bonini (Rabi splitting: NEP vs explicit CBOA QEDFT)

The CBOA QEDFT reference branches (`data/octopus_CO2_CBOA_FD.txt`) are from
Bonini and Flick, J. Chem. Theory Comput. 18, 2764 (2022), DOI
10.1021/acs.jctc.1c01035 (and the comparison methodology from Bonini,
Ahmadabadi, and Flick, J. Chem. Phys. 161, 154104 (2024), DOI
10.1063/5.0230983).

The CO2 vibrational polariton frequencies versus cavity coupling strength
`lambda`, comparing the NEP machine-learned potential against the explicit cavity
Born-Oppenheimer QEDFT reference (octopus).

Four series, each with a lower and an upper polariton branch:

- **MLIP, chi included**: NEP run with the molecular polarizability term on
  (`polar=true`, `polar_force=true`). The cavity coupling is screened by `chi`
  (`denom = 1 + lambda^2 eps.chi.eps`), so the upper polariton stays near the bare
  mode, tracking QEDFT.
- **explicit CBOA QEDFT**: the reference, `data/octopus_CO2_CBOA_FD.txt` (4xN:
  rows are `lambda`, symmetric stretch, lower polariton, upper polariton; the symmetric-stretch row is not
  plotted, since the `dipole_x` spectrum only carries the asymmetric-stretch
  polaritons).
- **MLIP, chi neglected**: NEP run with the polarizability off. Without screening
  the upper polariton blows up with `lambda` (and the lower polariton red-shifts
  less), the deviation the comparison highlights.
- **MLIP, chi incl., omega_c on resonance**: same chi-included dynamics, but the
  BARE cavity frequency is un-screened per lambda,
  `omega_bare = omega_c * sqrt(1 + lambda^2 * chi_xx)` (with `chi_xx` the CO2
  polarizability along the cavity polarization from the NEP pol model, 21.83 a.u.),
  so that the dynamics' own `1/(1 + lambda^2 chi)` screening leaves the EFFECTIVE
  cavity frequency on the nominal resonance. Isolates the coupling-side effect of
  the polarizability from its detuning of the cavity mode: the polariton pair then
  stays centred on the bare mode (a symmetric anti-crossing, Rabi growing to
  ~630 cm-1 at lambda=0.3), whereas the un-adjusted chi-included pair slides down
  ~500 cm-1 as the screening red-detunes the cavity.

The cavity mode is resonant with the CO2 asymmetric stretch (`omega = 2400 cm-1`);
the `lambda` grid is taken from the first row of the reference file so the curves
overlay exactly. The lower/upper polariton frequencies are read off the `dipole_x`
power spectrum (two most intense peaks; degenerate at `lambda=0`).

## Cached references

`generate_nep.py` runs each NEP variant over the full `lambda` grid
(`steps = 10000`, for the FFT resolution needed to resolve the small splittings at
weak coupling) and caches the extracted branches in `data/`:

- `nep_chi_included.txt` / `nep_chi_neglected.txt` / `nep_chi_resonant.txt`:
  `3xN` tables (`lambda`, lower, upper) in cm^-1. `nep_chi_resonant.txt` is the
  chi-included run with the bare `omega_c` un-screened per lambda so the effective
  frequency stays on resonance (see the fourth series above).
- `nep_spectra.npz`: the full dipole_x **spectra** (intensity vs wavenumber) for
  the `SPEC_LAMBDAS` panels (outside-cavity + lambda = 0.1, 0.2, 0.3), per variant,
  for the stacked spectrum figure (`plot.spectrum_figure`). Shows how each variant
  redistributes the oscillator strength between the polariton peaks; the
  on-resonance panels annotate the bare `omega_c` fed to that variant. Plotted from
  this cache (no re-run).

A variant is recomputed only if its file is missing. `plot.collect()` falls back to
recomputing on the fly if a cache is absent (slow: the whole grid at 10000 steps).

## Models

NEP energy/dipole/polarizability models are shared with
`../no-polar/mlip/models/` (`nep-energy.txt`, `nep-dipole.txt`, `nep-polar.txt`).
NEP reproduces the ab initio polarizability tensor, which is why the chi-included
branch matches the QEDFT reference.

## Tests and plotting

`plot.py` holds the shared logic (NEP run, peak extraction, loading, plotting); the
tests import it. The figure is `../figures/compare_with_bonini2024_rabi.png`.

```bash
cd tests/e2e/compare_with_bonini2024
PYTHONPATH=<repo>/src:<repo> python generate_nep.py   # cache NEP branches (slow, once)
PYTHONPATH=<repo>/src:<repo> python plot.py           # render the Rabi figure
PYTHONPATH=<repo>/src:<repo> pytest                   # run the comparison tests
```
