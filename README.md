# cboamd

Cavity Born-Oppenheimer approximation (CBOA) molecular dynamics with
machine-learned potentials. `cboamd` propagates nuclei and cavity photon modes
together on a single potential-energy surface: an ML interatomic potential
supplies the bare energy and forces, an ML dipole model supplies the
light-matter coupling, and an optional ML polarizability model supplies the
screening of that coupling. Energies and forces come from pluggable drivers
(`pyscf` for ab initio DFT reference dynamics, `nep` for NEP potentials through
calorine, `deepmd` for DeepMD-kit graphs), so adding a new backend means adding
one driver class. Photon modes are arbitrary in number, frequency, and
polarization direction, and both the chi-included (polarizability-screened) and
chi-neglected variants of the e-mode dynamics are available, alongside the
q-mode (explicit photon-coordinate) dynamics with optional photon
thermostatting. The `pyscf` driver here implements the e-mode CBOA dynamics; the
self-consistent q-mode variant (the cavity coupling inside the SCF) ships with
an extension package, which registers a subclass of this driver
under the same `pyscf` name. The ML drivers run off nothing but the models you
supply: no classical force field, and no electronic-structure code unless you
install the `pyscf` extra.

## Install

```bash
pip install -e .[nep]        # NEP driver (calorine)
pip install -e .[deepmd]     # DeepMD-kit driver
pip install -e .[pyscf]      # pyscf driver (ab initio DFT reference dynamics)
```

Python 3.10 or newer. The core install needs only numpy, ase, and jsonpickle;
the driver backends are optional extras, so you install only the one you use.
The `pyscf` extra pulls `pyscf-properties` (the polarizability and IR
dipole-derivative add-on) straight from git, pinned to a specific commit, so
it needs a pip that can install from a git URL.

## Quickstart

```bash
cd examples
cboamd -i in.nep.json        # or: cboamd -i in.deepmd.json, in.pyscf.json
```

See `examples/README.md` for what the example runs and where to put the model
files. Each input is a JSON dictionary: geometry, driver name, model paths, MD
settings, and the cavity photon-mode block (`nphoton`, `omega_photon`,
`lambda_photon`, `lambda_vector`). The run writes `energy.dat`, `dipole.dat`,
`force.dat`, the photon observables, and an ASE trajectory.

The `infrared` console script post-processes a dipole trajectory into an IR or
polariton spectrum:

```bash
infrared -i dipole.dat
```

## Citation

If you use this code, please cite the cboamd methods paper:

```bibtex
@misc{li2026cboamd,
  author       = {Li, Yifan and Car, Roberto and Flick, Johannes},
  title        = {cboamd: A Machine Learning Molecular Dynamics Framework
                  for Vibrational Strong Coupling},
  year         = {2026},
  eprint       = {2609.22022},
  archivePrefix = {arXiv},
  doi          = {10.48550/arXiv.2609.22022},
  url          = {https://arxiv.org/abs/2609.22022}
}
```

The trained energy, dipole, and polarizability models are distributed through
the accompanying data repository
(<https://github.com/flickgroup/datasets_used_in_manuscripts/tree/main/li2026cboamd>).

### Reference data

The end-to-end comparison suites under `tests/e2e` redistribute reference data
from the following publications:

```bibtex
@article{bonini2024cavity,
  title   = {Cavity Born--Oppenheimer approximation for molecules and materials via electric field response},
  author  = {Bonini, John and Ahmadabadi, Iman and Flick, Johannes},
  journal = {J. Chem. Phys.},
  volume  = {161},
  pages   = {154104},
  year    = {2024},
  doi     = {10.1063/5.0230983}
}

@article{bonini2022abinitio,
  title   = {Ab Initio Linear-Response Approach to Vibro-Polaritons in the Cavity Born--Oppenheimer Approximation},
  author  = {Bonini, John and Flick, Johannes},
  journal = {J. Chem. Theory Comput.},
  volume  = {18},
  pages   = {2764--2773},
  year    = {2022},
  doi     = {10.1021/acs.jctc.1c01035}
}
```

## License

GNU General Public License v3.0 or later. See `LICENSE`.
