# Example: single CO2 molecule under vibrational strong coupling

This example runs a short polaritonic molecular dynamics trajectory for a
single CO2 molecule coupled to a cavity photon mode. The trained machine
learning potential, dipole, and polarizability models are not included in
this repository. Download them from the data repository published with the
li2026cboamd paper, then place the model files under `examples/models/` so
the relative paths in `in.deepmd.json` (for DeepMD models) or `in.nep.json`
(for NEP models) resolve correctly.

To run the example, use `cboamd -i in.deepmd.json` or `cboamd -i in.nep.json`
from inside the `examples/` directory. Each input runs 10 MD steps on the
geometry in `co2.xyz` with the cavity photon mode enabled, and can be used as
a starting point for longer production runs.

`in.pyscf.json` needs no downloaded models: install the `pyscf` extra
(`pip install -e .[pyscf]`, which pulls a pinned commit of
`pyscf-properties` from git) and run `cboamd -i in.pyscf.json` directly. It
reproduces the manuscript's ab initio AIMD e-mode protocol (PBE0,
aug-cc-pVDZ, the same cavity photon mode) at a reduced step count of 10, so
it finishes quickly on a laptop.
