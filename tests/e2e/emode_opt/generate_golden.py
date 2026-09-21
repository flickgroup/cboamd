"""Regenerate the legacy golden anchor for test_ab_identity.py.

Runs arm A (legacy flags: central FD, emode_projection off, scf_warmstart off
-- byte-equivalent to the pre-optimization code for coupled runs) and writes
data/co2_sto3g_lam0.1_legacy.dat: a flat vector of NSTEPS+1 total energies (eV)
followed by the 9 final position components (Angstrom). Regenerate whenever
pyscf (grid defaults / libxc) is upgraded:

    python tests/e2e/emode_opt/generate_golden.py
"""
import os

import numpy as np
import pyscf

from test_ab_identity import _build, _run, NSTEPS, _DATA


def main():
    energies, positions = _run(_build('central', False, False))
    os.makedirs(os.path.dirname(_DATA), exist_ok=True)
    header = (f"legacy arm A golden: CO2 sto3g e-mode lambda=0.1 qa=2.0 "
              f"{NSTEPS} steps dt=20au conv_tol=1e-11\n"
              f"pyscf {pyscf.__version__}; format: {NSTEPS + 1} energies (eV) "
              f"then 9 final positions (A)")
    np.savetxt(_DATA, np.concatenate([energies, positions.ravel()]),
               header=header)
    print(f"wrote {_DATA} (pyscf {pyscf.__version__})")


if __name__ == '__main__':
    main()
