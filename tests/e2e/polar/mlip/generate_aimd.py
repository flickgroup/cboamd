"""Generate the cached ab initio (DFT) reference dipole trajectories locally.

The `aimd_lam*.dat` references are normally taken from the data repository
published with the li2026cboamd paper; this script regenerates them from
scratch with the pyscf driver (e-mode CBOA). Run it in an environment with the
pyscf extra installed:

    pip install -e .[pyscf]
    cd tests/e2e/<suite>/mlip && PYTHONPATH=<repo>/src:<repo> python generate_aimd.py

For each lambda it runs a `steps`-step CO2 trajectory (driver=pyscf, cavity
coupling eps_tilde = lambda*omega, polar settings from params.json) and saves the
dipole_x(t) trajectory to data/aimd_lam<lambda>.dat. A lambda is only recomputed
if its file is missing. This is slow (DFT energy/gradient, plus the IR
dipole-derivative and the finite-difference polarizability gradient every step
in the polar suite).
"""
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _repo_root(d):
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root (pyproject.toml) not found above " + HERE)


REPO = _repo_root(HERE)
P = json.load(open(os.path.join(HERE, "params.json")))
GEOM = os.path.join(REPO, P["geometry"])

from cboamd import pymd_ase


def run_aimd(lam):
    cfg = {
        "xyz_file": GEOM, "driver": "pyscf",
        "photons": lam > 0.0, "polar": P.get("polar", False),
        "polar_force": P.get("polar_force", False),
        "nphoton": 1, "omega_photon": [P["omega_au"]], "lambda_photon": [lam],
        "lambda_vector": [P["pol_vector"]], "steps": P["steps"], "timestep": P["timestep_fs"],
    }
    cwd = os.getcwd()
    d = tempfile.mkdtemp()
    os.chdir(d)
    try:
        json.dump(cfg, open("in.json", "w"))
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            sys.argv = ["cboamd", "-i", "in.json"]
            pymd_ase.main()
        dip = np.loadtxt("dipole.dat")
    finally:
        os.chdir(cwd)
    return dip[:, 2]   # dipole_x (a.u.)


def main():
    os.makedirs(DATA, exist_ok=True)
    for lam in P["lambdas"]:
        out = os.path.join(DATA, f"aimd_lam{lam}.dat")
        if os.path.exists(out):
            print("present, skipping:", os.path.basename(out))
            continue
        print(f"generating ab initio lambda={lam} ({P['steps']} steps)...")
        dx = run_aimd(lam)
        np.savetxt(out, dx, header=f"ab initio dipole_x(t) [a.u.]; omega={P['omega_cm']} cm-1, "
                                   f"lambda={lam}, {P['steps']} steps @ {P['timestep_fs']} fs, "
                                   f"polar={str(P.get('polar', False)).lower()}")
        print("saved", os.path.basename(out))


if __name__ == "__main__":
    main()
