"""Generate cached NEP reference dipole trajectories for the polar MLIP tests.

Run in an environment with ase + calorine:

    cd tests/e2e/polar/mlip
    PYTHONPATH=<repo>/src:<repo> python generate_nep.py

For each lambda it runs a 1000-step CO2 trajectory (driver=nep, polar=true,
cavity coupling eps_tilde = lambda*omega) and saves dipole_x(t) to
data/nep_lam<lambda>.dat. A lambda is only recomputed if its file is missing.
"""
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "..", "..", "no-polar", "mlip", "models")  # shared models


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


def run_nep(lam):
    nep = P["nep"]
    cfg = {
        "xyz_file": GEOM, "driver": "nep",
        "nep_pot": os.path.join(MODELS, nep["pot"]),
        "nep_dip": os.path.join(MODELS, nep["dip"]),
        "nep_pol": os.path.join(MODELS, nep["pol"]),
        "photons": lam > 0.0, "polar": P["polar"], "polar_force": P["polar_force"],
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
    return dip[:, 2]


def main():
    os.makedirs(DATA, exist_ok=True)
    for lam in P["lambdas"]:
        out = os.path.join(DATA, f"nep_lam{lam}.dat")
        if os.path.exists(out):
            print("present, skipping:", os.path.basename(out))
            continue
        print(f"generating NEP lambda={lam} ({P['steps']} steps)...")
        dx = run_nep(lam)
        np.savetxt(out, dx, header=f"NEP dipole_x(t) [a.u.]; omega={P['omega_cm']} cm-1, "
                                   f"lambda={lam}, {P['steps']} steps @ {P['timestep_fs']} fs, polar=true")
        print("saved", os.path.basename(out))


if __name__ == "__main__":
    main()
