"""Generate cached DeepMD reference dipole trajectories for the polar MLIP tests.

Run once in an environment that has ase + deepmd-kit (with a backend able to
load the .pb graphs) + calorine + jsonpickle:

    cd tests/e2e/polar/mlip
    PYTHONPATH=<repo>/src:<repo> python generate_deepmd.py

For each lambda it runs a 1000-step CO2 trajectory (driver=deepmd, polar=true,
cavity coupling eps_tilde = lambda*omega) and saves dipole_x(t) to
data/deepmd_lam<lambda>.dat. A lambda is only recomputed if its file is missing.
The DeepMD polarizability model (dp-polar.pb) is the documented outlier (see
README): it returns chi_xx~12.57 with chi_yy=chi_zz=0.

Note: running the deepmd driver needs ase AND deepmd-kit in the same
environment; set such an environment up before running.
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


def run_deepmd(lam, pol_key="pol"):
    dp = P["deepmd"]
    cfg = {
        "xyz_file": GEOM, "driver": "deepmd",
        "dp_pot": os.path.join(MODELS, dp["pot"]),
        "dp_dip": os.path.join(MODELS, dp["dip"]),
        "dp_pol": os.path.join(MODELS, dp[pol_key]),
        "deepmd_type_map": ["C", "O"],
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


def _generate(prefix, pol_key, note):
    for lam in P["lambdas"]:
        out = os.path.join(DATA, f"{prefix}_lam{lam}.dat")
        if os.path.exists(out):
            print("present, skipping:", os.path.basename(out))
            continue
        print(f"generating {prefix} lambda={lam} ({P['steps']} steps)...")
        dx = run_deepmd(lam, pol_key=pol_key)
        np.savetxt(out, dx, header=f"DeepMD dipole_x(t) [a.u.]; omega={P['omega_cm']} cm-1, "
                                   f"lambda={lam}, {P['steps']} steps @ {P['timestep_fs']} fs, "
                                   f"polar=true, {note}")
        print("saved", os.path.basename(out))


def main():
    os.makedirs(DATA, exist_ok=True)
    # real trained polar model (the documented outlier) ...
    _generate("deepmd", "pol", "polar model dp-polar.pb")
    # ... and the constant initial-state chi stand-in (correct value, zero gradient)
    _generate("deepmd_const", "pol_const", "constant chi dp-polar-const.npz")


if __name__ == "__main__":
    main()
