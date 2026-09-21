"""Fixtures for the degenerate two-mode e2e (Task 8).

Runs short NEP CO2 trajectories: a single x-polarized cavity mode and two
degenerate (x, y) modes at the same resonance/coupling. For an x-aligned linear
CO2 the y-polarized mode is IR-dark (mu_y ~ 0) and decoupled (chi off-diagonal
~ 0), so the asymmetric-stretch Rabi splitting from the dipole_x spectrum must be
the same with one axial mode or two degenerate modes -- a direct check that the
matrix screening reduces correctly and the extra mode is a spectator.

The suite skips cleanly if calorine or the NEP CO2 models are unavailable.
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(d):
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root not found")


REPO = _repo_root(HERE)
MODELS = os.path.join(REPO, "tests", "e2e", "no-polar", "mlip", "models")
GEOM = os.path.join(REPO, "examples", "co2.xyz")
OMEGA_AU = 0.0109354          # CO2 asymmetric-stretch resonance
LAMBDA = 0.1
STEPS = 8000
DT_FS = 0.5
BAND_CM = (1200.0, 3300.0)


def _dt_au():
    from ase import units
    from cboamd.atomic_constants import P_Ang, P_pe, P_Har
    return DT_FS * units.fs * P_Ang * (P_pe * P_Har) ** 0.5


def _run(omega_photon, lambda_photon, lambda_vector):
    """Run one NEP CO2 trajectory; return (dipole_x(t), total_energy(t))."""
    import contextlib
    import json
    import sys
    import tempfile
    from cboamd import pymd_ase
    nph = len(omega_photon)
    cfg = {
        "xyz_file": GEOM, "driver": "nep",
        "nep_pot": os.path.join(MODELS, "nep-energy.txt"),
        "nep_dip": os.path.join(MODELS, "nep-dipole.txt"),
        "nep_pol": os.path.join(MODELS, "nep-polar.txt"),
        "photons": True, "polar": True, "polar_force": True,
        "nphoton": nph, "omega_photon": omega_photon,
        "lambda_photon": lambda_photon, "lambda_vector": lambda_vector,
        "steps": STEPS, "timestep": DT_FS,
    }
    cwd, d = os.getcwd(), tempfile.mkdtemp()
    os.chdir(d)
    try:
        json.dump(cfg, open("in.json", "w"))
        argv = sys.argv
        sys.argv = ["cboamd", "-i", "in.json"]
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            pymd_ase.main()
        sys.argv = argv
        dip = np.loadtxt("dipole.dat")[:, 2]          # dipole_x
        energy = np.loadtxt("energy.dat")[:, 2]       # total energy (eV)
    finally:
        os.chdir(cwd)
    return dip, energy


def _split_cm(dipole_x):
    """Lower/upper polariton splitting (cm^-1) from the dipole_x spectrum."""
    import scripts.infrared as infrared
    from cboamd.atomic_constants import P_cm1
    sig = np.asarray(dipole_x) - np.mean(dipole_x)
    w, ft = infrared.fourier_transform(sig, _dt_au())
    freq, amp = w * P_cm1, np.abs(ft)
    m = (freq > BAND_CM[0]) & (freq < BAND_CM[1])
    f, a = freq[m], amp[m]
    loc = [k for k in range(1, len(a) - 1) if a[k] > a[k - 1] and a[k] > a[k + 1]]
    top = sorted(sorted(loc, key=lambda k: -a[k])[:2], key=lambda k: f[k])
    return f[top[-1]] - f[top[0]] if len(top) >= 2 else 0.0


@pytest.fixture(scope="session")
def two_mode_collected():
    pytest.importorskip("calorine")
    if not os.path.exists(os.path.join(MODELS, "nep-energy.txt")):
        pytest.skip("NEP CO2 models not available")
    dip1, e1 = _run([OMEGA_AU], [LAMBDA], [[1, 0, 0]])
    dip2, e2 = _run([OMEGA_AU, OMEGA_AU], [LAMBDA, LAMBDA], [[1, 0, 0], [0, 1, 0]])
    return {
        "split_1mode": _split_cm(dip1),
        "split_2mode": _split_cm(dip2),
        "energy1_drift": float(np.max(e1) - np.min(e1)),
        "energy2_drift": float(np.max(e2) - np.min(e2)),
    }
