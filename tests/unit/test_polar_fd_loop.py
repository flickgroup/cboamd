"""Unit test for the finite-difference polarizability-gradient loop in
``MDCalculator.calculate`` (Task 4).

Contract pinned here:
  * the FD loop must call ``driver.polarizability`` ``2 * natom`` times (one per +/- displaced
    geometry) -- unchanged;
  * the FD loop must NOT call ``driver.energy_and_dipole`` (its energy/dipole return
    is discarded -- for an SCF driver only the SCF drive is needed, not
    ``dip_moment``);
  * the FD gradient written into ``gpolarizability_array[0, 0, iatom, 0]`` is
    derived solely from ``driver.polarizability`` and ``deltax``, so it is
    bit-for-bit identical before and after removing the discarded
    ``driver.energy_and_dipole``.

The driver-specific electronic structure is faked: ``calc.driver`` is a counting
Driver stub (and ``get_cboa_forces_bonini`` is monkeypatched), so no real
potential backend is needed.
"""
from types import SimpleNamespace

import numpy as np

from cboamd import pycalculator as pc
from cboamd.atomic_constants import P_a_B
from cboamd.drivers.base import Driver


def _make_calc(polar_fd="central"):
    """Minimal MDCalculator instance (bypassing the driver-dependent __init__)."""
    calc = pc.MDCalculator.__new__(pc.MDCalculator)
    calc._directory = "."
    calc.results = {}
    calc.p = SimpleNamespace(driver="nep", photons=True, dt=0.5, time=0.0,
                             deltax=0.01, polar_fd=polar_fd)
    # vectorized photon_mode contract (Task 1): omega/lam/qa/pa/fa/ea are (M,)
    # and pol_vec is (3, M); single mode here.
    calc.pt = SimpleNamespace(polar=True, update_polar=True, polar_force=True,
                              polar_value=np.zeros((3, 3)),
                              lam=np.array([0.1]),
                              pol_vec=np.array([[1.0], [0.0], [0.0]]),
                              omega=np.array([0.2]), fa=np.array([0.0]),
                              pa=np.array([0.0]), qa=np.array([0.0]),
                              ea=np.array([0.0]))
    calc.istep = 0
    return calc


class _CountingDriver(Driver):
    """Counting/deterministic Driver stub shared by both FD-mode tests.

    ``polarizability`` returns a distinct diagonal per call (v=1,2,3,...), so the
    FD differences are well-defined and reproducible regardless of mode.
    """
    name = 'counting'

    def __init__(self, calls, natom):          # no calc/coord/cell needed here
        self._calls = calls
        self._natom = natom

    def set_geometry(self, atoms):
        self._calls["set_geometry"] += 1

    def energy_and_dipole(self):
        self._calls["ed"] += 1
        return -1.0, np.array([0.1, 0.0, 0.0])

    def energy_gradient(self):
        return np.ones((self._natom, 3))

    def polarizability(self):
        self._calls["polar"] += 1
        v = float(self._calls["polar"])
        return np.diag([v, v, v])

    def dipole_gradient(self):
        return np.zeros((3, self._natom, 3))

    def polarizability_gradient(self, natom):
        return None  # forces the FD loop to run (np.any(zeros) is False)

    def refresh_displaced(self):
        # a force-engine driver has no SCF to re-solve; count the calls so the
        # loop's displaced-solve bookkeeping stays pinned.
        self._calls["refresh"] += 1


def _patch_glue(monkeypatch, calls, captured, natom):
    """Install the counting driver stub and the cavity-force spy."""
    def fake_cboa(pt, e, gfkernel, dipole, gdip, gpol):
        captured["gpol"] = np.copy(gpol)
        return gfkernel + gpol[0, 0, :, 0][:, None], np.array([0.5])  # ea is (M,)

    monkeypatch.setattr(pc, "get_cboa_forces_bonini", fake_cboa)
    return _CountingDriver(calls, natom)


def test_fd_loop_central_mode_unchanged(monkeypatch):
    """polar_fd='central' reproduces the exact prior behaviour (byte-identity
    guard for the opt-out path)."""
    natom = 2
    calls = {"ed": 0, "polar": 0, "set_geometry": 0, "refresh": 0}
    captured = {}
    driver = _patch_glue(monkeypatch, calls, captured, natom)

    from ase import Atoms
    calc = _make_calc(polar_fd="central")
    calc.driver = driver
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

    calc.calculate(atoms, properties=["energy", "forces"])

    # energy_and_dipole runs ONCE (the top-of-step call), never inside the loop
    assert calls["ed"] == 1
    # polarizability: one pre-loop value + 2 per atom in the FD loop
    assert calls["polar"] == 1 + 2 * natom
    # set_geometry is pushed before each polarizability: 1 (top) + 3 per atom
    # (pos, neg, restore).
    assert calls["set_geometry"] == 1 + 3 * natom
    # refresh_displaced runs once per displaced solve (2 per atom) and never on
    # the restore push, so a force-engine driver does no extra work.
    assert calls["refresh"] == 2 * natom

    # FD gradient is exactly (polar_pos - polar_neg) / (2 * dx_bohr), from the
    # deterministic stub: pre-loop call v=1, then per atom (pos, neg) = (2,3), (4,5)
    dx_bohr = calc.p.deltax / P_a_B
    expected = np.zeros((3, 3, natom, 3))
    expected[0, 0, 0, 0] = (2.0 - 3.0) / (2 * dx_bohr)
    expected[0, 0, 1, 0] = (4.0 - 5.0) / (2 * dx_bohr)
    np.testing.assert_allclose(captured["gpol"], expected, rtol=1e-12)

    # forces are produced and finite
    assert calc.results["forces"].shape == (natom, 3)
    assert np.all(np.isfinite(calc.results["forces"]))


def test_fd_loop_forward_mode_halves_solves(monkeypatch):
    """polar_fd='forward' reuses chi(R) (already computed this step), so it needs
    one displaced solve per atom instead of two, and two geometry pushes per atom
    (displace, restore) instead of three."""
    natom = 2
    calls = {"ed": 0, "polar": 0, "set_geometry": 0, "refresh": 0}
    captured = {}
    driver = _patch_glue(monkeypatch, calls, captured, natom)

    from ase import Atoms
    calc = _make_calc(polar_fd="forward")
    calc.driver = driver
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

    calc.calculate(atoms, properties=["energy", "forces"])

    assert calls["ed"] == 1
    # polarizability: one pre-loop value (chi0, reused) + 1 per atom in the FD loop
    assert calls["polar"] == 1 + natom
    # set_geometry: 1 (top) + 2 per atom (displace, restore) -- no negative point.
    assert calls["set_geometry"] == 1 + 2 * natom
    assert calls["refresh"] == natom

    # chi0 comes from the pre-loop polarizability call (v=1); the calculator stores it
    # into pt.polar_value at line ~186. chi0 = e_hat @ polar_value @ e_hat with
    # e_hat = x_hat = [1,0,0] -> polar_value[0,0] == 1.0. Per-atom displaced calls
    # then return v = 2, 3. Forward: gpol = (polar_pos[0,0] - chi0) / dx_bohr.
    dx_bohr = calc.p.deltax / P_a_B
    expected = np.zeros((3, 3, natom, 3))
    expected[0, 0, 0, 0] = (2.0 - 1.0) / dx_bohr
    expected[0, 0, 1, 0] = (3.0 - 1.0) / dx_bohr
    np.testing.assert_allclose(captured["gpol"], expected, rtol=1e-12)

    assert calc.results["forces"].shape == (natom, 3)
    assert np.all(np.isfinite(calc.results["forces"]))
