"""Unit tests for the DeepMD glue in DeepMDDriver, using a mock interface.

DeepMD-kit and the .pb graphs are not needed: the glue only touches a tiny API
surface (``eval_full`` on the dipole/polar models, and ``get_potential_energy`` /
``get_forces`` / ``get_positions`` / ``get_cell`` on the potential). We stand in
fakes implementing exactly that surface, feed *analytic* raw values in DeepMD's
native layout/units, and assert the glue returns the correct atomic-unit tensors
in the ``(component, atom, dim)`` convention the cavity force consumes.

This validates the implementation (unit conversions, sign, axis order, the
9 -> (3,3) reshape) independently of model quality -- the distinction that matters
because the supplied dp-polar.pb returns an incomplete polarizability (see the
polar e2e README). Deliberately asymmetric tensors are used so a wrong transpose
or reshape would be caught.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from cboamd.atomic_constants import P_a_B, P_Har
from cboamd.drivers.deepmd import DeepMDDriver


class _FakeTensorModel:
    """Stand-in for DeepDipole / DeepPolar: eval_full returns preset raw arrays."""

    def __init__(self, value, gradient):
        self._value = np.asarray(value, dtype=float)
        self._gradient = np.asarray(gradient, dtype=float)

    def eval_full(self, coords, cells, atom_types, atomic=False):
        return self._value, self._gradient, None


class _FakePotential:
    """Stand-in for the DP ASE calculator surface the glue reads."""

    def __init__(self, natom, energy_eV=0.0, forces_eVA=None):
        self._n = natom
        self._e = energy_eV
        self._f = np.zeros((natom, 3)) if forces_eVA is None else np.asarray(forces_eVA, float)

    def __len__(self):
        return self._n

    def get_positions(self):
        return np.zeros((self._n, 3))

    def get_cell(self):
        return np.zeros((3, 3))

    def get_potential_energy(self):
        return self._e

    def get_forces(self):
        return self._f


def _driver(natom=3, dipole_unit="au", polar_unit="au",
            dip=None, pol=None, pot=None):
    calc = SimpleNamespace()
    calc.p = SimpleNamespace(driver="deepmd",
                             dipole_unit=dipole_unit, polarizability_unit=polar_unit)
    calc.pt = SimpleNamespace(polar=True, polar_force=True)
    drv = DeepMDDriver.__new__(DeepMDDriver)   # skip __init__: tests stub calc_pot etc.
    drv.calc = calc
    drv.p = calc.p
    drv.pt = calc.pt
    drv.calc_pot = pot if pot is not None else _FakePotential(natom)
    drv.calc_dip = dip
    drv.calc_pol = pol
    drv.deepmd_atom_types = list(range(natom))
    drv._tensor_cache = {}
    return drv


# ---------------------------------------------------------------- energy / forces
def test_energy_and_forces_units():
    """energy: eV -> Ha (/P_Har); gfkernel: eV/Ang -> Ha/Bohr with a sign flip."""
    forces_eVA = np.array([[0.1, -0.2, 0.3], [0.0, 0.4, -0.1], [-0.1, -0.2, -0.2]])
    drv = _driver(pot=_FakePotential(3, energy_eV=-5.0, forces_eVA=forces_eVA))
    e, gfkernel = drv._energy_and_forces()
    assert e == pytest.approx(-5.0 / P_Har)
    np.testing.assert_allclose(gfkernel, -forces_eVA * (P_a_B / P_Har), rtol=1e-12)


# ----------------------------------------------------------------------- dipole
def test_dipole_value_and_gradient_au():
    """dipole value unchanged in au; gradient -> (component, atom, dim), x P_a_B,
    negated. Asymmetric input guards against a wrong transpose."""
    natom = 3
    dip_raw = np.array([0.11, -0.22, 0.33])
    # g_raw[a, i, j] all-distinct so any axis permutation is detectable
    g_raw = np.arange(3 * natom * 3, dtype=float).reshape(3, natom, 3) + 1.0
    drv = _driver(dip=_FakeTensorModel(dip_raw, g_raw))

    dipole, gradient = drv._dipole_and_gradient()
    np.testing.assert_allclose(dipole, dip_raw, rtol=1e-12)
    assert gradient.shape == (3, natom, 3)
    np.testing.assert_allclose(gradient, -g_raw * P_a_B, rtol=1e-12)


def test_dipole_eangstrom_unit_conversion():
    """eAngstrom: value /P_a_B, gradient left as-is (then negated)."""
    natom = 3
    dip_raw = np.array([1.0, 2.0, 3.0])
    g_raw = np.ones((3, natom, 3))
    drv = _driver(dipole_unit="eAngstrom", dip=_FakeTensorModel(dip_raw, g_raw))
    dipole, gradient = drv._dipole_and_gradient()
    np.testing.assert_allclose(dipole, dip_raw / P_a_B, rtol=1e-12)
    np.testing.assert_allclose(gradient, -g_raw, rtol=1e-12)


# ------------------------------------------------------------- polarizability
def test_polarizability_value_and_gradient_au():
    """polar 9 -> (3,3); gradient 9 -> (3,3,atom,dim), x P_a_B, negated."""
    natom = 3
    pol_raw = np.array([21.8, 0.1, 0.2, 0.1, 7.77, 0.0, 0.2, 0.0, 7.77])
    # distinct per (i,j,atom,dim) so the 9->(3,3) split and axis order are pinned
    g_raw = np.arange(9 * natom * 3, dtype=float).reshape(9, natom, 3) + 1.0
    drv = _driver(pol=_FakeTensorModel(pol_raw, g_raw))

    polar, gradient = drv._polarizability_and_gradient()
    np.testing.assert_allclose(polar, pol_raw.reshape(3, 3), rtol=1e-12)
    assert gradient.shape == (3, 3, natom, 3)
    np.testing.assert_allclose(gradient, -g_raw.reshape(3, 3, natom, 3) * P_a_B, rtol=1e-12)


def test_polarizability_angstrom3_unit_conversion():
    """Angstrom^3: value /P_a_B**3, gradient /P_a_B**2 (then negated)."""
    natom = 3
    pol_raw = np.arange(9, dtype=float) + 1.0
    g_raw = np.ones((9, natom, 3))
    drv = _driver(polar_unit="Angstrom3", pol=_FakeTensorModel(pol_raw, g_raw))
    polar, gradient = drv._polarizability_and_gradient()
    np.testing.assert_allclose(polar, pol_raw.reshape(3, 3) / P_a_B ** 3, rtol=1e-12)
    np.testing.assert_allclose(gradient, -g_raw.reshape(3, 3, natom, 3) / P_a_B ** 2, rtol=1e-12)


def test_glue_recovers_correct_tensor_so_model_is_the_outlier():
    """If the model returned the *correct* CO2 polarizability (matching NEP and
    the ab initio reference),
    the glue reproduces diag(21.8, 7.77, 7.77). The real dp-polar.pb instead
    returns chi_xx~12.57 / chi_yy=chi_zz=0 -- a model issue, not a glue issue."""
    natom = 3
    correct = np.diag([21.8, 7.77, 7.77]).reshape(-1)
    drv = _driver(pol=_FakeTensorModel(correct, np.zeros((9, natom, 3))))
    polar, gradient = drv._polarizability_and_gradient()
    np.testing.assert_allclose(polar, np.diag([21.8, 7.77, 7.77]), rtol=1e-12)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-15)


def test_gradient_size_mismatch_raises():
    """A wrong tensor size is rejected, not silently mis-reshaped."""
    drv = _driver(pol=_FakeTensorModel(np.zeros(9), np.zeros((9, 2, 3))))  # natom mismatch
    with pytest.raises(ValueError):
        drv._polarizability_and_gradient()
