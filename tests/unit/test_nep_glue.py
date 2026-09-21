"""Unit tests for the NEP driver glue, using a mock interface, plus a
cross-driver equivalence check against the DeepMD driver glue.

Like test_deepmd_glue, calorine/NEP is not needed: the glue only calls
``get_potential_energy`` / ``get_forces`` on the potential and ``get_dipole_moment``
/ ``get_dipole_gradient`` / ``get_polarizability`` on the tensor models. We stand in
fakes and feed analytic raw values.

Key point (answering "should NEP return exactly the same as DeepMD?"): yes for the
final atomic-unit tensors, BUT only when each model is fed in its own native
convention. NEP and DeepMD differ in the raw dipole-gradient layout and sign:

  * NEP raw gradient is (atom, dim, component); the glue transposes (2,0,1) and does
    NOT negate.
  * DeepMD raw gradient is (component, atom, dim); the glue does NOT transpose and
    DOES negate.

Feeding both the *same* raw gradient array would therefore disagree (by a transpose
and a sign) -- the glue is exactly what reconciles them to one common convention.
NEP has no analytic polarizability gradient (it uses the finite-difference path), so
only the value glues are cross-checked for the polarizability.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from cboamd.atomic_constants import P_a_B, P_Har
from cboamd.drivers.deepmd import DeepMDDriver
from cboamd.drivers.nep import NEPDriver


class _FakePotential:
    def __init__(self, natom, energy_eV=0.0, forces_eVA=None):
        self._n = natom
        self._e = energy_eV
        self._f = np.zeros((natom, 3)) if forces_eVA is None else np.asarray(forces_eVA, float)

    def __len__(self):
        return self._n

    def get_positions(self):           # used by the DeepMD coords/cells path
        return np.zeros((self._n, 3))

    def get_cell(self):
        return np.zeros((3, 3))

    def get_potential_energy(self):
        return self._e

    def get_forces(self):
        return self._f


class _FakeNEPDipole:
    """calorine dipole model surface: value + (atom, dim, component) gradient."""

    def __init__(self, dipole, gradient_adc):
        self._d = np.asarray(dipole, float)
        self._g = np.asarray(gradient_adc, float)

    def get_dipole_moment(self):
        return self._d

    def get_dipole_gradient(self, charge=0):
        return self._g


class _FakeNEPPolar:
    """calorine polar model surface: (3,3) value and (atom, dim, i, j) gradient."""

    def __init__(self, polar_33=None, gradient_adij=None):
        self._p = None if polar_33 is None else np.asarray(polar_33, float)
        self._g = None if gradient_adij is None else np.asarray(gradient_adij, float)

    def get_polarizability(self):
        return self._p

    def get_polarizability_gradient(self):
        return self._g


class _FakeDeepMDModel:
    """DeepMD model surface: eval_full returns (value, gradient, atomic)."""

    def __init__(self, value, gradient):
        self._v = np.asarray(value, float)
        self._g = np.asarray(gradient, float)

    def eval_full(self, coords, cells, atom_types, atomic=False):
        return self._v, self._g, None


def _nep_driver(natom=3, dipole_unit="au", polar_unit="au",
                pot=None, dip=None, pol=None, polar_gradient_analytic=False):
    calc = SimpleNamespace()
    calc.p = SimpleNamespace(driver="nep",
                             dipole_unit=dipole_unit, polarizability_unit=polar_unit)
    calc.pt = SimpleNamespace(polar=True, update_polar=True, polar_force=True,
                              polar_gradient_analytic=polar_gradient_analytic)
    drv = NEPDriver.__new__(NEPDriver)   # skip __init__: tests stub calc_pot etc.
    drv.calc = calc
    drv.p = calc.p
    drv.pt = calc.pt
    drv.calc_pot = pot if pot is not None else _FakePotential(natom)
    drv.calc_dip = dip
    drv.calc_pol = pol
    return drv


def _deepmd_driver(natom=3, dipole_unit="au", polar_unit="au", dip=None, pol=None):
    calc = SimpleNamespace()
    calc.p = SimpleNamespace(driver="deepmd",
                             dipole_unit=dipole_unit, polarizability_unit=polar_unit)
    calc.pt = SimpleNamespace(polar=True, update_polar=True, polar_force=True)
    drv = DeepMDDriver.__new__(DeepMDDriver)   # skip __init__: tests stub calc_pot etc.
    drv.calc = calc
    drv.p = calc.p
    drv.pt = calc.pt
    drv.deepmd_atom_types = list(range(natom))
    drv._tensor_cache = {}
    drv.calc_pot = _FakePotential(natom)
    drv.calc_dip = dip
    drv.calc_pol = pol
    return drv


# --------------------------------------------------------------- NEP glue alone
def test_nep_energy_and_forces_units():
    forces_eVA = np.array([[0.1, -0.2, 0.3], [0.0, 0.4, -0.1], [-0.1, -0.2, -0.2]])
    drv = _nep_driver(pot=_FakePotential(3, energy_eV=-5.0, forces_eVA=forces_eVA),
                      dip=_FakeNEPDipole([0.0, 0.0, 0.0], np.zeros((3, 3, 3))))
    e, _ = drv.energy_and_dipole()
    gfkernel = drv.energy_gradient()
    assert e == pytest.approx(-5.0 / P_Har)
    np.testing.assert_allclose(gfkernel, -forces_eVA * (P_a_B / P_Har), rtol=1e-12)


def test_nep_dipole_value_and_gradient():
    """value unchanged in au; gradient (atom,dim,comp) -> transpose(2,0,1) -> x P_a_B."""
    natom = 3
    dip_raw = np.array([0.11, -0.22, 0.33])
    g_adc = np.arange(natom * 3 * 3, dtype=float).reshape(natom, 3, 3) + 1.0  # (atom,dim,comp)
    drv = _nep_driver(dip=_FakeNEPDipole(dip_raw, g_adc))
    _, dipole = drv.energy_and_dipole()
    gdip = drv.dipole_gradient()
    np.testing.assert_allclose(dipole, dip_raw, rtol=1e-12)
    assert gdip.shape == (3, natom, 3)
    np.testing.assert_allclose(gdip, np.transpose(g_adc, (2, 0, 1)) * P_a_B, rtol=1e-12)


def test_nep_polar_value():
    polar_raw = np.diag([21.8, 7.77, 7.77])
    drv = _nep_driver(pol=_FakeNEPPolar(polar_raw))
    np.testing.assert_allclose(drv.polarizability(), polar_raw, rtol=1e-12)


# ------------------------------------------------- NEP vs DeepMD: same final out
def test_nep_and_deepmd_glue_agree_when_fed_native_conventions():
    """Pick a common final dipole gradient OUT (component, atom, dim); express it in
    each driver's native raw convention; both glues must return OUT exactly."""
    natom = 3
    out = np.arange(3 * natom * 3, dtype=float).reshape(3, natom, 3) + 1.0  # target

    # NEP: out = transpose(nep_raw, (2,0,1)) * P_a_B  ->  nep_raw = transpose(out,(1,2,0))/P_a_B
    nep_raw = np.transpose(out, (1, 2, 0)) / P_a_B            # (atom, dim, comp)
    # DeepMD: out = -reshape(dmd_raw) * P_a_B           ->  dmd_raw = -out / P_a_B
    dmd_raw = -out / P_a_B                                    # (comp, atom, dim)

    dip_val = np.array([0.11, -0.22, 0.33])
    nep = _nep_driver(dip=_FakeNEPDipole(dip_val, nep_raw))
    dmd = _deepmd_driver(dip=_FakeDeepMDModel(dip_val, dmd_raw))

    _, nep_dipole = nep.energy_and_dipole()
    nep_grad = nep.dipole_gradient()
    dmd_dipole, dmd_grad = dmd._dipole_and_gradient()

    # both reconstruct the same physical dipole gradient
    np.testing.assert_allclose(nep_grad, out, rtol=1e-12)
    np.testing.assert_allclose(dmd_grad, out, rtol=1e-12)
    np.testing.assert_allclose(nep_grad, dmd_grad, rtol=1e-12)
    # dipole value matches too (same conversion path)
    np.testing.assert_allclose(nep_dipole, dmd_dipole, rtol=1e-12)


def test_nep_and_deepmd_polar_gradient_agree_when_fed_native_conventions():
    """Same idea for d(chi)/dR. NEP has an opt-in analytic path
    (polarizability_gradient with polar_gradient_analytic=True): calorine raw is (atom, dim, i, j) -> transpose
    (2,3,0,1), no sign flip; DeepMD raw is (i, j, atom, dim) flattened -> negated. Fed
    each native form, both reconstruct the same (i, j, atom, dim) tensor."""
    natom = 3
    out = np.arange(3 * 3 * natom * 3, dtype=float).reshape(3, 3, natom, 3) + 1.0

    # NEP: out = transpose(nep_raw, (2,3,0,1)) * P_a_B; (2,3,0,1) is its own inverse
    nep_raw = np.transpose(out, (2, 3, 0, 1)) / P_a_B          # (atom, dim, i, j)
    # DeepMD: out = -reshape(dmd_raw) * P_a_B
    dmd_raw = -out / P_a_B                                     # (i, j, atom, dim)

    nep = _nep_driver(pol=_FakeNEPPolar(gradient_adij=nep_raw), polar_gradient_analytic=True)
    dmd = _deepmd_driver(pol=_FakeDeepMDModel(np.diag([21.8, 7.77, 7.77]).reshape(-1), dmd_raw))

    nep_grad = nep.polarizability_gradient(natom)
    dmd_grad = dmd._polarizability_and_gradient()[1]
    assert nep_grad.shape == (3, 3, natom, 3)
    np.testing.assert_allclose(nep_grad, out, rtol=1e-12)
    np.testing.assert_allclose(dmd_grad, out, rtol=1e-12)
    np.testing.assert_allclose(nep_grad, dmd_grad, rtol=1e-12)


def test_nep_polar_gradient_defaults_to_finite_difference():
    """Without the opt-in flag, polarizability_gradient returns None for NEP, so
    the engine falls back to the (faster) finite-difference polar gradient."""
    nep = _nep_driver(pol=_FakeNEPPolar(gradient_adij=np.ones((3, 3, 3, 3))))
    assert nep.polarizability_gradient(3) is None


def test_identical_raw_gradient_does_NOT_agree():
    """Feeding both drivers the *same* raw gradient array disagrees -- by exactly the
    transpose and sign the glue exists to reconcile. This documents the conventions."""
    natom = 3
    same_raw = np.arange(3 * natom * 3, dtype=float).reshape(3, natom, 3) + 1.0
    nep = _nep_driver(dip=_FakeNEPDipole(np.zeros(3), same_raw))   # interpreted (atom,dim,comp)
    dmd = _deepmd_driver(dip=_FakeDeepMDModel(np.zeros(3), same_raw))  # interpreted (comp,atom,dim)
    nep_grad = nep.dipole_gradient()
    _, dmd_grad = dmd._dipole_and_gradient()
    assert not np.allclose(nep_grad, dmd_grad)
    # specifically: deepmd negates + keeps layout; nep transposes + keeps sign
    np.testing.assert_allclose(dmd_grad, -same_raw * P_a_B, rtol=1e-12)
    np.testing.assert_allclose(nep_grad, np.transpose(same_raw, (2, 0, 1)) * P_a_B, rtol=1e-12)
