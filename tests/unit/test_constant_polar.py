"""Unit tests for the constant-polarizability stand-in (ConstantPolar).

Verifies it mimics the DeepPolar eval_full contract, round-trips through the .npz
file, and -- fed through the actual DeepMD driver glue -- yields the constant chi with an
exactly zero polarizability gradient (so it screens the cavity correctly but adds
no polarizability force).
"""
from types import SimpleNamespace

import numpy as np

from cboamd.constant_polar import ConstantPolar
from cboamd.drivers.deepmd import DeepMDDriver

CHI = np.diag([21.8327, 7.7676, 7.7676])


class _FakePotential:
    def __init__(self, natom):
        self._n = natom

    def __len__(self):
        return self._n

    def get_positions(self):
        return np.zeros((self._n, 3))

    def get_cell(self):
        return np.zeros((3, 3))


def _deepmd_driver(natom=3, polar_unit="au", pol=None):
    calc = SimpleNamespace()
    calc.p = SimpleNamespace(driver="deepmd", dipole_unit="au", polarizability_unit=polar_unit)
    calc.pt = SimpleNamespace(polar=True, update_polar=True, polar_force=True)
    drv = DeepMDDriver.__new__(DeepMDDriver)   # skip __init__: tests stub calc_pot etc.
    drv.calc = calc
    drv.p = calc.p
    drv.pt = calc.pt
    drv.deepmd_atom_types = list(range(natom))
    drv._tensor_cache = {}
    drv.calc_pot = _FakePotential(natom)
    drv.calc_pol = pol
    return drv


def test_eval_full_shapes_and_value():
    cp = ConstantPolar(CHI)
    value, gradient, atomic = cp.eval_full(coords=np.zeros((1, 9)), cells=None,
                                           atom_types=[1, 0, 1])
    assert value.shape == (1, 9)
    assert gradient.shape == (1, 9, 3, 3)
    assert not gradient.any()
    np.testing.assert_allclose(value.reshape(3, 3), CHI)
    assert atomic is None


def test_value_is_geometry_independent():
    cp = ConstantPolar(CHI)
    v1, _, _ = cp.eval_full(np.zeros((1, 9)), None, [1, 0, 1])
    v2, _, _ = cp.eval_full(np.ones((1, 9)) * 3.7, None, [1, 0, 1])
    np.testing.assert_allclose(v1, v2)


def test_through_deepmd_glue_gives_constant_chi_and_zero_gradient():
    drv = _deepmd_driver(pol=ConstantPolar(CHI))
    polar, gradient = drv._polarizability_and_gradient()
    np.testing.assert_allclose(polar, CHI, rtol=1e-12)
    assert gradient.shape == (3, 3, 3, 3)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-15)


def test_npz_roundtrip(tmp_path):
    path = str(tmp_path / "const.npz")
    ConstantPolar(CHI).save(path)
    loaded = ConstantPolar(path)
    np.testing.assert_allclose(loaded.polarizability, CHI)
