"""Unit tests for the cavity (photon) coupling force, get_cboa_forces_bonini.

Pure-numpy; no ase or electronic-structure driver needed. Guards the
dipole-gradient contraction: the
cavity force on each nucleus must contract the dipole-COMPONENT index of
gdipole_array (shape (3, natom, 3)) with the polarization vector,
lambda_gdipole[i,j] = sum_a pol_vec[a] d(mu_a)/dR_{i,j}.
"""
import numpy as np
import pytest

from cboamd.photon_mode import photon_mode
from cboamd.pycboa_functions import get_cboa_forces_bonini


def _pt(pol_vec=(1.0, 0.0, 0.0), omega=0.01, lam=0.1, qa=0.5):
    pt = photon_mode(1, omega, lam, list(pol_vec), qa=qa)
    pt.polar_value = np.zeros((3, 3))   # chi = 0 (point-charge / no polarizability)
    pt.polar_force = False
    return pt


def test_cavity_force_dipole_gradient_contraction():
    """natom=2 (deliberately != 3) so the old `pol_vec @ gdipole` would error;
    checks Ea and the q_i-weighted cavity force are exactly right."""
    pt = _pt()
    natom = 2
    gfkernel = np.zeros((natom, 3))
    dipole = np.array([0.2, 0.0, 0.0])
    q = np.array([0.5, -0.5])                 # per-atom charges
    gdipole = np.zeros((3, natom, 3))         # d(mu_a)/dR_{i,j} = q_i delta_{a,j}
    for i in range(natom):
        for a in range(3):
            gdipole[a, i, a] = q[i]
    gpol = np.zeros((3, 3, natom, 3))

    force, Ea = get_cboa_forces_bonini(pt, 0.0, gfkernel.copy(), dipole, gdipole, gpol)

    # Ea = omega*qa - lam*(pol_vec . dipole), denom = 1 (chi=0)
    assert Ea == pytest.approx(pt.omega * pt.qa - pt.lam * 0.2)
    # lambda_gdipole[i,j] = pol_vec[j]*q_i  => force = gfkernel - Ea*lam*lambda_gdipole
    lam_gd = np.array([[q[0], 0.0, 0.0], [q[1], 0.0, 0.0]])
    np.testing.assert_allclose(force, -Ea * pt.lam * lam_gd, atol=1e-12)


def test_cavity_force_zero_coupling_returns_bare_forces():
    pt = _pt(lam=0.0)
    natom = 3
    gfkernel = np.arange(natom * 3, dtype=float).reshape(natom, 3)
    dipole = np.array([0.1, -0.2, 0.05])
    gdipole = np.zeros((3, natom, 3))
    for i in range(natom):
        for a in range(3):
            gdipole[a, i, a] = 0.3
    force, Ea = get_cboa_forces_bonini(pt, 0.0, gfkernel.copy(), dipole, gdipole,
                                       np.zeros((3, 3, natom, 3)))
    np.testing.assert_allclose(force, gfkernel)        # lam=0 -> unchanged
    assert Ea == pytest.approx(pt.omega * pt.qa)
