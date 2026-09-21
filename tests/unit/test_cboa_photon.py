"""Stage 1 unit tests: pure-numpy q-mode (photon-coordinate) helpers.

No ase or electronic-structure driver needed. Guards the closed-form photon
energy/force/screening equations from Bonini 2022. The
cavity field uses the (omega*q - lam*(pol.mu)) convention (Bonini 2022 eq 4/6),
matching the e-mode get_cboa_forces_bonini and making the SCF coupling the exact
derivative of the photon energy.
"""
import numpy as np
import pytest

from cboamd.photon_mode import photon_mode
from cboamd.pycboa_functions import (
    cboa_field_scalar, cavity_field, cboa_photon_energy, cboa_photon_force_q)


def _pt(pol_vec=(1.0, 0.0, 0.0), omega=0.2, lam=0.3, qa=0.5):
    return photon_mode(1, omega, lam, list(pol_vec), qa=qa)


# --- cboa_field_scalar / cavity_field ---------------------------------------

def test_field_scalar_closed_form():
    pt = _pt(omega=0.2, lam=0.3, qa=0.5)
    mu = np.array([0.4, -0.1, 0.2])
    # cboa_field_scalar is now per-mode (M,); index mode 0 for the single mode
    assert cboa_field_scalar(pt, mu)[0] == pytest.approx(0.2 * 0.5 - 0.3 * 0.4)


def test_cavity_field_is_lam_pol_times_scalar():
    pt = _pt(pol_vec=(0, 1, 0), omega=0.2, lam=0.3, qa=0.5)
    mu = np.array([0.4, 0.7, 0.2])
    scal = cboa_field_scalar(pt, mu)
    np.testing.assert_allclose(cavity_field(pt, mu), 0.3 * np.array([0, 1, 0]) * scal)


# --- cboa_photon_energy -----------------------------------------------------

def test_photon_energy_closed_form():
    pt = _pt(omega=0.2, lam=0.3, qa=0.5)
    mu = np.array([0.4, -0.1, 0.2])
    expected = 0.5 * 0.2**2 * (0.5 - 0.3 * 0.4 / 0.2)**2  # pol=x -> mu_x=0.4
    assert cboa_photon_energy(pt, mu) == pytest.approx(expected)


def test_photon_energy_equals_half_field_scalar_squared():
    pt = _pt(omega=0.2, lam=0.3, qa=0.5)
    mu = np.array([0.4, 0.0, 0.0])
    assert cboa_photon_energy(pt, mu) == pytest.approx(0.5 * cboa_field_scalar(pt, mu)**2)


def test_photon_energy_lambda_zero_is_pure_oscillator():
    pt = _pt(lam=0.0, omega=0.2, qa=0.5)
    mu = np.array([0.4, 0.0, 0.0])
    assert cboa_photon_energy(pt, mu) == pytest.approx(0.5 * 0.2**2 * 0.5**2)


def test_photon_energy_qa_zero_is_dipole_self_energy():
    pt = _pt(lam=0.3, omega=0.2, qa=0.0)
    mu = np.array([0.4, 0.0, 0.0])
    assert cboa_photon_energy(pt, mu) == pytest.approx(0.5 * 0.3**2 * 0.4**2)


# --- cboa_photon_force_q ----------------------------------------------------

def test_photon_force_closed_form():
    """Denom-free (Bonini 2022 eq 6): q-mode uses no polarizability -- the SCF
    dipole mu already carries the screening."""
    pt = _pt(omega=0.2, lam=0.3, qa=0.5)
    mu = np.array([0.4, 0.0, 0.0])
    expected = -0.2 * (0.2 * 0.5 - 0.3 * 0.4)
    assert cboa_photon_force_q(pt, mu)[0] == pytest.approx(expected)


def test_photon_force_lambda_zero_is_harmonic():
    pt = _pt(lam=0.0, omega=0.2, qa=0.5)
    mu = np.array([0.4, 0.0, 0.0])
    assert cboa_photon_force_q(pt, mu)[0] == pytest.approx(-0.2**2 * 0.5)


def test_photon_force_is_negative_energy_gradient():
    """F_q = -dE/dq verified by finite difference (q-mode is energy-conserving)."""
    pt = _pt(omega=0.2, lam=0.3, qa=0.5)
    mu = np.array([0.4, 0.0, 0.0])
    dq = 1e-6
    ptp, ptm = _pt(qa=0.5 + dq), _pt(qa=0.5 - dq)
    f_fd = -(cboa_photon_energy(ptp, mu) - cboa_photon_energy(ptm, mu)) / (2 * dq)
    assert cboa_photon_force_q(pt, mu)[0] == pytest.approx(f_fd, rel=1e-5)
