"""Unit tests for the vectorized multi-mode photon_mode and the pure q-mode
field helpers.

Storage conventions mirror the reference implementation: omega/lam are
(M,) arrays and pol_vec is (3, M) (column m is mode m's polarization). The
constructor is idempotent so the q-mode finite-difference helpers can rebuild a
photon_mode from an existing one's arrays without nesting shapes.
"""
import numpy as np
import pytest

from cboamd.photon_mode import photon_mode


# --- data model (Task 1) ----------------------------------------------------

def test_single_mode_scalar_inputs_are_broadcast():
    pt = photon_mode(1, 0.2, 0.3, [1, 0, 0], qa=0.5)
    assert pt.nmodes == 1 and len(pt) == 1
    assert pt.omega.shape == (1,) and pt.lam.shape == (1,)
    assert pt.pol_vec.shape == (3, 1)                       # (3, nmodes), reference convention
    assert pt.qa.shape == (1,) and pt.pa.shape == (1,)
    np.testing.assert_allclose(pt.omega, [0.2])
    np.testing.assert_allclose(pt.pol_vec, [[1], [0], [0]])  # column 0 is mode 0's pol
    np.testing.assert_allclose(pt.qa, [0.5])


def test_multi_mode_arrays():
    # multi-mode pol_vec is (3, M): column m is mode m's polarization
    pol = np.array([[1, 0, 0], [0, 1, 0]]).T               # (3, 2)
    pt = photon_mode(2, [0.2, 0.25], [0.3, 0.1], pol, qa=[0.5, -0.2], pa=[0.0, 1.0])
    assert pt.nmodes == 2 and len(pt) == 2
    assert pt.omega.shape == (2,) and pt.pol_vec.shape == (3, 2)
    np.testing.assert_allclose(pt.lam, [0.3, 0.1])
    np.testing.assert_allclose(pt.pol_vec[:, 0], [1, 0, 0])
    np.testing.assert_allclose(pt.pol_vec[:, 1], [0, 1, 0])
    np.testing.assert_allclose(pt.qa, [0.5, -0.2])
    np.testing.assert_allclose(pt.pa, [0.0, 1.0])
    np.testing.assert_allclose(pt.fa, [0.0, 0.0])


def test_scalar_qa_broadcasts_across_modes():
    pol = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T    # (3, 3)
    pt = photon_mode(3, [0.2, 0.25, 0.3], [0.3, 0.1, 0.2], pol, qa=0.0)
    assert pt.qa.shape == (3,)
    np.testing.assert_allclose(pt.qa, [0.0, 0.0, 0.0])


def test_inconsistent_lengths_raise():
    with pytest.raises(ValueError):
        photon_mode(2, [0.2], [0.3, 0.1], np.array([[1, 0, 0], [0, 1, 0]]).T)


def test_pol_vec_is_normalized_with_warning():
    """A non-unit pol_vec column is normalized to unit length (direction kept),
    and a warning is emitted (the CBOA force code uses pol_vec directly, so a
    non-unit column would silently rescale the coupling)."""
    with pytest.warns(UserWarning, match="NOT unit-normalized"):
        pt = photon_mode(1, 0.02, 0.1, [0, 2, 0], qa=3.0)
    np.testing.assert_allclose(pt.pol_vec[:, 0], [0, 1, 0])       # rescaled, direction kept
    np.testing.assert_allclose(pt.lam, [0.1])   # coupling NOT folded into lam
    # multi-mode: each column normalized independently
    with pytest.warns(UserWarning, match="NOT unit-normalized"):
        m = photon_mode(2, [0.2, 0.25], [0.3, 0.1],
                        np.array([[3, 0, 0], [0, 0, 4]]).T)
    np.testing.assert_allclose(np.linalg.norm(m.pol_vec, axis=0), [1.0, 1.0])


def test_already_unit_pol_vec_emits_no_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # any warning -> test failure
        photon_mode(1, 0.2, 0.3, [1, 0, 0])
        photon_mode(2, [0.2, 0.25], [0.3, 0.1], np.array([[1, 0, 0], [0, 1, 0]]).T)


def test_zero_length_pol_vec_raises():
    with pytest.raises(ValueError, match="zero-length"):
        photon_mode(1, 0.2, 0.3, [0, 0, 0])
    with pytest.raises(ValueError):
        photon_mode(1, 0.2, 0.3, [np.nan, 0, 1])   # non-finite is a hard error too


def test_rebuild_of_normalized_nonunit_pol_is_silent():
    """The q-mode FD helpers rebuild photon_mode from an existing pt's arrays
    every step. After a non-unit input warned ONCE at construction, rebuilding
    from the (now normalized) pol_vec must be silent and bit-stable -- no
    warning spam, no normalization drift."""
    import warnings
    with pytest.warns(UserWarning, match="NOT unit-normalized"):
        pt = photon_mode(1, 0.2, 0.3, [1, 1, 0])     # irrational unit vector after
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # any warning -> failure
        pt2 = photon_mode(pt.nmodes, pt.omega, pt.lam, pt.pol_vec, qa=pt.qa)
    np.testing.assert_array_equal(pt2.pol_vec, pt.pol_vec)   # bitwise stable


def test_reconstruction_is_idempotent():
    """The q-mode FD helpers rebuild a photon_mode from an existing one's arrays:
    photon_mode(pt.nmodes, pt.omega, pt.lam, pt.pol_vec, qa=...). Re-feeding the
    already-shaped (M,)/(3,M) arrays must NOT nest or mangle them (regression for
    the brittle np.array([omega]) -> (1,1) failure)."""
    pt = photon_mode(1, 0.2, 0.4, [0, 0, 1], qa=1.0)
    pt2 = photon_mode(pt.nmodes, pt.omega, pt.lam, pt.pol_vec, qa=pt.qa)
    assert pt2.omega.shape == (1,) and pt2.lam.shape == (1,)
    assert pt2.pol_vec.shape == (3, 1) and pt2.qa.shape == (1,)
    np.testing.assert_allclose(pt2.omega, [0.2])
    np.testing.assert_allclose(pt2.pol_vec, [[0], [0], [1]])
    # multi-mode round-trip
    pol = np.array([[1, 0, 0], [0, 1, 0]]).T
    m = photon_mode(2, [0.2, 0.3], [0.1, 0.4], pol, qa=[0.5, -0.5])
    m2 = photon_mode(m.nmodes, m.omega, m.lam, m.pol_vec, qa=m.qa)
    np.testing.assert_allclose(m2.pol_vec, m.pol_vec)
    np.testing.assert_allclose(m2.omega, m.omega)


# --- pure q-mode field helpers (Task 2) -------------------------------------

from cboamd.pycboa_functions import (   # noqa: E402
    cboa_field_scalar, cavity_field, cboa_photon_energy, cboa_photon_force_q)

POL2 = np.array([[1, 0, 0], [0, 1, 0]]).T          # (3, 2): columns are mode pols


def test_field_scalar_is_per_mode_array():
    pt = photon_mode(2, [0.2, 0.25], [0.3, 0.1], POL2, qa=[0.5, -0.2])
    mu = np.array([0.4, 0.7, 0.2])
    ea = cboa_field_scalar(pt, mu)
    assert ea.shape == (2,)
    np.testing.assert_allclose(
        ea, [0.2 * 0.5 - 0.3 * 0.4, 0.25 * (-0.2) - 0.1 * 0.7])


def test_cavity_field_sums_over_modes():
    pt = photon_mode(2, [0.2, 0.25], [0.3, 0.1], POL2, qa=[0.5, -0.2])
    mu = np.array([0.4, 0.7, 0.2])
    ea = cboa_field_scalar(pt, mu)
    expected = 0.3 * np.array([1, 0, 0]) * ea[0] + 0.1 * np.array([0, 1, 0]) * ea[1]
    np.testing.assert_allclose(cavity_field(pt, mu), expected)


def test_photon_energy_sums_over_modes():
    pt = photon_mode(2, [0.2, 0.25], [0.3, 0.1], POL2, qa=[0.5, -0.2])
    mu = np.array([0.4, 0.7, 0.2])
    ea = cboa_field_scalar(pt, mu)
    assert cboa_photon_energy(pt, mu) == pytest.approx(0.5 * np.sum(ea ** 2))


def test_photon_force_q_is_per_mode_array():
    pt = photon_mode(2, [0.2, 0.25], [0.3, 0.1], POL2, qa=[0.5, -0.2])
    mu = np.array([0.4, 0.7, 0.2])
    ea = cboa_field_scalar(pt, mu)
    np.testing.assert_allclose(cboa_photon_force_q(pt, mu),
                               -np.array([0.2, 0.25]) * ea)
