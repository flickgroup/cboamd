"""Multi-mode e-mode (Bonini 2024) screened forces: matrix screening, exact
reduction to single mode at M=1, and the two-mode coupled linear solve.

The single-mode scalar screening denom = 1 + lam^2 (pol.chi.pol) generalizes to
the M x M matrix A[m,n] = delta_mn + lam_m lam_n (pol_m . chi . pol_n) (Bonini
2024 eq 26); ea = solve(A, b). Forces contract with the total field
E_tot = sum_m lam_m pol_m ea_m (eq 8).
"""
import numpy as np

from cboamd.photon_mode import photon_mode
from cboamd.pycboa_functions import get_cboa_forces_bonini


def _inputs(natom=3, seed=0):
    rng = np.random.default_rng(seed)
    gfkernel = rng.standard_normal((natom, 3))
    dipole = rng.standard_normal(3)
    gdipole = rng.standard_normal((3, natom, 3))
    chi = rng.standard_normal((3, 3))
    chi = chi + chi.T                              # symmetric polarizability
    gpolar = rng.standard_normal((3, 3, natom, 3))
    return gfkernel, dipole, gdipole, chi, gpolar


def test_reduces_to_single_mode():
    """One mode through the matrix path == the historical scalar-denom result."""
    gfkernel, dipole, gdipole, chi, gpolar = _inputs()
    omega, lam, qa, pol = 0.0211, 0.1, 5.0, np.array([0, 1, 0])
    pt = photon_mode(1, omega, lam, pol, qa=qa)
    pt.polar_value = chi
    pt.polar_force = True
    force, ea = get_cboa_forces_bonini(pt, 0.0, gfkernel.copy(), dipole,
                                       gdipole, gpolar)
    # closed-form single mode reference
    denom = 1.0 + lam ** 2 * (pol @ chi @ pol)
    ea0 = (omega * qa - lam * (pol @ dipole)) / denom
    E = lam * pol * ea0
    ref = gfkernel - np.einsum('a,aij->ij', E, gdipole) \
        - 0.5 * np.einsum('a,abij,b->ij', E, gpolar, E)
    np.testing.assert_allclose(ea, [ea0], atol=1e-12)
    np.testing.assert_allclose(force, ref, atol=1e-12)


def test_two_mode_matrix_solve():
    """ea solves A @ ea = b with A[m,n] = d_mn + lam_m lam_n (e_m.chi.e_n)."""
    gfkernel, dipole, gdipole, chi, gpolar = _inputs(seed=1)
    omega = np.array([0.0211, 0.0211])
    lam = np.array([0.1, 0.08])
    pol = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])     # (M,3) for ref math
    qa = np.array([5.0, -3.0])
    pt = photon_mode(2, omega, lam, pol.T, qa=qa)          # store as (3,M)
    pt.polar_value = chi
    pt.polar_force = True
    force, ea = get_cboa_forces_bonini(pt, 0.0, gfkernel.copy(), dipole,
                                       gdipole, gpolar)
    b = omega * qa - lam * (pol @ dipole)
    A = np.eye(2) + np.einsum('m,n,md,de,ne->mn', lam, lam, pol, chi, pol)
    np.testing.assert_allclose(ea, np.linalg.solve(A, b), atol=1e-12)
    E = np.einsum('m,m,md->d', lam, ea, pol)
    ref = gfkernel - np.einsum('a,aij->ij', E, gdipole) \
        - 0.5 * np.einsum('a,abij,b->ij', E, gpolar, E)
    np.testing.assert_allclose(force, ref, atol=1e-12)


def test_two_orthogonal_modes_decouple_with_diagonal_chi():
    """Orthogonal polarizations + diagonal chi -> A diagonal -> ea[m] = b[m]/denom_m."""
    natom = 3
    gfkernel = np.zeros((natom, 3))
    dipole = np.array([0.5, 0.2, 0.0])
    gdipole = np.zeros((3, natom, 3))
    chi = np.diag([2.0, 3.0, 1.0])
    gpolar = np.zeros((3, 3, natom, 3))
    omega = np.array([0.02, 0.02]); lam = np.array([0.1, 0.1])
    pol = np.array([[1.0, 0, 0], [0, 1.0, 0]]); qa = np.array([1.0, 1.0])
    pt = photon_mode(2, omega, lam, pol.T, qa=qa)          # store as (3,M)
    pt.polar_value = chi; pt.polar_force = False
    _, ea = get_cboa_forces_bonini(pt, 0.0, gfkernel.copy(), dipole, gdipole,
                                   np.zeros((3, 3, natom, 3)))
    d0 = 1 + 0.1 ** 2 * 2.0; d1 = 1 + 0.1 ** 2 * 3.0
    np.testing.assert_allclose(
        ea, [(0.02 - 0.1 * 0.5) / d0, (0.02 - 0.1 * 0.2) / d1], atol=1e-12)
