#
import numpy as np

M_EPS = 1e-20


# --- q-mode (photon-coordinate / Bonini 2022) pure helpers ------------------
# These mirror the reference equations and carry no electronic-structure or
# ase dependency so they
# can be unit-tested in isolation and reused by both the calculator and the
# finite-difference force routines (DRY).

def cboa_field_scalar(pt, dipole):
  """Per-mode cavity field omega_m*qa_m - lam_m*(pol_m . mu) (Bonini 2022 eq 4/6).

  Returns an array of shape (M,). For M == 1 this is the historical scalar
  wrapped in a length-1 array. The same (omega*q - lam*mu) combination drives the
  SCF coupling, the photon energy/force, and the nuclear gradient (DRY); the minus
  sign matches the e-mode get_cboa_forces_bonini and makes the SCF coupling the
  exact derivative of cboa_photon_energy.

  NOTE: the name is now a misnomer -- it returns a per-mode *array*, not a
  scalar. Kept to avoid churn across cavity_scf.py / pycalculator.py imports;
  rename to ``cboa_field_per_mode`` only as a separate, mechanical refactor.
  """
  mu = np.asarray(dipole)
  pol_dot_mu = mu @ pt.pol_vec                      # (3,) @ (3, M) -> (M,)
  return pt.omega * pt.qa - pt.lam * pol_dot_mu


def cavity_field(pt, dipole):
  """Summed per-component cavity field sum_m lam_m * pol_m * ea_m (3-vector).

  This is the total field that couples to the dipole integrals in the q-mode
  SCF Fock and the nuclear-gradient cavity term. Summing over modes here keeps
  cavity_scf.py mode-agnostic (DRY). pol_vec is (3, M), so pol_vec @ (lam*ea)
  contracts the mode axis -> (3,).
  """
  ea = cboa_field_scalar(pt, dipole)               # (M,)
  return pt.pol_vec @ (pt.lam * ea)                # (3, M) @ (M,) -> (3,)


def cboa_photon_energy(pt, dipole):
  """Total cavity (photon) energy 0.5 * sum_m ea_m^2 (scalar).

  Reduces to the single-mode 0.5 * omega^2 * (qa - lam*(pol.mu)/omega)^2 at
  M == 1; to 0 for lam == 0 and to 0.5*lam^2*(pol.mu)^2 for qa == 0.
  """
  ea = cboa_field_scalar(pt, dipole)
  return 0.5 * float(np.sum(ea ** 2))


def cboa_photon_force_q(pt, dipole):
  """Per-mode photon force F_q,m = -omega_m * ea_m (Bonini 2022 eq 6), shape (M,).

  This is the exact -dE/dq Hellmann-Feynman force: the dipole ``mu`` is the
  self-consistent cavity-polarized dipole from the q-mode SCF, so the screening
  by the molecular polarizability is *already* contained in mu. q-mode therefore
  uses no polarizability/denom at all (that screening belongs to the e-mode,
  which couples to the bare dipole). Denom-free => q-mode MD conserves energy."""
  return -pt.omega * cboa_field_scalar(pt, dipole)


def get_cboa_forces_bonini(pt, e, gfkernel, dipole, gdipole_array, gpolarizability_array):
   """Multi-mode e-mode nuclear force (Bonini 2024, electric-field response).

   The single-mode scalar screening denom = 1 + lam^2 (pol.chi.pol) generalizes
   to the M x M matrix A[m,n] = delta_mn + lam_m lam_n (pol_m . chi . pol_n)
   (Bonini 2024 eq 26); the screened field amplitudes solve A @ ea = b with
   b[m] = omega_m*qa_m - lam_m*(pol_m . mu). The matter responds to the *total*
   effective field E_tot = sum_m lam_m pol_m ea_m, so the dipole-gradient and
   dipole-self-energy force terms (Bonini 2024 eq 8) contract with E_tot. Reduces
   exactly to the scalar single-mode result at M == 1. Returns (force, ea) with
   ea the per-mode screened field amplitude (shape (M,)); the calculator sets
   pt.fa = -pt.ea * pt.omega.

   pol_vec is (3, M): column m is mode m's polarization.
   """
   del e  # kept for call-site compatibility; force is gradient-based
   chi = np.asarray(pt.polar_value)
   mu = np.asarray(dipole)

   b = pt.omega * pt.qa - pt.lam * (mu @ pt.pol_vec)                 # (3,)@(3,M)->(M,)
   # screening matrix A[m,n] = delta_mn + lam_m lam_n (pol_m . chi . pol_n)
   amat = np.eye(pt.nmodes) + np.einsum(
       'm,n,dm,de,en->mn', pt.lam, pt.lam, pt.pol_vec, chi, pt.pol_vec)
   ea = np.linalg.solve(amat, b)                                     # (M,)

   e_tot = pt.pol_vec @ (pt.lam * ea)                               # (3,M)@(M,)->(3,)

   force_cboa_list = np.array(gfkernel, dtype=float)
   force_cboa_list -= np.einsum('a,aij->ij', e_tot, gdipole_array)
   if pt.polar_force is True:
      gpolar = gpolarizability_array
      if gpolar.ndim == 5:
         gpolar = gpolar[0]
      force_cboa_list -= 0.5 * np.einsum('a,abij,b->ij', e_tot, gpolar, e_tot)

   return force_cboa_list, ea

def get_q_p(pt, time, dt, integral, dipole):
    denom_chi = 0.
    denom_chi += pt.pol_vec @ pt.polar_value @ pt.pol_vec.T
    denom = 1. + pt.lam**2*denom_chi
    omegap = pt.omega / np.sqrt(denom)
    lam_pol_dipole = pt.lam*(pt.pol_vec @ dipole) / np.sqrt(denom)
    time_former = time - dt
    if abs(time) < M_EPS:
        integral = 0.0
    elif abs(integral) < M_EPS:
        integral = 0.5*lam_pol_dipole*np.exp(-1.0j*omegap*time_former)
    else:
        integral += lam_pol_dipole*np.exp(-1.0j*omegap*time_former)

    integrand = integral*np.exp(1.0j*omegap*time)*dt
    pt_q = np.imag(integrand)
    pt_p = omegap*np.real(integrand)

    q0 = pt.qa0*np.cos(omegap*time)
    q0 += pt.pa0/omegap*np.sin(omegap*time)
    pt_q += q0

    p0 = -pt.qa0*omegap*np.sin(omegap*time)
    p0 += pt.pa0*np.cos(omegap*time)
    pt_p += p0

    ea = (pt.omega*pt.qa - pt.lam*sum(pt.pol_vec*dipole))/denom

    dipolepol = np.copy(dipole)

    for idim in range(0, 3):
         dipolepol += pt.lam*pt.pol_vec[idim]*pt.polar_value[idim,:]*ea

    # from octopus/src/system/forces.F90 l.316
    # geo%atom(iatom)%f_photons(1:gr%sb%dim) = - P_PROTON_CHARGE*species_zval(geo%atom(iatom)%species)* &
    #        hm%ep%photon_forces(1:gr%sb%dim)

    return pt_q, pt_p, integral, ea, dipolepol
