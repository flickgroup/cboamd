# ASE calculator for the CBOA MD engine (originally an ASE calculator for an
# electronic-structure code)
# by Jakob Kraus
# units:  ase         -> units [eV,Angstroem,eV/Angstroem,e*A,A**3]
#         drivers     -> units [Ha,Bohr,Ha/Bohr,e*Bohr,Bohr**3]
# the todict trick enabling ASE trajectories comes from an upstream ASE
# calculator issue discussion
# modified by Johannes Flick

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from cboamd.atomic_constants import P_Har, P_a_B
from cboamd.pycboa_functions import get_cboa_forces_bonini, cboa_photon_force_q, cboa_field_scalar
from cboamd.drivers.registry import get_driver_class
import jsonpickle

class parameters():
    # holds the calculation mode and user-chosen attributes of post-HF objects
    def __init__(self):
        self.photons = False
        # dchi/dR finite-difference scheme for the engine's generic FD loop:
        # 'forward' (default; reuses chi(R)) or 'central' (2*natom displaced solves).
        self.polar_fd = 'forward'
    def show(self):
        print('------------------------')
        print('calculation-specific parameters set by the user')
        print('------------------------')
        for v in vars(self):
            print('{}:  {}'.format(v,vars(self)[v]))
        print('\n\n')

def todict(x):
    return jsonpickle.encode(x, unpicklable=False)

class MDCalculator(Calculator):

    implemented_properties = ['energy','forces','dipole','polarizability']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='cboamd', atoms=None, directory='.', **kwargs):
        # constructor
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, directory, **kwargs)
        self.initialize(**kwargs)
        self.p.show()

    def initialize(self, coord=None, cell=None, p=None, pt=None, istep=-1):
        # attach parameters and photon mode; ASE trajectory support via todict
        # (see the todict workaround referenced at the top of the file)
        self.p = p
        self.pt = pt
        self.istep = istep
        self.p.todict = lambda: todict(self.p)
        self.pt.todict = lambda: todict(self.pt)
        self.driver = get_driver_class(self.p.driver)(self, coord, cell)

    def calculate(self,atoms=None,properties=['energy'],system_changes=all_changes):
        Calculator.calculate(self,atoms=atoms,properties=properties,system_changes=system_changes)

        self.driver.set_geometry(atoms)

        natom = atoms.get_global_number_of_atoms()

        # q-mode only: the cavity SCF depends on the photon coordinate, so the
        # velocity-Verlet position half-step must precede the electronic-structure
        # solve. (e-mode keeps its in-block update; its SCF is qa-independent.)
        q_mode = self.p.photons and getattr(self.pt, 'cboa_mode', 'e-mode') == 'q-mode'
        q_fa0 = None
        if q_mode and self.istep >= 0:
            self.pt.qa += (self.pt.pa + self.pt.fa * self.p.dt * 0.5) * self.p.dt
            q_fa0 = np.copy(self.pt.fa)

        e, dipole = self.driver.energy_and_dipole()

        self.results['energy'] = e * P_Har
        self.results['dipole'] = dipole

        if 'forces' in properties:
            gfkernel = self.driver.energy_gradient()
            # Stash the BARE (zero-field) electronic force -- the ML-training
            # label -- in the same units/sign convention as results['forces']
            # (eV/A), BEFORE any cavity augmentation overwrites gfkernel below.
            # force.dat logs only the cavity-augmented total, so without this the
            # bare gradient would never be recorded.
            if q_mode:
                # q-mode: get_forces already returns the cavity-polarized SCF
                # gradient (the coupling is self-consistent in the density), so
                # no bare force exists here -- recovering one would need a
                # separate zero-field SCF. Keep the in-memory result as NaN of
                # the right shape; save_observables skips force_bare.dat entirely
                # in q-mode, so this NaN never reaches disk.
                self.results['forces_bare'] = np.full_like(
                    np.array(gfkernel, dtype=float), np.nan)
            else:
                # e-mode (and photons=False / lambda=0): gfkernel is still the
                # bare electronic gradient here; for those two paths the total
                # equals the bare force, for e-mode-with-photons it differs.
                # np.array(...) copies so the later in-place-ish overwrite of
                # gfkernel cannot alias this array.
                self.results['forces_bare'] = -1. * np.array(gfkernel) * (P_Har / P_a_B)
            # q-mode is self-consistent: the cavity polarization is already in the
            # SCF density, so the polarizability is neither needed nor computed
            # (it belongs to the post-hoc e-mode screening).
            # Only the e-mode with active coupling needs the matter response
            # properties; at lambda == 0 every cavity correction is provably zero
            # (denom -> 1, b's -lam(mu.pol) -> 0, e_tot -> 0), so they are skipped
            # (reaching q-mode speed for lambda=0 control runs). photons=False
            # bare-chi recording (a legitimate Raman-type use) is preserved.
            coupled = bool(np.any(self.pt.lam != 0.0))
            if self.pt.polar and not q_mode and (coupled or not self.p.photons):
                self.results['polarizability'] = self.driver.polarizability()
            else:
                self.results['polarizability'] = np.zeros((3,3))
            if self.p.photons == False:
                self.results['dipolepol'] = self.results['dipole']
            elif q_mode:
                # q-mode: nuclear forces (gfkernel) already include the cavity
                # coupling (FD of the total cavity energy in get_forces); the
                # photon force is the exact -dE/dq (no polarizability/denom -- the
                # screening is already in the self-consistent dipole).
                self.pt.ea = cboa_field_scalar(self.pt, dipole)
                self.pt.fa = cboa_photon_force_q(self.pt, dipole)
                if self.istep >= 0:
                    self.pt.pa += (q_fa0 + self.pt.fa) * self.p.dt * 0.5
                self.results['dipolepol'] = np.copy(dipole)
            elif self.p.photons == True:
                self.pt.polar_value = np.copy(self.results['polarizability'])
                if coupled:
                    gdipole_array = self.driver.dipole_gradient()
                    gpolarizability_array = self.driver.polarizability_gradient(natom)
                    if gpolarizability_array is None:
                        gpolarizability_array = np.zeros((3, 3, natom, 3))
                else:
                    # lambda == 0: no cavity coupling, so the response gradients
                    # multiply a zero field -- skip the expensive solves and pass
                    # zeros to get_cboa_forces_bonini (which then returns the bare
                    # gradient, ea = omega*qa, fa = -omega^2*qa, dipolepol = dipole).
                    gdipole_array = np.zeros((3, natom, 3))
                    gpolarizability_array = np.zeros((3, 3, natom, 3))

                coord0 = atoms.get_positions()
                # TODO(projection-opt): get_cboa_forces_bonini only ever uses the
                # polarization projection eps.alpha.eps (and eps.(dalpha/dR).eps),
                # never the full 3x3 tensor -- and likewise only eps.(dmu/dR) of
                # the dipole gradient. So every finite-difference path here (this
                # generic polarizability-gradient loop, and get_gdipol / the
                # polarizability FD for any driver that lacks analytic gradients)
                # could compute just the projected scalar/column instead of the
                # full tensor. For an expensive get_polar (e.g. an external-field
                # response: 2 field solves projected vs 6 for the full tensor)
                # that is a ~3x speedup. A driver that supplies a projected
                # polarizability and its projected gradient directly bypasses
                # this loop; generalize the same projection to the other FD
                # paths later.
                if coupled and self.pt.polar and self.pt.polar_force == True and not np.any(gpolarizability_array):
                    # positions are displaced by deltax Angstrom, but polar is in a.u.
                    # (Bohr^3) and the cavity force expects d(chi)/dR in atomic units;
                    # convert the step to Bohr (deltax / P_a_B).
                    dx_bohr = self.p.deltax / P_a_B
                    # 'forward' (default) reuses chi(R) already computed this step
                    # (self.pt.polar_value) for one displaced solve per atom;
                    # 'central' keeps the exact 2*natom-solve prior behaviour.
                    fd = getattr(self.p, 'polar_fd', 'forward')
                    # chi0 must be representation-independent: e_hat.chi.e_hat gives
                    # the same value under both the full-tensor and the rank-1
                    # packed polar_value, so this composes with the projected path.
                    e_hat = self.pt.pol_vec[:, 0]
                    chi0 = e_hat @ self.pt.polar_value @ e_hat
                    # double-eps pack: get_cboa_forces_bonini contracts gpolar as
                    # e_tot . gpolar . e_tot (einsum 'a,abij,b->ij'), so packing
                    # outer(e_hat, e_hat) * d(eps.chi.eps)/dR yields
                    # (e_tot . e_hat)^2 * scalar for any e_hat. Displaced scalars use
                    # e_hat . polar . e_hat (representation-independent). For e_hat=x_hat
                    # outer(x_hat, x_hat) has its only 1 at [0,0], so this reduces
                    # exactly to the old [0,0]-only fill (legacy byte-parity); for other
                    # e_hat it corrects the previously-zero x-only slot (plan carve-out).
                    ehat_outer = np.outer(e_hat, e_hat)
                    for iatom in range(natom):
                        coords_pos = np.copy(coord0)
                        coords_pos[iatom][0] += self.p.deltax
                        atoms.set_positions(coords_pos)
                        self.driver.set_geometry(atoms)
                        # only polarizability() is used; SCF drivers re-solve in
                        # refresh_displaced (their response needs a converged
                        # density) while the discarded energy/dipole is skipped.
                        # nep/deepmd polarizability() is self-contained.
                        self.driver.refresh_displaced()
                        polar_pos = self.driver.polarizability()

                        if fd == 'central':
                            coords_neg = np.copy(coord0)
                            coords_neg[iatom][0] -= self.p.deltax
                            atoms.set_positions(coords_neg)
                            self.driver.set_geometry(atoms)
                            self.driver.refresh_displaced()
                            polar_neg = self.driver.polarizability()
                            dchi = (e_hat @ polar_pos @ e_hat
                                    - e_hat @ polar_neg @ e_hat) / (2 * dx_bohr)
                        else:  # forward: reuse chi(R)
                            dchi = (e_hat @ polar_pos @ e_hat - chi0) / dx_bohr
                        gpolarizability_array[:, :, iatom, 0] = ehat_outer * dchi

                        atoms.set_positions(coord0)
                        self.driver.set_geometry(atoms)

                # update the photon displacement
                if self.istep >= 0:
                    self.pt.qa += (self.pt.pa + self.pt.fa * self.p.dt * 0.5) * self.p.dt
                    fa0 = np.copy(self.pt.fa)

                # calculate the forces
                gfkernel, self.pt.ea = get_cboa_forces_bonini(self.pt, e, gfkernel, dipole, gdipole_array, gpolarizability_array)
                self.pt.fa = -self.pt.ea * self.pt.omega

                # update the photon momentum
                if self.istep >= 0:
                    self.pt.pa += (fa0 + self.pt.fa) * self.p.dt * 0.5
                # self.pt.qa, self.pt.pa, self.pt.integral, self.pt.ea, dipolepol = get_q_p(self.pt, self.p.time, self.p.dt, self.pt.integral, dipole)
                # induced dipole from the total effective field: mu + chi @ E_tot,
                # E_tot = sum_m lam_m pol_m ea_m (pol_vec is (3, M)).
                e_tot = self.pt.pol_vec @ (self.pt.lam * self.pt.ea)   # (3,M)@(M,) -> (3,)
                dipolepol = np.copy(dipole) + self.pt.polar_value @ e_tot
                self.results['dipolepol'] = dipolepol
                self.results['polarizability'] = self.pt.polar_value

            self.p.time += self.p.dt
            forces = -1. * gfkernel * (P_Har / P_a_B)
            totalforces = []
            totalforces.extend(forces)
            totalforces = np.array(totalforces)
            self.results['forces'] = totalforces

        self.istep += 1
