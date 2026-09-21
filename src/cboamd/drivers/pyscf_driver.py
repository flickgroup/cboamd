"""pyscf driver: RKS electronic structure for e-mode CBOA dynamics.

The self-consistent q-mode variant (CavityRKS) ships with an extension
extension package, which registers a subclass of this driver under the same
'pyscf' driver name.
"""
import numpy as np
from pyscf import gto, dft
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase

from cboamd.drivers.base import Driver
from cboamd.drivers.registry import register_driver


@register_driver
class PySCFDriver(Driver):
    name = 'pyscf'
    supported_cboa_modes = ('e-mode',)

    @classmethod
    def parse_input(cls, jdata, p, settings):
        # moved from pymd_ase.py lines 184-195 (comments included):
        # pyscf SCF convergence tolerance (both CBOA modes).
        p.conv_tol = jdata.get('conv_tol', 1e-6)
        # pyscf level of theory (both CBOA modes). The defaults match the
        # historical hard-coded settings (pyscf's RKS default LDA,VWN + cc-pVDZ),
        # so key-less inputs reproduce older results unchanged.
        p.xc = jdata.get('xc', 'lda,vwn')
        p.basis = jdata.get('basis', 'ccpvdz')
        # deterministic SCF warm-start via the previous step's converged density.
        p.scf_warmstart = jdata.get('scf_warmstart', True)
        # single-mode pyscf e-mode: 1-RHS projected polarizability (chi.eps) instead of
        # the full 3x3 tensor. no-op for nmodes > 1 (full-tensor path kept).
        p.emode_projection = jdata.get('emode_projection', True)
        settings['conv_tol'] = p.conv_tol
        settings['xc'] = p.xc
        settings['basis'] = p.basis
        settings['scf_warmstart'] = p.scf_warmstart
        settings['emode_projection'] = p.emode_projection

    def __init__(self, calc, coord, cell):
        super().__init__(calc, coord, cell)
        self.mf = self._init_pyscf(coord)     # helper_functions.init_pyscf body
        from cboamd.pycalculator import todict
        self.mf.todict = lambda: todict(self.mf)

    def set_geometry(self, atoms):
        # init_geo pyscf branch (helper_functions.py lines 185-192)
        # convert ASE structural information to PySCF information; the locally
        # built Mole is already a private copy, so reset with it directly (no
        # redundant second deep copy).
        mol = self.mf.mol.copy()
        mol.atom = atoms_from_ase(atoms)
        mol.build()
        self.mf.reset(mol=mol)

    def _solve_scf(self):
        # get_energy_and_dipole pyscf branch (lines 241-263); the warm-start
        # density lives on self._dm_prev now (was calc._dm_prev)
        # Deterministic SCF warm-start: seed from the previous step's central
        # converged density. The shape guard falls back to pyscf's default guess
        # after a geometry/basis reset (nao mismatch). Capture _dm_prev right
        # after convergence -- before the caller's e-mode FD loop displaces mf
        # onto another geometry's density.
        # Own method so the q-mode subclass in the extension package reuses the
        # identical solve instead of duplicating it.
        warmstart = getattr(self.p, 'scf_warmstart', True)
        dm0 = getattr(self, '_dm_prev', None) if warmstart else None
        if dm0 is not None and dm0.shape[-1] == self.mf.mol.nao:
            self.mf.kernel(dm0=dm0)
        else:
            self.mf.kernel()
        if warmstart:
            self._dm_prev = self.mf.make_rdm1()

    def energy_and_dipole(self):
        self._solve_scf()
        e = self.mf.e_tot
        dipole = self.mf.dip_moment(unit='au')
        return e, dipole

    def energy_gradient(self):
        # get_forces pyscf branch (lines 300-316)
        gf = self.mf.nuc_grad_method()
        gf.verbose = self.mf.verbose
        #if self.p.mode.lower() == 'dft':
        #    gf.grid_response = True
        gfkernel = gf.kernel()
        return gfkernel

    def polarizability(self):
        # get_polar pyscf branch (lines 464-492); projected_response becomes
        # self._projected_response(eps)
        # importing pyscf.prop.polarizability monkeypatches RKS.Polarizability();
        # done lazily here so non-pyscf drivers don't pull in pyscf-properties.
        from pyscf.prop.polarizability.rhf import Polarizability  # noqa: F401
        # single-mode with the projection flag on: solve one field-along-eps RHS
        # (chi . eps) instead of the full 3-RHS tensor, packed rank-1 as the
        # TRANSPOSE outer(chi_eps, eps) so get_cboa_forces_bonini is untouched and
        # the dipolepol contraction polar_value @ e_tot == chi @ e_tot is exact for
        # e_tot parallel to eps (see plan Task 3). nmodes > 1 (or flag off) keeps
        # the full-tensor path unchanged.
        # TODO(multimode projection): project onto each distinct mode polarization
        projected = (getattr(self.pt, 'nmodes', 1) == 1
                     and getattr(self.p, 'emode_projection', True))

        def _recompute():
            if projected:
                eps = self.pt.pol_vec[:, 0]
                return np.outer(self._projected_response(eps), eps)
            return self.mf.Polarizability().polarizability()

        if self.pt.polar and self.pt.update_polar:
            polar = _recompute()
        elif self.pt.polar and not self.pt.update_polar:
            if abs(sum(sum(self.pt.polar_value - np.zeros((3, 3))))) < 1e-12:
                polar = _recompute()
            else:
                polar = self.pt.polar_value
        else:
            polar = self.pt.polar_value
        return polar

    def dipole_gradient(self):
        # get_gdipol pyscf branch (lines 546-567)
        # single-mode with the projection flag on: only eps . dmu/dR is ever used
        # by get_cboa_forces_bonini (e_tot is parallel to eps), so skip the full
        # Hessian-CPHF kernel_dipderiv and pack rank-1: gdip[a,i,t] =
        # eps[a] * (eps . dmu/dR)[i,t], whose e_tot contraction is exact.
        # eps . dmu/dR is an analytic contraction of the field CPHF solution
        # already computed for chi . eps this step (no extra SCF/CPHF); the
        # None -> kernel_dipderiv fallback is kept so tests can pin the full path.
        # TODO(multimode projection): project onto each distinct mode polarization
        if (getattr(self.pt, 'nmodes', 1) == 1
                and getattr(self.p, 'emode_projection', True)):
            eps = self.pt.pol_vec[:, 0]
            gdip_eps = self._analytic_projected_dipole_gradient(eps)
            if gdip_eps is not None:
                return np.einsum('a,it->ait', eps, gdip_eps)
        from pyscf.prop.infrared.rks import Infrared
        from pyscf.prop.infrared.rhf import kernel_dipderiv
        mf_ir = Infrared(self.mf)
        # kernel_dipderiv returns de[atom, nuclear-coord, dipole-component] = dmu_t/dR_{A,x};
        # transpose to the (dipole-component, atom, dim) convention used by the other
        # drivers and by get_cboa_forces_bonini.
        gdipole_array = np.transpose(kernel_dipderiv(mf_ir), (2, 0, 1))
        return gdipole_array

    def refresh_displaced(self):
        # was pycalculator.py lines 291-292 and 300-301:
        self.mf.kernel(dm0=getattr(self, '_dm_prev', None))

    # --- moved module functions, now methods (bodies verbatim, calc -> self.calc,
    #     cache attributes on self) ---

    def _init_pyscf(self, coord):
        # helper_functions.py 221-238
        conv_tol = getattr(self.p, 'conv_tol', 1e-06)
        xc = getattr(self.p, 'xc', 'lda,vwn')
        basis = getattr(self.p, 'basis', 'ccpvdz')
        mol = gto.Mole(atom=coord, basis=basis, spin=0, charge=0)
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = xc
        mf.verbose = 0
        mf.conv_tol = conv_tol
        mf.conv_tol_grad = mf.conv_tol
        return mf

    def _field_response(self, eps):
        """Solve ONE field-along-`eps` CPHF and return (chi_eps, mo1F, e1F).

        chi_eps (shape (3,)) is chi . eps; (mo1F, e1F) are the raw CPHF solution the
        analytic dipole gradient reuses (mo1F: (nmo, nocc); e1F: (nocc, nocc)).

        Mirrors pyscf.prop.polarizability.rhf.polarizability's construction (same
        dipole-integral h1 blocks, gen_vind / cphf.solve) but with a single RHS
        h1_eps = sum_x eps[x] h1[x] instead of all three field directions, and with
        the PLAIN (0,0,0) origin (not charge-center) so that the analytic gdip's
        e1F.s1oo trace matches kernel_dipderiv's h1_dip convention. chi_eps itself is
        origin-independent for the neutral molecules init_pyscf builds (charge=0);
        the origin only affects the (mo1F, e1F) gauge, pinned by a test.

        Result cached on `self._field_response_cache = (mo_coeff, eps_arr, chi, mo1F,
        e1F)`. Invalidation: (a) `self.mf.mo_coeff is` the cached object -- every
        kernel()/init_geo() replaces mo_coeff, so a new geometry/density misses; and
        (b) exact-equal `eps` -- so a different polarization direction misses. Both
        are cheap and correct; no fragile geometry hashing.
        """
        assert np.isclose(np.linalg.norm(eps), 1.0), \
            '_field_response needs a unit direction'
        eps = np.asarray(eps, dtype=float)
        cache = getattr(self, '_field_response_cache', None)
        if (cache is not None and cache[0] is self.mf.mo_coeff
                and np.array_equal(cache[1], eps)):
            return cache[2], cache[3], cache[4]

        from pyscf import lib
        from pyscf.scf import cphf
        # importing pyscf.prop.polarizability monkeypatches RKS.Polarizability();
        # it also supplies gen_vind / max_cycle_cphf / conv_tol defaults we reuse.
        from pyscf.prop.polarizability.rhf import Polarizability  # noqa: F401
        mf = self.mf
        polobj = mf.Polarizability()
        mol = mf.mol
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mocc = mo_coeff[:, mo_occ > 0]

        # PLAIN (0,0,0) origin -- see docstring. (3, nmo, nocc) dipole integrals in
        # the AO->MO occ block, exactly as the library.
        int_r = mol.intor_symmetric('int1e_r', comp=3)
        h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), mocc)
        # project the single field-along-eps RHS (keep a leading dim of 1).
        h1_eps = np.einsum('x,xij->ij', eps, h1)[None]
        mo1F, e1F = cphf.solve(polobj.gen_vind(mf, mo_coeff, mo_occ),
                               mf.mo_energy, mo_occ, h1_eps, np.zeros_like(h1_eps),
                               polobj.max_cycle_cphf, polobj.conv_tol)
        mo1F, e1F = mo1F[0], e1F[0]              # (nmo, nocc), (nocc, nocc)
        # library contraction e2 = einsum('xpi,ypi->xy', h1, mo1); (e2 + e2.T) * -2.
        # With one solved column y == eps and the tensor's symmetry, chi . eps is
        # chi_eps[x] = -4 * einsum('pi,pi', h1[x], mo1F).
        chi_eps = -4.0 * np.einsum('xpi,pi->x', h1, mo1F)
        self._field_response_cache = (mo_coeff, eps, chi_eps, mo1F, e1F)
        return chi_eps, mo1F, e1F

    def _projected_response(self, eps):
        """Single-RHS CPKS: return chi . eps (shape (3,)) for a field along `eps`.

        Thin wrapper over `_field_response` (which also returns the raw CPHF solution
        reused by the analytic dipole gradient). See `_field_response` for details.
        """
        return self._field_response(eps)[0]

    def _analytic_projected_dipole_gradient(self, eps):
        """eps . dmu/dR (shape (natom, 3)) by contracting the field CPHF solution.

        No extra SCF/CPHF: reuses the (mo1F, e1F) that `_field_response` already
        solved for chi . eps this step (cache hit under the normal get_polar ->
        get_gdipol calculate() order). Derivation (eps . dmu/dR = -d^2E/dF_eps dR;
        a uniform field has no basis dependence, S^F = 0):

            eps . dmu/dR[A,x] = Z_A eps_x                          (nuclear)
                              + sum_uv P_uv (eps.r)^{A,x}_uv        (Pulay, h2ao)
                              - [ 4 Tr(h1ao[A,x] . D1F)
                                  - 4 Tr(s1ao[A,x] . D1Fe)
                                  - 2 Tr(s1oo[A,x] . e1F) ]         (response bracket)

        with D1F = C.mo1F.C_occ^T, D1Fe = C.mo1F.diag(e_occ).C_occ^T. The bracket is
        pyscf's own Hessian cross-term (hessian/rhf.py hess_elec) with one side the
        field response. h1ao = mf.Hessian().make_h1(...); s1ao/s1oo from int1e_ipovlp.

        Sign -1 on the bracket is pinned NUMERICALLY (proto_3b_analytic.py): sto3g
        RKS, sign -1 -- H2O x_hat 1.2e-14 (exact: the formula is complete, no missing
        term), H2O off-axis 1.5e-9, CO2 off-axis 9.2e-7, CO2 x_hat 5.0e-6 (degenerate
        pi occ-occ gauge residual, 1000x below the accepted force band). make_h1 costs
        ~0.20 s at CO2/ccpvdz vs the finite-field route's ~0.65 s. The field h1 uses
        the plain (0,0,0) origin (via _field_response) to match kernel_dipderiv.
        """
        assert np.isclose(np.linalg.norm(eps), 1.0), \
            'analytic_projected_dipole_gradient needs a unit direction'
        from pyscf.prop.infrared.rhf import get_h2ao_dipderiv
        eps = np.asarray(eps, dtype=float)
        mf = self.mf
        mol = mf.mol
        _, mo1F, e1F = self._field_response(eps)
        mo_coeff, mo_occ, mo_energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
        mocc = mo_coeff[:, mo_occ > 0]
        natm = mol.natm

        # AO-basis perturbed density pieces (as in hessian.rhf.hess_elec).
        mo1F_ao = mo_coeff @ mo1F                # (nao, nocc)
        dm1 = np.einsum('pi,qi->pq', mo1F_ao, mocc)
        dm1e = np.einsum('pi,qi,i->pq', mo1F_ao, mocc, mo_energy[mo_occ > 0])

        # nuclear skeleton derivatives.
        hobj = mf.Hessian()
        h1ao = hobj.make_h1(mo_coeff, mo_occ)    # (natm, 3, nao, nao)
        s1a = -mol.intor('int1e_ipovlp', comp=3)
        aoslices = mol.aoslice_by_atom()

        P = mf.make_rdm1()
        h2ao = get_h2ao_dipderiv(mf)             # (natm, 3, 3(dip), nao, nao)
        pulay = np.einsum('axtuv,uv,t->ax', h2ao, P, eps)
        charges = np.asarray([mol.atom_charge(a) for a in range(natm)], float)

        de = np.zeros((natm, 3))
        for ia in range(natm):
            p0, p1 = aoslices[ia][2:]
            s1ao = np.zeros((3, mol.nao, mol.nao))
            s1ao[:, p0:p1] += s1a[:, p0:p1]
            s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
            s1oo = np.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)
            resp = (4 * np.einsum('xpq,pq->x', h1ao[ia], dm1)
                    - 4 * np.einsum('xpq,pq->x', s1ao, dm1e)
                    - 2 * np.einsum('xij,ij->x', s1oo, e1F))
            # sign -1 on the response bracket, pinned numerically (see docstring).
            de[ia] = charges[ia] * eps + pulay[ia] - resp
        return de
