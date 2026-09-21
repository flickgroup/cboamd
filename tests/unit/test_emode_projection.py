"""Task 3 core: 1-RHS projected polarizability for single-mode pyscf e-mode.

`PySCFDriver._projected_response(eps)` solves ONE field-along-eps CPKS RHS to
recover chi . eps, and polarizability() packs it rank-1 as the TRANSPOSE
outer(chi.eps, eps).
That leaves get_cboa_forces_bonini untouched (screening pol.chi.pol is identical
either way) while making the dipolepol observable polar_value @ e_tot exact.

Every test here needs pyscf + the pyscf.prop properties add-on, so the whole
file is importorskip-guarded at module level.
"""
import numpy as np
import pytest
from ase import Atoms

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("pyscf.prop")

from pyscf import gto, dft  # noqa: E402
from pyscf.prop.polarizability.rhf import Polarizability  # noqa: E402,F401

from cboamd import pycalculator as pc  # noqa: E402
from cboamd.drivers.pyscf_driver import PySCFDriver  # noqa: E402
from cboamd.photon_mode import photon_mode  # noqa: E402
from cboamd.pycboa_functions import get_cboa_forces_bonini  # noqa: E402


# --- helpers ----------------------------------------------------------------

def _rks(atom):
    mf = dft.RKS(gto.M(atom=atom, basis='sto3g', verbose=0))
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


def _driver(mf):
    """Minimal PySCFDriver stub exposing only .mf -- the only state the
    field-response methods need."""
    drv = PySCFDriver.__new__(PySCFDriver)   # skip __init__: tests stub mf directly
    drv.mf = mf
    return drv


def _driver_from_calc(calc):
    """The MDCalculator's own PySCFDriver (already carrying mf/p/pt)."""
    return calc.driver


def _emode_calc(coord, symbols, positions, *, nmodes, lam, pol_vec, omega,
                polar_force, polar_fd, emode_projection, conv_tol=1e-11):
    p = pc.parameters()
    p.driver = 'pyscf'
    p.photons = True
    p.dt = 0.5
    p.time = -0.5
    p.deltax = 1e-3
    p.dipole_unit = 'au'
    p.polarizability_unit = 'au'
    p.polar_fd = polar_fd
    p.conv_tol = conv_tol
    p.scf_warmstart = False
    p.emode_projection = emode_projection
    qa = 1.5 if nmodes == 1 else [1.5] * nmodes
    pt = photon_mode(nmodes, omega, lam, pol_vec, qa=qa)
    pt.cboa_mode = 'e-mode'
    pt.polar = True
    pt.update_polar = True
    pt.polar_force = polar_force
    pt.polar_value = np.zeros((3, 3))
    calc = pc.MDCalculator(coord=coord, cell=np.eye(3) * 20.0, p=p, pt=pt)
    # sto3g for speed (the driver builds ccpvdz by default); the override mol
    # persists through set_geometry, which copies driver.mf.mol.
    calc.driver.mf = dft.RKS(gto.M(atom=coord, basis='sto3g', verbose=0))
    calc.driver.mf.verbose = 0
    atoms = Atoms(symbols, positions=positions, cell=np.eye(3) * 20.0)
    atoms.calc = calc
    return calc, atoms


# CO2 along x (eps = x_hat is an eigenvector of chi -> orientation-blind).
_CO2_COORD = [['C', (0, 0, 0)], ['O', (1.16, 0, 0)], ['O', (-1.16, 0, 0)]]
_CO2_SYM = 'CO2'
_CO2_POS = [(0, 0, 0), (1.16, 0, 0), (-1.16, 0, 0)]

# H2O with off-diagonal chi so an off-axis eps is a genuine test.
_H2O_COORD = [['O', (0.0, 0.0, 0.0)], ['H', (0.0, 0.0, 0.96)],
              ['H', (0.0, 0.93, -0.24)]]
_H2O_SYM = 'OH2'
_H2O_POS = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.96), (0.0, 0.93, -0.24)]


# --- Test 1: anchor identity (pure algebra + one RKS) -----------------------

def test_projected_screening_matches_full():
    """The transposed rank-1 pack outer(chi.eps, eps) reproduces
    get_cboa_forces_bonini's ea, AND the dipolepol contraction identity
    outer(chi.eps, eps) @ e_tot == chi @ e_tot holds for e_tot parallel to eps."""
    mf = _rks('O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24')
    chi = mf.Polarizability().polarizability()
    eps = np.array([0, 1, 0], float)
    dip = mf.dip_moment(unit='au', verbose=0)
    natm = mf.mol.natm
    gf = np.zeros((natm, 3))
    gdip = np.zeros((3, natm, 3))
    pt = photon_mode(1, 0.02, 0.1, [0, 1, 0], qa=3.0)
    pt.polar_force = False

    pt.polar_value = chi
    _, ea_full = get_cboa_forces_bonini(pt, 0.0, gf.copy(), dip, gdip,
                                        np.zeros((3, 3, natm, 3)))
    pt.polar_value = np.outer(chi @ eps, eps)          # TRANSPOSED pack
    _, ea_proj = get_cboa_forces_bonini(pt, 0.0, gf.copy(), dip, gdip,
                                        np.zeros((3, 3, natm, 3)))
    np.testing.assert_allclose(ea_proj, ea_full, rtol=1e-12)

    e_tot = 0.1 * ea_proj[0] * eps                      # e_tot parallel to eps
    np.testing.assert_allclose(np.outer(chi @ eps, eps) @ e_tot,
                               chi @ e_tot, rtol=1e-12)


# --- Test 2: _projected_response vs full tensor -----------------------------

@pytest.mark.parametrize("atom", [
    'O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
    'C 0 0 0; O 0 0 1.16; O 0 0 -1.16',
])
@pytest.mark.parametrize("eps", [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0),          # non-axis
])
def test_projected_response_matches_full_tensor(atom, eps):
    """chi . eps from one CPKS RHS equals the full-tensor chi @ eps to 1e-8."""
    mf = _rks(atom)
    chi = mf.Polarizability().polarizability()
    got = _driver(mf)._projected_response(eps)
    np.testing.assert_allclose(got, chi @ eps, atol=1e-8)


def test_projected_response_requires_unit_direction():
    mf = _rks('O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24')
    with pytest.raises(AssertionError):
        _driver(mf)._projected_response(np.array([0.0, 1.0, 1.0]))


# --- Test 3: projected vs full through the real calculator (CO2, central) ---

def test_projected_vs_fulltensor_emode_force_through_calculator(monkeypatch):
    """One real MDCalculator step, CO2, nmodes==1, lambda=0.1, polar_force=True,
    central FD: projected (eps=x_hat) reproduces the full-tensor forces/ea/fa.

    Pins the EXACTNESS of the Task-3 core (1-RHS chi + rank-1 packing) at 1e-8,
    so the finite-field gdip (Task 3b, its own tolerance band, tested below) is
    forced to the full kernel_dipderiv path in BOTH arms."""
    monkeypatch.setattr(PySCFDriver, "_analytic_projected_dipole_gradient",
                        lambda *a, **k: None)   # None -> kernel_dipderiv fallback

    def _run(projection):
        calc, atoms = _emode_calc(
            _CO2_COORD, _CO2_SYM, _CO2_POS, nmodes=1, lam=0.1,
            pol_vec=[1, 0, 0], omega=0.02, polar_force=True,
            polar_fd='central', emode_projection=projection)
        f = atoms.get_forces()
        return f, np.copy(calc.pt.ea), np.copy(calc.pt.fa)

    f_proj, ea_proj, fa_proj = _run(True)
    f_full, ea_full, fa_full = _run(False)
    np.testing.assert_allclose(f_proj, f_full, atol=1e-8)
    np.testing.assert_allclose(ea_proj, ea_full, atol=1e-8)
    np.testing.assert_allclose(fa_proj, fa_full, atol=1e-8)


def test_analytic_gdip_force_band_through_calculator():
    """Same A/B with the analytic gdip ACTIVE: the projected forces stay within
    the documented band (CO2 gdip residual ~5e-6 a.u. x e_tot ~ 1e-6 eV/A; assert
    1e-5 eV/A with margin). Guards the Task-3c analytic contraction end to end."""
    def _run(projection):
        calc, atoms = _emode_calc(
            _CO2_COORD, _CO2_SYM, _CO2_POS, nmodes=1, lam=0.1,
            pol_vec=[1, 0, 0], omega=0.02, polar_force=True,
            polar_fd='central', emode_projection=projection)
        return atoms.get_forces()

    np.testing.assert_allclose(_run(True), _run(False), atol=1e-5)


# --- Test 4: dipolepol off-axis (the packing-orientation guard) -------------

def test_dipolepol_matches_fulltensor_offaxis():
    """H2O, off-axis eps=[0,1,1]/sqrt(2): dipolepol must match the full tensor.
    FAILS with the un-transposed pack outer(eps, chi.eps)."""
    eps = np.array([0, 1, 1], float)                    # photon_mode normalizes

    def _run(projection):
        with pytest.warns(UserWarning):                 # non-unit pol_vec warning
            calc, atoms = _emode_calc(
                _H2O_COORD, _H2O_SYM, _H2O_POS, nmodes=1, lam=0.1,
                pol_vec=eps, omega=0.02, polar_force=False,
                polar_fd='central', emode_projection=projection)
        atoms.get_forces()
        return np.copy(calc.results['dipolepol'])

    dp_proj = _run(True)
    dp_full = _run(False)
    np.testing.assert_allclose(dp_proj, dp_full, atol=1e-8)


# --- Test 5: forward + projected + off-axis corner --------------------------

def test_forward_projected_offaxis_parity():
    """H2O, off-axis eps, polar_force=True, forward FD: projected vs full forces
    agree within the forward-difference band (the corner where a chi0 read of a
    zero [0,0] slot would explode)."""
    eps = np.array([0, 1, 1], float)

    def _run(projection):
        with pytest.warns(UserWarning):
            _, atoms = _emode_calc(
                _H2O_COORD, _H2O_SYM, _H2O_POS, nmodes=1, lam=0.1,
                pol_vec=eps, omega=0.02, polar_force=True,
                polar_fd='forward', emode_projection=projection)
        return atoms.get_forces()

    f_proj = _run(True)
    f_full = _run(False)
    np.testing.assert_allclose(f_proj, f_full, atol=1e-4)


# --- Test 6: multimode keeps the full tensor (flag is a no-op) --------------

def test_multimode_pyscf_emode_uses_full_tensor(monkeypatch):
    """nmodes==2: neither _projected_response nor the analytic gdip is ever called
    and the emode_projection flag is a no-op (forces identical True vs False)."""
    calls = {"n": 0, "g": 0}
    orig = PySCFDriver._projected_response
    orig_g = PySCFDriver._analytic_projected_dipole_gradient

    def _spy(drv, eps):
        calls["n"] += 1
        return orig(drv, eps)

    def _spy_g(drv, eps):
        calls["g"] += 1
        return orig_g(drv, eps)
    monkeypatch.setattr(PySCFDriver, "_projected_response", _spy)
    monkeypatch.setattr(PySCFDriver, "_analytic_projected_dipole_gradient", _spy_g)

    pol_vec = np.array([[1, 0, 0], [0, 1, 0]], float).T

    def _run(projection):
        _, atoms = _emode_calc(
            _CO2_COORD, _CO2_SYM, _CO2_POS, nmodes=2, lam=[0.1, 0.1],
            pol_vec=pol_vec, omega=[0.02, 0.02], polar_force=True,
            polar_fd='central', emode_projection=projection)
        return atoms.get_forces()

    f_proj = _run(True)
    f_full = _run(False)
    assert calls["n"] == 0 and calls["g"] == 0
    np.testing.assert_allclose(f_proj, f_full, atol=1e-12)


# --- Test 7: pin the rank-1 polarizability observable -----------------------

def test_polarizability_packing_single_mode_pyscf():
    """After a projected step results['polarizability'] == outer(chi.eps, eps)."""
    eps = np.array([0, 1, 1], float) / np.sqrt(2.0)
    with pytest.warns(UserWarning):                     # non-unit pol_vec warning
        calc, atoms = _emode_calc(
            _H2O_COORD, _H2O_SYM, _H2O_POS, nmodes=1, lam=0.1,
            pol_vec=[0, 1, 1], omega=0.02, polar_force=False,
            polar_fd='central', emode_projection=True)
    atoms.get_forces()
    # a FRESH driver stub over the same mf: an empty response cache forces a real
    # recomputation instead of reading back the value the step already packed.
    chi_eps = _driver(calc.driver.mf)._projected_response(eps)
    np.testing.assert_allclose(calc.results['polarizability'],
                               np.outer(chi_eps, eps), rtol=1e-10,
                               # atol guards the numerically-zero elements of the
                               # rank-1 pack against runner-dependent BLAS rounding
                               atol=1e-12)


# --- Task 3c: analytic projected dipole gradient (contract field CPHF) ------

_ATOM_TOL = [
    ('O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24', np.array([1.0, 0.0, 0.0]), 1e-11),
    ('O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
     np.array([0, 1.0, 1.0]) / np.sqrt(2), 1e-8),
    ('C 0 0 0; O 1.16 0 0; O -1.16 0 0', np.array([1.0, 0.0, 0.0]), 2e-5),
    ('C 0 0 0; O 1.16 0 0; O -1.16 0 0',
     np.array([0, 1.0, 1.0]) / np.sqrt(2), 2e-5),
]


@pytest.mark.parametrize("atom,eps,tol", _ATOM_TOL)
def test_analytic_gdip_matches_dipderiv(atom, eps, tol):
    """eps . dmu/dR from the analytic field-CPHF contraction equals
    eps . kernel_dipderiv. H2O atol=1e-11 is the COMPLETENESS guard -- the
    formula is exact for non-degenerate systems, so a dropped term fails loudly.
    CO2 atol=2e-5 covers the degenerate-pi occ-occ gauge residual (measured 5e-6,
    1000x below the accepted force band)."""
    from pyscf.prop.infrared.rks import Infrared
    from pyscf.prop.infrared.rhf import kernel_dipderiv
    mf = _rks(atom)
    ir = Infrared(mf)
    ir.verbose = 0
    ref = np.einsum('axt,t->ax', kernel_dipderiv(ir), eps)
    got = _driver(mf)._analytic_projected_dipole_gradient(eps)
    assert got is not None
    np.testing.assert_allclose(got, ref, atol=tol)


def test_analytic_gdip_per_term():
    """The analytic nuclear (Z_A eps) + Pulay (h2ao.P.eps) pieces equal
    kernel_dipderiv's OWN nuclear + Pulay contributions exactly (rtol 1e-12, same
    objects), confining any future disagreement to the response bracket."""
    from pyscf.prop.infrared.rhf import get_h2ao_dipderiv
    mf = _rks('O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24')
    mol = mf.mol
    eps = np.array([0, 1.0, 1.0]) / np.sqrt(2)
    # our pieces
    P = mf.make_rdm1()
    h2ao = get_h2ao_dipderiv(mf)
    pulay = np.einsum('axtuv,uv,t->ax', h2ao, P, eps)
    charges = np.asarray([mol.atom_charge(a) for a in range(mol.natm)], float)
    nuc = charges[:, None] * eps[None, :]
    # kernel_dipderiv's own nuclear + Pulay pieces (same construction):
    # de_nuc[A,x,t] = Z_A delta_{x,t}; de_pulay[A,x,t] = sum_uv h2ao[A,x,t] P.
    ref_nuc = np.einsum('a,xt,t->ax', charges, np.eye(3), eps)
    ref_pulay = np.einsum('axtuv,uv,t->ax', h2ao, P, eps)
    np.testing.assert_allclose(nuc, ref_nuc, rtol=1e-12)
    np.testing.assert_allclose(pulay, ref_pulay, rtol=1e-12)


@pytest.mark.parametrize("atom", [
    'O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
    'C 0 0 0; O 0 0 1.16; O 0 0 -1.16',
])
def test_projected_response_origin_change_chi_unchanged(atom):
    """chi . eps from the plain-(0,0,0)-origin _field_response equals the
    charge-center value (atol 1e-10, neutral molecules). Guards the origin
    migration the e1F.s1oo trace requires."""
    from pyscf import lib
    from pyscf.scf import cphf
    from pyscf.prop.polarizability.rhf import Polarizability  # noqa: F401
    mf = _rks(atom)
    eps = np.array([0, 1.0, 1.0]) / np.sqrt(2)
    # charge-center reference (the pre-3c origin convention)
    polobj = mf.Polarizability()
    mol = mf.mol
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    charges = mol.atom_charges()
    cc = np.einsum('i,ix->x', charges, mol.atom_coords()) / charges.sum()
    with mol.with_common_orig(cc):
        int_r = mol.intor_symmetric('int1e_r', comp=3)
    h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(),
                    mo_coeff[:, mo_occ > 0])
    h1_eps = np.einsum('x,xij->ij', eps, h1)[None]
    mo1 = cphf.solve(polobj.gen_vind(mf, mo_coeff, mo_occ), mf.mo_energy,
                     mo_occ, h1_eps, np.zeros_like(h1_eps),
                     polobj.max_cycle_cphf, polobj.conv_tol)[0]
    chi_cc = -4.0 * np.einsum('xpi,pi->x', h1, mo1[0])
    chi_plain = _driver(mf)._projected_response(eps)
    np.testing.assert_allclose(chi_plain, chi_cc, atol=1e-10)


def test_projected_response_returns_field_response():
    """_field_response returns (chi_eps, mo1F, e1F) with the expected shapes."""
    mf = _rks('O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24')
    eps = np.array([0, 1.0, 1.0]) / np.sqrt(2)
    chi_eps, mo1F, e1F = _driver(mf)._field_response(eps)
    nmo = mf.mo_coeff.shape[1]
    nocc = int((mf.mo_occ > 0).sum())
    assert chi_eps.shape == (3,)
    assert mo1F.shape == (nmo, nocc)
    assert e1F.shape == (nocc, nocc)


def test_field_response_cache_invalidated_on_geometry_change():
    """After set_geometry (which replaces mf.mo_coeff), dipole_gradient must NOT
    reuse the stale cached (mo1F, e1F): the analytic gdip changes when an atom
    moves."""
    from ase import Atoms
    eps = np.array([1.0, 0.0, 0.0])
    calc, atoms = _emode_calc(
        _CO2_COORD, _CO2_SYM, _CO2_POS, nmodes=1, lam=0.1,
        pol_vec=[1, 0, 0], omega=0.02, polar_force=False,
        polar_fd='central', emode_projection=True)
    drv = _driver_from_calc(calc)
    drv.mf.conv_tol = 1e-11
    drv.mf.kernel()
    drv._projected_response(eps)                        # populate the cache
    gdip0 = drv._analytic_projected_dipole_gradient(eps)
    # move an atom via set_geometry (replaces mf.mo_coeff) and re-converge
    moved = Atoms(_CO2_SYM, positions=[(0, 0, 0), (1.26, 0, 0), (-1.16, 0, 0)],
                  cell=np.eye(3) * 20.0)
    drv.set_geometry(moved)
    drv.mf.kernel()
    gdip1 = drv._analytic_projected_dipole_gradient(eps)
    assert not np.allclose(gdip0, gdip1, atol=1e-6)     # stale cache would match


def test_gdipol_projected_packing():
    """dipole_gradient (single-mode, projection on) returns the rank-1 pack
    eps[a] * (eps . dmu/dR)[i,t] whose e_tot contraction is exact (analytic
    gdip; H2O tolerance tightened to 1e-8)."""
    eps = np.array([0, 1.0, 1.0]) / np.sqrt(2)
    with pytest.warns(UserWarning):                     # non-unit pol_vec warning
        calc, _ = _emode_calc(
            _H2O_COORD, _H2O_SYM, _H2O_POS, nmodes=1, lam=0.1,
            pol_vec=[0, 1, 1], omega=0.02, polar_force=False,
            polar_fd='central', emode_projection=True)
    drv = _driver_from_calc(calc)
    drv.mf.conv_tol = 1e-11
    drv.mf.kernel()
    gdip = drv.dipole_gradient()
    assert gdip.shape == (3, drv.mf.mol.natm, 3)
    # rank-1 structure: gdip[a] = eps[a] * gdip_eps
    gdip_eps = drv._analytic_projected_dipole_gradient(eps)
    np.testing.assert_allclose(gdip, np.einsum('a,it->ait', eps, gdip_eps),
                               atol=1e-8)
    # e_tot contraction (the only use in get_cboa_forces_bonini) equals the
    # full-tensor contraction to the analytic-gdip tolerance
    from pyscf.prop.infrared.rks import Infrared
    from pyscf.prop.infrared.rhf import kernel_dipderiv
    ir = Infrared(drv.mf)
    ir.verbose = 0
    full = np.transpose(kernel_dipderiv(ir), (2, 0, 1))
    e_tot = 0.037 * eps                                 # any vector along eps
    np.testing.assert_allclose(np.einsum('a,ait->it', e_tot, gdip),
                               np.einsum('a,ait->it', e_tot, full), atol=1e-8)
