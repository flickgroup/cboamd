"""Cross-cutting identity guard for the pyscf driver's e-mode optimizations
(1-RHS projected polarizability, analytic projected dipole gradient, forward-FD
dchi/dR, SCF warm-start).

Same-session A/B: three 30-step pyscf CO2 e-mode trajectories (sto3g,
conv_tol=1e-11, lambda=0.1, polar_force=True) in one process, so BLAS, grids
and pyscf version are identical between arms:

  A legacy      central FD, emode_projection off, scf_warmstart off
                (byte-equivalent to the pre-optimization code for coupled runs)
  B exact-opt   central FD, projection ON with the analytic dipole gradient
                pinned to the full kernel_dipderiv path, warmstart off
                -> differs from A only in exact-path components (1-RHS chi,
                rank-1 packing); must reproduce A to ~1e-10
  C production  forward FD + projection incl. analytic dipole gradient + warmstart
                (the shipped defaults) -> documented tolerance band, and NVE
                drift must not exceed legacy

Measured on implementation (30 steps, Task 3c analytic gdip): B vs A 3.7e-10 eV /
6.8e-11 A; C vs A 1.6e-6 eV / 1.1e-6 A (band unchanged from 3b -- the forward-FD
dchi/dR error dominates, not the gdip route); drift ratio 1.00.

A committed golden (data/co2_sto3g_lam0.1_legacy.dat, generated from arm A by
generate_golden.py) anchors the absolute numbers loosely across
machines/pyscf versions (energies 1e-6 Ha, positions 1e-5 A) -- regenerate it
on pyscf upgrades.
"""
import os

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("pyscf.prop")

from ase import Atoms  # noqa: E402
from ase import units  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402
from pyscf import gto, dft  # noqa: E402

from cboamd import pycalculator as pc  # noqa: E402
from cboamd.drivers.pyscf_driver import PySCFDriver  # noqa: E402
from cboamd.photon_mode import photon_mode  # noqa: E402

NSTEPS = 30
COORD = [['C', (0.0, 0.0, 0.0)], ['O', (1.16, 0.0, 0.0)], ['O', (-1.16, 0.0, 0.0)]]
_DATA = os.path.join(os.path.dirname(__file__), 'data', 'co2_sto3g_lam0.1_legacy.dat')


def _build(polar_fd, projection, warmstart):
    p = pc.parameters()
    p.driver = 'pyscf'
    p.photons = True
    p.dt = 20.0
    p.time = -p.dt
    p.deltax = 1e-3
    p.dipole_unit = 'au'
    p.polarizability_unit = 'au'
    p.polar_fd = polar_fd
    p.conv_tol = 1e-11
    p.scf_warmstart = warmstart
    p.emode_projection = projection
    pt = photon_mode(1, 0.0111, 0.1, [1, 0, 0], qa=2.0)
    pt.cboa_mode = 'e-mode'
    pt.polar = True
    pt.update_polar = True
    pt.polar_force = True
    pt.polar_value = np.zeros((3, 3))
    pt.integral = 0
    pt.analytic_nuclear_gradient = True
    calc = pc.MDCalculator(coord=COORD, cell=np.eye(3) * 20.0, p=p, pt=pt)
    calc.driver.mf = dft.RKS(gto.M(atom=COORD, basis='sto3g', verbose=0))
    calc.driver.mf.verbose = 0
    calc.driver.mf.conv_tol = p.conv_tol
    return calc


def _run(calc, nsteps=NSTEPS):
    atoms = Atoms('CO2', positions=[(0, 0, 0), (1.16, 0, 0), (-1.16, 0, 0)],
                  cell=np.eye(3) * 20.0)
    atoms.calc = calc
    dyn = VelocityVerlet(atoms, timestep=calc.p.dt * units.fs / 41.341374575751)
    energies = []
    dyn.attach(lambda: energies.append(
        atoms.get_potential_energy() + atoms.get_kinetic_energy()), interval=1)
    dyn.run(nsteps)
    return np.array(energies), atoms.get_positions()


@pytest.fixture(scope='module')
def arms(request):
    """Run the three arms once for the whole module."""
    e_a, x_a = _run(_build('central', False, False))
    orig = PySCFDriver._analytic_projected_dipole_gradient
    # arm B: gdip -> full kernel_dipderiv
    PySCFDriver._analytic_projected_dipole_gradient = lambda *a, **k: None
    try:
        e_b, x_b = _run(_build('central', True, False))
    finally:
        PySCFDriver._analytic_projected_dipole_gradient = orig
    e_c, x_c = _run(_build('forward', True, True))
    return dict(A=(e_a, x_a), B=(e_b, x_b), C=(e_c, x_c))


@pytest.mark.slow
def test_exact_optimizations_reproduce_legacy(arms):
    """Arm B (1-RHS chi + rank-1 packing, everything else legacy) is a strict
    identity with legacy over 30 steps: the Task-3 core changes nothing."""
    e_a, x_a = arms['A']
    e_b, x_b = arms['B']
    np.testing.assert_allclose(e_b, e_a, atol=1e-9)    # eV; measured 1.9e-11
    np.testing.assert_allclose(x_b, x_a, atol=1e-8)    # A;  measured 1.3e-10


@pytest.mark.slow
def test_production_defaults_within_band_and_driftfree(arms):
    """Arm C (shipped defaults: forward FD + projection + FF gdip + warmstart)
    stays in the documented tolerance band and conserves energy as well as
    legacy (drift ratio ~1)."""
    e_a, x_a = arms['A']
    e_c, x_c = arms['C']
    np.testing.assert_allclose(e_c, e_a, atol=1e-4)    # eV; measured 1.6e-6
    np.testing.assert_allclose(x_c, x_a, atol=1e-4)    # A;  measured 1.1e-6
    drift_a = np.abs(e_a - e_a[0]).max()
    drift_c = np.abs(e_c - e_c[0]).max()
    assert drift_c < 2.0 * drift_a + 1e-9              # measured ratio 1.00


@pytest.mark.slow
def test_golden_anchor(arms):
    """Loose absolute anchor against the committed legacy reference (guards
    accidental physics changes across sessions; regenerate on pyscf upgrades
    with generate_golden.py)."""
    e_a, x_a = arms['A']
    ref = np.loadtxt(_DATA)                # flat: NSTEPS+1 energies, 9 positions
    e_ref, x_ref = ref[:NSTEPS + 1], ref[NSTEPS + 1:].reshape(3, 3)
    np.testing.assert_allclose(e_a, e_ref, atol=2.7e-5)   # 1e-6 Ha in eV
    np.testing.assert_allclose(x_a, x_ref, atol=1e-5)     # A
