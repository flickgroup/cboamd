"""store_observables writes 3 photon columns (qa, pa, Ea) per mode to photon.dat
(Task 7), so a multi-mode run records every photon coordinate."""
from types import SimpleNamespace

import numpy as np

from cboamd.helper_functions import store_observables


def _fake_atoms(nmodes, natom=3):
    pt = SimpleNamespace(qa=np.linspace(0.5, 0.5 + 0.1 * (nmodes - 1), nmodes),
                         pa=np.linspace(0.1, 0.1 + 0.1 * (nmodes - 1), nmodes),
                         ea=np.linspace(0.3, 0.3 + 0.1 * (nmodes - 1), nmodes))
    calc = SimpleNamespace(
        pt=pt, istep=0,
        p=SimpleNamespace(time=0.0),
        results={'dipole': np.array([0.1, 0.2, 0.3]),
                 'dipolepol': np.array([0.1, 0.2, 0.3]),
                 'polarizability': np.zeros((3, 3)),
                 'energy': -1.0,
                 'forces': np.zeros((natom, 3)),
                 'forces_bare': np.zeros((natom, 3))})
    atoms = SimpleNamespace(calc=calc, positions=np.zeros((natom, 3)))
    atoms.get_velocities = lambda: np.zeros((natom, 3))
    return atoms, natom


def _arrays(natom, nmodes, nsteps=1):
    return dict(
        dipole_array=np.zeros((nsteps, 8)),
        photon_array=np.zeros((nsteps, 2 + 3 * nmodes)),
        polarizability_array=np.zeros((nsteps, 11)),
        energy_array=np.zeros((nsteps, 3)),
        force_array=np.zeros((nsteps, 2 + 3 * natom)),
        force_bare_array=np.zeros((nsteps, 2 + 3 * natom)),
        position_array=np.zeros((nsteps, 2 + 6 * natom)))


def test_photon_row_has_three_columns_per_mode():
    nmodes = 2
    atoms, natom = _fake_atoms(nmodes)
    arr = _arrays(natom, nmodes)
    store_observables(atoms, 0, arr['dipole_array'], arr['photon_array'],
                      arr['polarizability_array'], arr['energy_array'],
                      arr['force_array'], arr['force_bare_array'],
                      arr['position_array'])
    row = arr['photon_array'][0]
    assert row.shape == (2 + 3 * nmodes,)
    assert row[0] == 0 and row[1] == 0.0          # istep, time
    # per-mode block: q_0,p_0,ea_0, q_1,p_1,ea_1
    np.testing.assert_allclose(row[2:], [0.5, 0.1, 0.3, 0.6, 0.2, 0.4])


def test_single_mode_row_unchanged_layout():
    atoms, natom = _fake_atoms(1)
    arr = _arrays(natom, 1)
    store_observables(atoms, 0, arr['dipole_array'], arr['photon_array'],
                      arr['polarizability_array'], arr['energy_array'],
                      arr['force_array'], arr['force_bare_array'],
                      arr['position_array'])
    np.testing.assert_allclose(arr['photon_array'][0], [0, 0.0, 0.5, 0.1, 0.3])
