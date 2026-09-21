import argparse
import json
import time
import numpy as np

from ase import units
import ase.io as aio
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory

from cboamd.pycalculator import parameters, MDCalculator
from cboamd.photon_mode import photon_mode

from cboamd.atomic_constants import P_Ang, P_Har, P_pe
from cboamd.helper_functions import write_xyz_velocities, printenergy, store_observables, save_observables
from cboamd.helper_functions import normalize_dipole_unit, normalize_polarizability_unit

from cboamd.drivers.registry import get_driver_class
# driver modules register themselves on import of the drivers package
# (see cboamd/drivers/__init__.py); importing get_driver_class
# above is enough to trigger it.


def coord_from_ase(atoms):
    """[[symbol, (x, y, z)], ...] in Angstrom (atom-list format, ase-only)."""
    return [[s, tuple(pos)] for s, pos in
            zip(atoms.get_chemical_symbols(), atoms.get_positions())]


def validate_cboa_mode(cboa_mode, driver):
    """Validate the CBOA run mode against the driver's declared support.

    "e-mode" (field/E-based, default) works with every driver; "q-mode"
    (photon-coordinate, self-consistent SCF) modifies the SCF and is only
    supported by drivers that declare it. Raises ValueError on an unknown
    mode or an unsupported pairing.
    """
    if cboa_mode not in ('e-mode', 'q-mode'):
        raise ValueError(
            f"cboa_mode must be 'e-mode' or 'q-mode', got {cboa_mode!r}.")
    cls = get_driver_class(driver)
    if cboa_mode not in cls.supported_cboa_modes:
        raise ValueError(
            f"cboa_mode={cboa_mode!r} is not supported by driver {driver!r} "
            f"(supported: {list(cls.supported_cboa_modes)}).")
    return cboa_mode


def build_photon_mode(jdata, cboa_mode):
    """Construct a (possibly multi-mode) photon_mode from the input dict.

    nphoton sets the number of modes M; omega_photon/lambda_photon/lambda_vector
    must each provide M entries. qa0/pa0 may be a scalar (broadcast to all modes)
    or a length-M list. photon_mode stores pol_vec as (3, M); the JSON
    lambda_vector is (M, 3), so it is transposed (a 1-D vector is passed for the
    single-mode case and the constructor makes it (3, 1)). All other per-run knobs
    mirror the historical single-mode defaults.
    """
    nphoton = int(jdata.get('nphoton', 1))
    omega = jdata.get('omega_photon', [0.01107])
    lam = jdata.get('lambda_photon', [0.05])
    pol = jdata.get('lambda_vector', [[1, 0, 0]])
    for name, seq in (('omega_photon', omega), ('lambda_photon', lam),
                      ('lambda_vector', pol)):
        if len(seq) != nphoton:
            raise ValueError(
                f"{name} has {len(seq)} entries, expected nphoton={nphoton}.")
    qa0 = np.full(nphoton, 0.0) if jdata.get('qa0', 0) == 0 \
        else np.broadcast_to(np.asarray(jdata['qa0'], float), (nphoton,)).copy()
    pa0 = np.full(nphoton, 0.0) if jdata.get('pa0', 0) == 0 \
        else np.broadcast_to(np.asarray(jdata['pa0'], float), (nphoton,)).copy()

    pol_arg = pol[0] if nphoton == 1 else np.asarray(pol, dtype=float).T
    pt = photon_mode(nphoton, omega, lam, pol_arg, qa=qa0, pa=pa0)
    pt.cboa_mode = cboa_mode
    # q-mode nuclear forces: analytic Hellmann-Feynman gradient (default) or the
    # finite-difference fallback. No effect in e-mode.
    pt.analytic_nuclear_gradient = jdata.get('analytic_nuclear_gradient', True)
    pt.integral = 0
    pt.qa0 = qa0
    pt.pa0 = pa0
    pt.ea = np.zeros(nphoton)
    pt.polar = jdata.get('polar', True)
    pt.update_polar = jdata.get('update_polar', True)
    pt.polar_force = jdata.get('polar_force', True)
    # NEP only: use calorine's analytic polarizability gradient instead of the
    # (faster) finite-difference fallback. Off by default -- analytic is ~6x slower.
    pt.polar_gradient_analytic = jdata.get('polar_gradient_analytic', False)
    pt.polar_value = np.zeros((3, 3))
    return pt


def main():
    parser = argparse.ArgumentParser(description="Parse input file name.")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="in.json",
        help="Name of the input file (default: in.json)"
    )

    args = parser.parse_args()
    input_file = args.input
    jdata = json.load(open(input_file, 'r'))
    settings = jdata.copy()

    xyz_file = jdata.get('xyz_file', None)
    settings['xyz_file'] = xyz_file
    if xyz_file is None:
        raise ValueError("xyz_file must be specified in input file.")
    else:
        atoms = aio.read(xyz_file)
        cell = atoms.get_cell()
        if np.all(cell == 0):
            raise ValueError("Cell is not set in xyz file. Please set cell in xyz file.")

    p = parameters()
    p.driver = jdata.get('driver')
    if p.driver is None:
        raise ValueError("driver must be specified in input file.")
    p.photons = jdata.get('photons', False)
    settings['driver'] = p.driver
    settings['photons'] = p.photons
    p.dipole_unit = jdata.get('dipole_unit', 'au')
    p.polarizability_unit = jdata.get('polarizability_unit', 'au')
    normalize_dipole_unit(p.dipole_unit)
    normalize_polarizability_unit(p.polarizability_unit)
    settings['dipole_unit'] = p.dipole_unit
    settings['polarizability_unit'] = p.polarizability_unit

    driver_cls = get_driver_class(p.driver)
    driver_cls.parse_input(jdata, p, settings)

    td_steps = jdata.get('steps', 1000)
    conv_fs_au = units.fs*P_Ang*np.sqrt(P_pe*P_Har)
    p.dt = jdata.get('timestep', 0.5) * conv_fs_au #parameter class needs au (0.5 * conv_fs_au will give 0.5 fs)
    p.time = -1*p.dt #start at -1, to be consistent with istep
    p.deltax = jdata.get('deltax', 0.001)  # finite-difference step in Angstrom (applied to ASE positions)
    # dchi/dR finite-difference scheme for the e-mode polarizability gradient:
    # 'forward' (default, half the displaced solves) | 'central' (exact prior cost).
    p.polar_fd = jdata.get('polar_fd', 'forward')

    settings['steps'] = td_steps
    settings['timestep'] = p.dt / conv_fs_au * units.fs  # convert to fs for ASE
    settings['deltax'] = p.deltax  # already in Angstrom
    settings['polar_fd'] = p.polar_fd

    cboa_mode = validate_cboa_mode(jdata.get('cboa_mode', 'e-mode'), p.driver)
    settings['cboa_mode'] = cboa_mode

    nphoton = jdata.get('nphoton', 1)
    omega_photon = jdata.get('omega_photon', [0.01107])
    lambda_photon = jdata.get('lambda_photon', [0.05])
    lambda_vector = jdata.get('lambda_vector', [[1, 0, 0]])

    settings['nphoton'] = nphoton
    settings['omega_photon'] = omega_photon
    settings['lambda_photon'] = lambda_photon
    settings['lambda_vector'] = lambda_vector

    pt = build_photon_mode(jdata, cboa_mode)
    settings['analytic_nuclear_gradient'] = pt.analytic_nuclear_gradient

    # qa0/pa0 are now numpy arrays (per mode); record JSON-native lists.
    settings['qa0'] = pt.qa0.tolist()
    settings['pa0'] = pt.pa0.tolist()
    settings['polar'] = pt.polar
    settings['update_polar'] = pt.update_polar
    settings['polar_force'] = pt.polar_force
    settings['polar_gradient_analytic'] = pt.polar_gradient_analytic

    with open("settings.json", "w") as f:
        json.dump(settings, f, indent=4)

    # atom-list format: [[symbol, (x, y, z)], ...]
    atom_coordinates = coord_from_ase(atoms)
    atoms.calc = MDCalculator(coord = atom_coordinates, cell = cell, p=p, pt=pt)

    #from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    #MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    write_xyz_velocities('velocities.xyz',atoms.get_chemical_symbols(), atoms.get_velocities())
    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    dyn = VelocityVerlet(atoms, timestep = p.dt / conv_fs_au * units.fs,
                        logfile='md.log')  # ase needs fs.

    dipole_array = np.zeros((td_steps + 1, 8))
    polarizability_array = np.zeros((td_steps + 1, 11))
    photon_array = np.zeros((td_steps + 1, 2 + 3 * pt.nmodes))  # istep, time, (q,p,Ea) per mode
    energy_array = np.zeros((td_steps + 1, 3))
    force_array = np.zeros((td_steps + 1, 3*atoms.get_global_number_of_atoms() + 2))
    force_bare_array = np.zeros((td_steps + 1, 3*atoms.get_global_number_of_atoms() + 2))
    position_array = np.zeros((td_steps + 1, 6*atoms.get_global_number_of_atoms() + 2))

    # Now run the dynamics
    def energy_observer():
        printenergy(atoms)
    def observables_observer():
        store_observables(atoms, atoms.calc.istep, dipole_array, photon_array, polarizability_array, energy_array, force_array, force_bare_array, position_array)
    # observables are dumped every 100 steps (and once at the end) by appending
    # only the rows accumulated since the last dump -- O(N) total I/O instead of
    # rewriting the whole array every step (O(N^2)).
    dump_state = {'last': -1}
    def observables_dump_observer():
        save_observables(atoms, dipole_array, photon_array, polarizability_array, energy_array, force_array, force_bare_array, position_array, dump_state)

    # dyn is the dynamics (e.g. VelocityVerlet, Langevin or similar)
    traj = Trajectory('md.traj', 'w', atoms)

    #from ase.io.trajectory import Trajectory
    #traj = Trajectory('md.traj')
    #for atoms in traj:
    #    print(atoms.positions)
    #    print(atoms.get_velocities())

    dyn.attach(energy_observer)
    dyn.attach(observables_observer)
    dyn.attach(observables_dump_observer, interval = 100)
    dyn.attach(traj.write)#, interval = 4)

    tstart = time.time()
    dyn.run(td_steps)
    observables_dump_observer()   # final flush of rows since the last interval dump
    tend = time.time()

    print(f"Driver: {p.driver}")
    print(f"Steps: {td_steps}")
    print(f"Times: {tend - tstart:.2f} s.")
    traj.close()

if __name__ == "__main__":
    main()
