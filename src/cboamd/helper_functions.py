# Unit normalizers/converters and observable IO shared by the engine and the
# drivers. Driver-specific physics lives in cboamd.drivers.*.
import numpy as np

from ase.units import kB

from cboamd.atomic_constants import P_a_B


def normalize_dipole_unit(unit):
    unit = str(unit).lower()
    aliases = {
        'au': 'ebohr',
        'atomic': 'ebohr',
        'atomic_unit': 'ebohr',
        'atomic_units': 'ebohr',
        'ebohr': 'ebohr',
        'e*bohr': 'ebohr',
        'e_bohr': 'ebohr',
        'eang': 'eangstrom',
        'eangstrom': 'eangstrom',
        'e*angstrom': 'eangstrom',
        'e_angstrom': 'eangstrom',
    }
    if unit not in aliases:
        raise ValueError(
            f"Unsupported dipole_unit {unit!r}; use 'au', 'eBohr', or 'eAngstrom'."
        )
    return aliases[unit]

def normalize_polarizability_unit(unit):
    unit = str(unit).lower()
    aliases = {
        'au': 'au',
        'atomic': 'au',
        'atomic_unit': 'au',
        'atomic_units': 'au',
        'bohr3': 'au',
        'bohr^3': 'au',
        'angstrom3': 'angstrom3',
        'angstrom^3': 'angstrom3',
        'a3': 'angstrom3',
    }
    if unit not in aliases:
        raise ValueError(
            f"Unsupported polarizability_unit {unit!r}; use 'au' or 'Angstrom3'."
        )
    return aliases[unit]

def convert_dipole_value_to_au(calc, dipole):
    dipole = np.asarray(dipole, dtype=float)
    if normalize_dipole_unit(getattr(calc.p, 'dipole_unit', 'au')) == 'eangstrom':
        return dipole / P_a_B
    return dipole

def convert_dipole_gradient_to_au(calc, gradient):
    gradient = np.asarray(gradient, dtype=float)
    if normalize_dipole_unit(getattr(calc.p, 'dipole_unit', 'au')) == 'eangstrom':
        return gradient
    return gradient * P_a_B

def convert_polarizability_value_to_au(calc, polar):
    polar = np.asarray(polar, dtype=float)
    if normalize_polarizability_unit(getattr(calc.p, 'polarizability_unit', 'au')) == 'angstrom3':
        return polar / (P_a_B ** 3)
    return polar

def convert_polarizability_gradient_to_au(calc, gradient):
    gradient = np.asarray(gradient, dtype=float)
    if normalize_polarizability_unit(getattr(calc.p, 'polarizability_unit', 'au')) == 'angstrom3':
        return gradient / (P_a_B ** 2)
    return gradient * P_a_B

def write_xyz_velocities(filename, symbols, velocities):
    #TODO: check units
    natm = len(symbols)
    f = open(filename, "w")
    f.write(str(natm))
    f.write('\n\n')

    for ii in range(0, natm):
        string = f'{symbols[ii]} {velocities[ii,0]} {velocities[ii,1]} {velocities[ii,2]}\n'
        f.write(string)
    f.close()

def printenergy(atoms):
    """Function to print the potential, kinetic and total energy"""
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)
    qa = atoms.calc.pt.qa
    pa = atoms.calc.pt.pa
    dipole = atoms.calc.results['dipole']
    td = atoms.calc.p.time
    istep = atoms.calc.istep

    # qa/pa are per-mode arrays now; print the scalar energetics with % and the
    # photon coordinate/momentum arrays via the f-string below.
    print('Step %i Time: %.3f Energy per atom: Epot = %.6e eV  Ekin = %.6e eV (T=%.6e K) Etot = %.6e eV, dipole_x %.6e, dipole_y %.6e, dipole_z %.6e' \
        % (istep, td, epot, ekin, ekin / (1.5 * kB), epot + ekin, \
            dipole[0], dipole[1], dipole[2]))
    print(f'time {atoms.calc.p.time} q {qa} p {pa}')

def store_observables(a, istep, dipole_array, photon_array, polarizability_array, energy_array, force_array, force_bare_array, position_array):
    qa = a.calc.pt.qa
    pa = a.calc.pt.pa
    ea = a.calc.pt.ea
    dipole = a.calc.results['dipole']
    dipolepol = a.calc.results['dipolepol']
    polarizability = a.calc.results['polarizability']
    td = a.calc.p.time
    istep = a.calc.istep
    # per-mode photon block: (q_0,p_0,ea_0, q_1,p_1,ea_1, ...) for M modes.
    photon_block = np.column_stack(
        [np.atleast_1d(qa), np.atleast_1d(pa), np.atleast_1d(ea)]).ravel()
    photon_array[istep, :] = np.concatenate(([istep, td], photon_block))
    dipole_array[istep, :] = [istep, td, dipolepol[0], dipolepol[1], dipolepol[2], dipole[0], dipole[1], dipole[2]]
    polarizability_array[istep,:] = [istep, td, polarizability[0,0], polarizability[0,1], polarizability[0,2], \
                                        polarizability[1,0], polarizability[1,1], polarizability[1,2],        \
                                        polarizability[2,0], polarizability[2,1], polarizability[2,2]]
    energy_array[istep, :] = [istep, td, a.calc.results['energy']]
    force_array[istep, :] = np.concatenate(([istep, td], a.calc.results['forces'].ravel()))
    force_bare_array[istep, :] = np.concatenate(([istep, td], a.calc.results['forces_bare'].ravel()))
    position_array[istep, :] = np.concatenate((np.concatenate(([istep, td], a.positions.ravel())),
                                               a.get_velocities().ravel()))

_OBSERVABLE_FILES = (
    ('dipole.dat', 'Step, Time, dipole_x, dipole_y, dipole_z (all in a.u.)'),
    ('photon.dat', 'Step, Time, then (qa, pa, Ea) per photon mode (all in a.u.)'),
    ('polarizability.dat', 'Step, Time, polxx, polxy, polxz, polxy, polyy, polyz, polzx, polzy, polzz (all in a.u.)'),
    ('energy.dat', 'Step, Time, Energy in eV'),
    ('force.dat', 'Step, Time, Forces on each atom and dimension in [eV/A]'),
    # force_bare.dat is written ONLY when it carries non-redundant information,
    # i.e. e-mode with photons on and some nonzero lambda (see save_observables).
    # For photons=False or lambda==0 the bare force equals force.dat, and in
    # q-mode no zero-field force exists (the SCF is cavity-polarized), so the
    # file is skipped entirely rather than duplicating force.dat or dumping NaN.
    ('force_bare.dat', 'Step, Time, Bare (zero-field) forces on each atom and dimension in [eV/A]'),
    ('position.dat', 'Step, Time, Coordinates [A], Velocities [fs/A?]'),
)

def save_observables(atoms, dipole_array, photon_array, polarizability_array, energy_array, force_array, force_bare_array, position_array, dump_state):
    """Append observable rows accumulated since the last dump to the .dat files.

    Called periodically (and once at the end) rather than every step: each call
    appends only the new rows [dump_state['last']+1 : istep+1] and writes the
    header once when each file is first created. This is O(N) total I/O over a
    run, versus the O(N^2) of rewriting the whole array on every step.
    """
    istep = atoms.calc.istep
    start = dump_state['last'] + 1
    if istep < start:        # nothing new since the previous dump
        return
    sl = slice(start, istep + 1)
    arrays = (dipole_array, photon_array, polarizability_array,
              energy_array, force_array, force_bare_array, position_array)
    # Only write force_bare.dat when the bare (zero-field) force is a distinct,
    # meaningful quantity: e-mode, photons on, and at least one nonzero lambda.
    # Otherwise it either duplicates force.dat (photons=False / lambda==0) or has
    # no bare counterpart (q-mode's cavity-polarized SCF), so skip the file.
    pt = atoms.calc.pt
    write_bare = (atoms.calc.p.photons
                  and getattr(pt, 'cboa_mode', 'e-mode') != 'q-mode'
                  and bool(np.any(np.asarray(pt.lam) != 0.0)))
    for (fname, header), arr in zip(_OBSERVABLE_FILES, arrays):
        if fname == 'force_bare.dat' and not write_bare:
            continue
        if start == 0:       # first dump: (over)write with the header
            np.savetxt(fname, arr[sl], header=header)
        else:                # subsequent dumps: append rows only
            with open(fname, 'a') as fh:
                np.savetxt(fh, arr[sl])
    dump_state['last'] = istep

