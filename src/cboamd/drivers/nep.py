# src/cboamd/drivers/nep.py
"""NEP driver (calorine CPUNEP): energy/force, dipole, and polarizability models."""
import numpy as np
from ase import Atoms

from cboamd.atomic_constants import P_Har, P_a_B
from cboamd.drivers.base import Driver
from cboamd.drivers.registry import register_driver
from cboamd.helper_functions import (
    convert_dipole_value_to_au, convert_dipole_gradient_to_au,
    convert_polarizability_value_to_au, convert_polarizability_gradient_to_au)


@register_driver
class NEPDriver(Driver):
    name = 'nep'

    @classmethod
    def parse_input(cls, jdata, p, settings):
        p.nep_pot = jdata.get('nep_pot', None)
        p.nep_dip = jdata.get('nep_dip', None)
        p.nep_pol = jdata.get('nep_pol', None)
        settings['nep_pot'] = p.nep_pot
        settings['nep_dip'] = p.nep_dip
        settings['nep_pol'] = p.nep_pol

    def __init__(self, calc, coord, cell):
        super().__init__(calc, coord, cell)
        from calorine.calculators import CPUNEP   # lazy: optional extra
        symbols = [atom[0] for atom in coord]
        positions = [atom[1:][0] for atom in coord]

        self.calc_pot = CPUNEP(self.p.nep_pot)
        self.calc_dip = CPUNEP(self.p.nep_dip)
        self.calc_pol = CPUNEP(self.p.nep_pol)
        self.calc_pot.atoms = Atoms(symbols = symbols, positions = positions, cell = cell)
        self.calc_dip.atoms = Atoms(symbols = symbols, positions = positions, cell = cell)
        self.calc_pol.atoms = Atoms(symbols = symbols, positions = positions, cell = cell)

    def set_geometry(self, atoms):
        # the CPUNEP models are built once in MDCalculator.initialize; push the new
        # geometry here via set_atoms (NOT `calc.atoms = ...`): set_atoms clears
        # the cached results and the nepy backend, so energy/forces/dipole/
        # polarizability are recomputed for the new geometry. A plain attribute
        # assignment leaves calorine's cache stale -> every property frozen at the
        # initial geometry while the nuclei drift under a constant (stale) force.
        atoms_tmp = Atoms(symbols = atoms.symbols, positions = atoms.positions, cell = atoms.cell)
        self.calc_pot.set_atoms(atoms_tmp)
        self.calc_dip.set_atoms(atoms_tmp)
        self.calc_pol.set_atoms(atoms_tmp)

    def energy_and_dipole(self):
        e = self.calc_pot.get_potential_energy() / P_Har
        dipole = convert_dipole_value_to_au(self.calc, self.calc_dip.get_dipole_moment())
        return e, dipole

    def energy_gradient(self):
        return -self.calc_pot.get_forces() * (P_a_B / P_Har)  # pylint: disable=invalid-unary-operand-type

    def polarizability(self):
        return convert_polarizability_value_to_au(self.calc, self.calc_pol.get_polarizability())

    def dipole_gradient(self):
        # calorine returns (atom, nuclear-coord, dipole-component); transpose to the
        # (dipole-component, atom, dim) convention used by get_cboa_forces_bonini.
        gdipole_array = np.transpose(self.calc_dip.get_dipole_gradient(charge=0), (2, 0, 1))
        gdipole_array = convert_dipole_gradient_to_au(self.calc, gdipole_array)
        return gdipole_array

    def polarizability_gradient(self, natom):
        # Opt-in only: calorine's analytic get_polarizability_gradient is correct but
        # ~6x slower than the finite-difference fallback in pycalculator, so NEP
        # defaults to FD. calorine returns d(chi_ij)/dR as (atom, dim, i, j);
        # transpose to the (i, j, atom, dim) convention used by get_cboa_forces_bonini.
        if (self.pt.polar and self.pt.polar_force
                and self.calc_pol is not None
                and getattr(self.pt, 'polar_gradient_analytic', False)):
            gradient = np.transpose(self.calc_pol.get_polarizability_gradient(), (2, 3, 0, 1))
            return convert_polarizability_gradient_to_au(self.calc, gradient)
        return None
