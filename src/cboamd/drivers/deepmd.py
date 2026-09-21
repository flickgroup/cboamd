# src/cboamd/drivers/deepmd.py
"""DeepMD-kit driver: energy/force, dipole, and polarizability models."""
import numpy as np
from ase import Atoms

from cboamd.atomic_constants import P_Har, P_a_B
from cboamd.constant_polar import ConstantPolar
from cboamd.drivers.base import Driver
from cboamd.drivers.registry import register_driver
from cboamd.helper_functions import (
    convert_dipole_value_to_au, convert_dipole_gradient_to_au,
    convert_polarizability_value_to_au, convert_polarizability_gradient_to_au)


def reshape_deepmd_value(value, ncomp):
    return np.asarray(value, dtype=float).reshape(-1)[:ncomp]


def reshape_deepmd_gradient(gradient, ncomp, natom):
    gradient = np.asarray(gradient, dtype=float)
    if gradient.size != ncomp * natom * 3:
        raise ValueError(
            f"Unexpected DeepMD tensor gradient size {gradient.size}; "
            f"expected {ncomp * natom * 3}."
        )
    return gradient.reshape(ncomp, natom, 3)


@register_driver
class DeepMDDriver(Driver):
    name = 'deepmd'

    @classmethod
    def parse_input(cls, jdata, p, settings):
        # moved verbatim from pymd_ase.py lines 117-125
        p.dp_pot = jdata.get('dp_pot', None)
        p.dp_dip = jdata.get('dp_dip', None)
        p.dp_pol = jdata.get('dp_pol', None)
        p.type_map = jdata.get('type_map', jdata.get('deepmd_type_map', None))
        settings['dp_pot'] = p.dp_pot
        settings['dp_dip'] = p.dp_dip
        settings['dp_pol'] = p.dp_pol
        settings['type_map'] = p.type_map

    def __init__(self, calc, coord, cell):
        super().__init__(calc, coord, cell)
        # moved verbatim from pycalculator.py lines 87-115
        try:
            from deepmd.calculator import DP
            from deepmd.infer.deep_dipole import DeepDipole
            from deepmd.infer.deep_polar import DeepPolar
        except ImportError as exc:
            raise ImportError("DeepMD driver requires deepmd-kit with DP, DeepDipole, and DeepPolar.") from exc
        # Extract atomic symbols and coordinates
        symbols = [atom[0] for atom in coord]
        positions = [atom[1:][0] for atom in coord]
        type_map = self.p.type_map or sorted(set(symbols))
        if isinstance(type_map, dict):
            type_dict = type_map
        else:
            type_dict = {symbol: idx for idx, symbol in enumerate(type_map)}
        self.deepmd_atom_types = [type_dict[symbol] for symbol in symbols]
        pbc = not np.allclose(cell, 0.0)
        self.calc_pot = Atoms(symbols, positions, cell=cell, pbc=pbc, calculator=DP(model=self.p.dp_pot))
        self.calc_dip = DeepDipole(self.p.dp_dip) if self.p.dp_dip is not None else None
        if self.p.dp_pol is None:
            self.calc_pol = None
        elif str(self.p.dp_pol).endswith('.npz'):
            # constant-polarizability stand-in (no TensorFlow / trained graph)
            self.calc_pol = ConstantPolar(self.p.dp_pol)
        else:
            self.calc_pol = DeepPolar(self.p.dp_pol)
        if self.p.photons and self.calc_dip is None:
            raise ValueError("DeepMD photon dynamics requires a dipole model via dp_dip.")
        if self.pt.polar and self.calc_pol is None:
            raise ValueError("DeepMD polarizability-on dynamics requires a polar model via dp_pol.")
        self._tensor_cache = {}

    def set_geometry(self, atoms):
        # body: helper_functions.init_geo deepmd branch (lines 194-196)
        self.calc_pot.positions = atoms.positions
        self.calc_pot.cell = atoms.cell
        self._tensor_cache = {}

    # --- private glue, moved from helper_functions.py lines 641-693 ---
    def _coords_cells(self):
        # get_deepmd_coords_cells body; returns (coords, cells, self.deepmd_atom_types)
        coords = np.asarray(self.calc_pot.get_positions()).reshape(1, -1)
        cell = np.asarray(self.calc_pot.get_cell())
        cells = None if np.allclose(cell, 0.0) else cell.reshape(1, 9)
        return coords, cells, self.deepmd_atom_types

    def _energy_and_forces(self):
        # get_deepmd_energy_and_forces body, cache key "potential" on self._tensor_cache
        if "potential" in self._tensor_cache:
            return self._tensor_cache["potential"]
        energy = self.calc_pot.get_potential_energy() / P_Har
        gfkernel = -self.calc_pot.get_forces() * (P_a_B / P_Har)
        self._tensor_cache["potential"] = (energy, gfkernel)
        return self._tensor_cache["potential"]

    def _dipole_and_gradient(self):
        # get_deepmd_dipole_and_gradient body, cache key "dipole";
        # unit converters are called as convert_dipole_value_to_au(self.calc, ...)
        if "dipole" in self._tensor_cache:
            return self._tensor_cache["dipole"]
        coords, cells, atom_types = self._coords_cells()
        dipole, gradient, _ = self.calc_dip.eval_full(
            coords=coords, cells=cells, atom_types=atom_types, atomic=False
        )
        dipole = convert_dipole_value_to_au(self.calc, reshape_deepmd_value(dipole, 3))
        gradient = -convert_dipole_gradient_to_au(
            self.calc, reshape_deepmd_gradient(gradient, 3, len(atom_types))
        )
        self._tensor_cache["dipole"] = (dipole, gradient)
        return self._tensor_cache["dipole"]

    def _polarizability_and_gradient(self):
        # get_deepmd_polarizability_and_gradient body, cache key "polarizability"
        if "polarizability" in self._tensor_cache:
            return self._tensor_cache["polarizability"]
        coords, cells, atom_types = self._coords_cells()
        polar, gradient, _ = self.calc_pol.eval_full(
            coords=coords, cells=cells, atom_types=atom_types, atomic=False
        )
        polar = convert_polarizability_value_to_au(self.calc, reshape_deepmd_value(polar, 9)).reshape(3, 3)
        gradient = -convert_polarizability_gradient_to_au(
            self.calc, reshape_deepmd_gradient(gradient, 9, len(atom_types))
        ).reshape(3, 3, len(atom_types), 3)
        self._tensor_cache["polarizability"] = (polar, gradient)
        return self._tensor_cache["polarizability"]

    # --- Driver interface ---
    def energy_and_dipole(self):
        # helper_functions.get_energy_and_dipole deepmd branch (lines 264-270)
        e, _ = self._energy_and_forces()
        if self.calc_dip is None:
            return e, np.zeros(3)
        dipole, _ = self._dipole_and_gradient()
        return e, dipole

    def energy_gradient(self):
        return self._energy_and_forces()[1]

    def polarizability(self):
        # get_polar deepmd branch (lines 493-502) verbatim, with
        # get_deepmd_polarizability_and_gradient(calc)[0] -> self._polarizability_and_gradient()[0]
        if self.pt.polar and self.pt.update_polar:
            polar = self._polarizability_and_gradient()[0]
        elif self.pt.polar and not self.pt.update_polar:
            if abs(sum(sum(self.pt.polar_value - np.zeros((3, 3))))) < 1e-12:
                polar = self._polarizability_and_gradient()[0]
            else:
                polar = self.pt.polar_value
        else:
            polar = self.pt.polar_value
        return polar

    def dipole_gradient(self):
        # get_gdipol deepmd branch (lines 568-572)
        if self.calc_dip is None:
            return np.zeros((3, len(self.calc_pot), 3))
        return self._dipole_and_gradient()[1]

    def polarizability_gradient(self, natom):
        # get_gpolarizability deepmd branch (lines 613-615)
        if self.pt.polar and self.pt.polar_force and self.calc_pol is not None:
            return self._polarizability_and_gradient()[1]
        return None
