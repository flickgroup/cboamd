"""Driver interface: everything the engine needs from a force/property backend.

All quantities cross this interface in Hartree atomic units with the engine's
sign convention: energy_gradient() returns +dE/dR (Ha/Bohr) and the ENGINE
negates it into forces. Dipoles are e*Bohr, polarizabilities Bohr^3; gradients
are (3, natom, 3) and (3, 3, natom, 3) with the property component first.
"""
import numpy as np


class Driver:
    """Base class for force/property backends ("drivers").

    A driver is constructed once per run with the owning calculator (giving it
    read access to calc.p / calc.pt) and the initial geometry; afterwards it
    only sees geometry updates through set_geometry().
    """

    # registry key; also the input file's "driver" value.
    name = None
    # CBOA run modes this driver supports. 'q-mode' needs a self-consistent
    # cavity SCF, so force-engine drivers stay e-mode only.
    supported_cboa_modes = ('e-mode',)

    @classmethod
    def parse_input(cls, jdata, p, settings):
        """Read driver-specific input keys from jdata onto p, mirror in settings."""

    def __init__(self, calc, coord, cell):
        self.calc = calc
        self.p = calc.p
        self.pt = calc.pt

    def set_geometry(self, atoms):
        raise NotImplementedError

    def energy_and_dipole(self):
        """Return (energy [Ha], dipole [(3,) e*Bohr])."""
        raise NotImplementedError

    def energy_gradient(self):
        """Return dE/dR [(natom, 3) Ha/Bohr] (NOT the force)."""
        raise NotImplementedError

    def polarizability(self):
        """Return the (3, 3) polarizability [Bohr^3], honoring pt.polar/update_polar."""
        return np.zeros((3, 3))

    def dipole_gradient(self):
        """Return dmu/dR [(3, natom, 3) a.u.]."""
        raise NotImplementedError

    def polarizability_gradient(self, natom):
        """Return dchi/dR [(3, 3, natom, 3) a.u.], or None to use the engine's FD loop."""
        return None

    def refresh_displaced(self):
        """Called after set_geometry() inside the engine's FD polarizability loop,
        before polarizability(); SCF drivers re-solve here."""
