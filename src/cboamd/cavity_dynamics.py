"""ASE Langevin dynamics that can also thermostat the cavity photon modes.

The cavity photon coordinates (``pt.qa``, ``pt.pa``) are velocity-Verlet
integrated *inside* ``MDCalculator.calculate()``, so ASE integrators never see
them: under plain ``ase.md.langevin.Langevin`` the atoms are thermostatted
while the photons evolve micro-canonically. At ``lambda = 0`` the photons then
never thermalize at all; at finite coupling they only thermalize indirectly
through the molecules. Codes that treat the photons as ordinary
(mass-1-a.u.) thermostatted degrees of freedom instead let the NVT
thermostat act on them like on any nucleus.

:class:`CavityLangevin` restores that parity behind an explicit flag: with
``thermostat_photons=True`` every MD step is followed by the exact
Ornstein-Uhlenbeck momentum update

    pa <- c1 * pa + c2 * xi,   c1 = exp(-friction * dt),
                               c2 = sqrt(kB T (1 - c1^2))   [photon mass 1 a.u.]

using the same friction, target temperature, and RNG as the atomic
thermostat. Combined with the velocity-Verlet photon step inside the
calculator this is a standard Langevin (BAOA-type) splitting, i.e. the same
quality of sampling as thermostatting atoms and photons side by side. With the flag off
(default) the class is exactly ``Langevin``.
"""

import numpy as np

from ase.md.langevin import Langevin

from cboamd.atomic_constants import P_Har


class CavityLangevin(Langevin):
    """Langevin NVT for atoms + optional OU thermostat on the photon modes.

    Parameters (in addition to ``ase.md.langevin.Langevin``):

    thermostat_photons : bool, default False
        When True, apply the exact Ornstein-Uhlenbeck update to the photon
        momenta ``atoms.calc.pt.pa`` after each MD step, at the thermostat's
        own temperature and friction. Requires the attached calculator to be
        an ``MDCalculator`` with a ``photon_mode`` (``calc.pt``).

    The photon momenta are in atomic units (photon mass 1 a.u.), so the OU
    noise amplitude uses kB*T in Hartree; friction*dt is dimensionless and is
    taken directly from the ASE-unit thermostat settings.
    """

    def __init__(self, atoms, timestep, thermostat_photons=False, **kwargs):
        super().__init__(atoms, timestep, **kwargs)
        self.thermostat_photons = bool(thermostat_photons)
        if self.thermostat_photons and getattr(atoms.calc, "pt", None) is None:
            raise ValueError(
                "thermostat_photons=True needs an MDCalculator with photon "
                "modes (atoms.calc.pt) attached before constructing the "
                "dynamics.")
        self._update_photon_ou()

    def _update_photon_ou(self):
        """(Re)derive the OU coefficients from the current dt/friction/temp.

        self.fr is in 1/ASE-time, self.dt in ASE-time (product dimensionless);
        self.temp is kB*T in eV -> convert to Hartree for the a.u. momenta.
        """
        c1 = np.exp(-float(np.max(self.fr)) * self.dt)
        kbt_au = float(np.max(self.temp)) / P_Har
        self._ph_c1 = c1
        self._ph_c2 = np.sqrt(kbt_au * (1.0 - c1 * c1))

    def set_friction(self, friction):
        super().set_friction(friction)
        self._update_photon_ou()

    def set_temperature(self, temperature=None, temperature_K=None):
        super().set_temperature(temperature=temperature,
                                temperature_K=temperature_K)
        self._update_photon_ou()

    def set_timestep(self, timestep):
        super().set_timestep(timestep)
        self._update_photon_ou()

    def step(self, forces=None):
        forces = super().step(forces)
        if self.thermostat_photons:
            pt = self.atoms.calc.pt
            pt.pa[:] = (self._ph_c1 * pt.pa
                        + self._ph_c2 * self.rng.standard_normal(pt.nmodes))
        return forces
