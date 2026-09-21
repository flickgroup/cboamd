"""Unit tests for the drivers' ``set_geometry`` geometry handoff.

These pin the per-step geometry update for the NEP and pyscf drivers without
needing real calorine models or a real pyscf Mole:

  * NEP (Task 1): ``set_geometry`` must push the new geometry onto the calculators
    built once in the driver constructor (no ``CPUNEP(...)`` reconstruction),
    and it must do so via ``set_atoms`` -- which clears calorine's cached
    ``results``/``nepy`` so properties recompute -- NOT a bare ``calc.atoms = ...``
    assignment. A bare assignment leaves the cache stale, freezing every NEP
    property at the initial geometry while the nuclei drift under a constant force.
  * pyscf (Task 5): ``set_geometry`` must reset the SCF with the new geometry using
    a single ``mol.copy()`` (the locally built Mole is already a private copy),
    not two.

Style mirrors test_nep_glue / test_deepmd_glue: driver instances built with
``__new__`` and small fakes; assert on the glue directly.
"""
import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("pyscf")

from pyscf.pbc.tools.pyscf_ase import atoms_from_ase  # noqa: E402

from cboamd.drivers.nep import NEPDriver  # noqa: E402
from cboamd.drivers.pyscf_driver import PySCFDriver  # noqa: E402


# ------------------------------------------------------------------ NEP (Task 1)
class _FakeNEPCalc:
    """Stand-in for a calorine CPUNEP. set_atoms updates the geometry and clears
    the cached results/nepy (as the real CPUNEP.set_atoms does), so a stale cache
    can't survive a geometry change. A plain ``calc.atoms = ...`` would bypass it
    and leave ``results``/``nepy`` stale -- the bug this guards against."""

    def __init__(self):
        self.atoms = None
        self.results = {"forces": "stale"}   # pretend a cached result exists
        self.nepy = "stale"

    def set_atoms(self, atoms):
        self.atoms = atoms
        self.results = {}
        self.nepy = None


def _nep_driver():
    drv = NEPDriver.__new__(NEPDriver)   # skip __init__: tests stub calc_pot etc.
    drv.calc_pot = _FakeNEPCalc()
    drv.calc_dip = _FakeNEPCalc()
    drv.calc_pol = _FakeNEPCalc()
    return drv


def test_set_geometry_nep_does_not_reconstruct_calculators():
    """The three model objects built in the constructor must survive set_geometry
    (same identities), and the new geometry must be pushed onto them."""
    calc = _nep_driver()
    pot0, dip0, pol0 = calc.calc_pot, calc.calc_dip, calc.calc_pol
    atoms = Atoms("CO2", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])

    calc.set_geometry(atoms)

    # identities unchanged -> no CPUNEP(...) reconstruction / disk reload
    assert calc.calc_pot is pot0
    assert calc.calc_dip is dip0
    assert calc.calc_pol is pol0
    # new geometry pushed onto every model AND the stale cache cleared (set_atoms,
    # not a bare .atoms assignment) so the next property eval recomputes
    for model in (calc.calc_pot, calc.calc_dip, calc.calc_pol):
        np.testing.assert_allclose(model.atoms.positions, atoms.positions)
        assert list(model.atoms.symbols) == list(atoms.symbols)
        assert model.results == {} and model.nepy is None


def test_set_geometry_nep_updates_geometry():
    """Two successive set_geometry calls -> the models see the second geometry, not
    a stale cached one."""
    calc = _nep_driver()
    atoms1 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    atoms2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, 1.20], [0, 0, -1.10]])

    calc.set_geometry(atoms1)
    calc.set_geometry(atoms2)

    for model in (calc.calc_pot, calc.calc_dip, calc.calc_pol):
        np.testing.assert_allclose(model.atoms.positions, atoms2.positions)


# ---------------------------------------------------------------- pyscf (Task 5)
class _FakeMol:
    """Counts copy()/build() across the copy chain via a shared counter dict."""

    def __init__(self, counter):
        self._counter = counter
        self.atom = None
        self.built = False

    def copy(self):
        self._counter["copies"] += 1
        return _FakeMol(self._counter)

    def build(self):
        self._counter["builds"] += 1
        self.built = True
        return self


class _FakeMF:
    def __init__(self, mol):
        self.mol = mol
        self.reset_mol = "<unset>"

    def reset(self, mol=None):
        self.reset_mol = mol


def test_set_geometry_pyscf_resets_with_new_geometry():
    """reset() must be called once with a built Mole carrying the new atoms, using
    a single copy of the base Mole (not two)."""
    counter = {"copies": 0, "builds": 0}
    mf = _FakeMF(_FakeMol(counter))
    drv = PySCFDriver.__new__(PySCFDriver)   # skip __init__: test stubs mf directly
    drv.mf = mf
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

    drv.set_geometry(atoms)

    # exactly one copy of the base Mole; no redundant second deep copy
    assert counter["copies"] == 1
    assert counter["builds"] == 1
    # reset got the built Mole carrying the new geometry
    assert isinstance(mf.reset_mol, _FakeMol)
    assert mf.reset_mol.built
    expected = atoms_from_ase(atoms)
    assert [sym for sym, _ in mf.reset_mol.atom] == [sym for sym, _ in expected]
    np.testing.assert_allclose([pos for _, pos in mf.reset_mol.atom],
                               [pos for _, pos in expected])
