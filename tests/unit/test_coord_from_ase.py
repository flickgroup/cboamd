import numpy as np
from ase import Atoms
from cboamd.pymd_ase import coord_from_ase


def test_coord_from_ase_matches_driver_atom_format():
    atoms = Atoms('CO2', positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
    coord = coord_from_ase(atoms)
    assert [c[0] for c in coord] == ['C', 'O', 'O']
    # drivers read the position as atom[1:][0]
    assert np.allclose(coord[1][1:][0], [1.16, 0.0, 0.0])
