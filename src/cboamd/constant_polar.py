"""A constant-polarizability stand-in for a DeepMD DeepPolar model.

The supplied dp-polar.pb returns an incomplete polarizability (chi_xx only). When
a sensible, geometry-independent polarizability is wanted instead -- e.g. the value
at the initial geometry -- this shim provides one with the same ``eval_full`` API
the deepmd glue calls, so it drops into the existing driver code path without
needing TensorFlow or a trained .pb. The polarizability is constant, hence its
gradient is exactly zero (no polarizability-force contribution).

The constant 3x3 tensor (atomic units, Bohr^3) is stored in a small .npz file
under key ``polarizability``; see scripts/make_constant_polar.py.
"""
import numpy as np


class ConstantPolar:
    """Mimics deepmd.infer.deep_polar.DeepPolar with a fixed polarizability.

    eval_full returns ``(value, gradient, atomic)`` in DeepMD's raw layout:
    value as (1, 9) and gradient as (1, 9, natom, 3) of zeros."""

    def __init__(self, source):
        if isinstance(source, str):
            polar = np.load(source)["polarizability"]
        else:
            polar = np.asarray(source, dtype=float)
        self.polarizability = np.asarray(polar, dtype=float).reshape(3, 3)

    @classmethod
    def from_file(cls, path):
        return cls(path)

    def save(self, path):
        np.savez(path, polarizability=self.polarizability)

    def eval_full(self, coords, cells, atom_types, atomic=False):
        natom = len(atom_types)
        value = self.polarizability.reshape(1, 9)
        gradient = np.zeros((1, 9, natom, 3))
        return value, gradient, None
