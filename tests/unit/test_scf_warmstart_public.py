"""Fast public unit tests for the pyscf SCF warm-start path.

`PySCFDriver._solve_scf` (src/cboamd/drivers/pyscf_driver.py) seeds pyscf's
kernel from the previous step's converged density (`self._dm_prev`) whenever
`p.scf_warmstart` is true, and falls back to pyscf's default initial guess
when the cached density's shape does not match the current mol (geometry or
basis change), or when `p.scf_warmstart` is false. This pins that contract on
a tiny CO2/sto3g molecule, so the whole file runs in a couple of seconds.

Driver construction mirrors test_emode_projection.py: a real `pc.parameters()`
filled in through `PySCFDriver.parse_input` (the same path a real input file
takes), fed to the real `PySCFDriver.__init__` with a minimal calc stub.
"""
import types

import numpy as np
import pytest

pytest.importorskip("pyscf")

from cboamd import pycalculator as pc  # noqa: E402
from cboamd.drivers.pyscf_driver import PySCFDriver  # noqa: E402

_COORD = 'C 0 0 0; O 0 0 1.16; O 0 0 -1.16'  # CO2, sto3g -> nao == 15


def _driver(scf_warmstart):
    """Real PySCFDriver.__init__ path, sto3g for speed, via parse_input (as a
    real input file would set p.basis / p.scf_warmstart)."""
    p = pc.parameters()
    settings = {}
    PySCFDriver.parse_input(
        {'basis': 'sto3g', 'scf_warmstart': scf_warmstart, 'conv_tol': 1e-9},
        p, settings)
    calc = types.SimpleNamespace(p=p, pt=None)
    return PySCFDriver(calc, _COORD, None)


def test_first_call_captures_dm_prev_with_correct_shape():
    drv = _driver(scf_warmstart=True)
    drv._solve_scf()
    assert drv.mf.converged
    assert drv._dm_prev.shape[-1] == drv.mf.mol.nao


def test_wrong_shape_dm_prev_falls_back_silently_and_refreshes():
    drv = _driver(scf_warmstart=True)
    nao = drv.mf.mol.nao
    assert nao != 2                       # sanity: (2, 2) below really is wrong
    drv._dm_prev = np.zeros((2, 2))       # wrong shape (stale geometry/basis)
    drv._solve_scf()                      # must not raise; silently falls back
    assert drv.mf.converged
    assert drv._dm_prev.shape[-1] == nao  # refreshed to the real shape after


def test_scf_warmstart_false_never_sets_dm_prev():
    drv = _driver(scf_warmstart=False)
    drv._solve_scf()
    assert drv.mf.converged
    assert getattr(drv, '_dm_prev', None) is None


def test_scf_warmstart_false_never_seeds_kernel():
    # Pin the seeding half of the contract too: even with a stale _dm_prev
    # present, scf_warmstart=False must call kernel() without a dm0 seed and
    # must leave the stale cache untouched (identity unchanged).
    drv = _driver(scf_warmstart=False)
    stale = np.zeros((drv.mf.mol.nao, drv.mf.mol.nao))
    drv._dm_prev = stale
    seen = {}
    real_kernel = drv.mf.kernel

    def spy_kernel(*args, **kwargs):
        seen['dm0'] = kwargs.get('dm0', args[0] if args else None)
        return real_kernel(*args, **kwargs)

    drv.mf.kernel = spy_kernel
    drv._solve_scf()
    assert seen['dm0'] is None
    assert drv._dm_prev is stale
