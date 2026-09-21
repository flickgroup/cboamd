import numpy as np
import pytest

from cboamd.drivers.base import Driver
from cboamd.drivers.registry import (
    register_driver, get_driver_class, available_drivers, _REGISTRY)


def _make_dummy(name_str):
    @register_driver
    class Dummy(Driver):
        name = name_str

        def set_geometry(self, atoms):
            pass

        def energy_and_dipole(self):
            return 0.0, np.zeros(3)

        def energy_gradient(self):
            return np.zeros((1, 3))
    return Dummy


def test_register_and_lookup():
    cls = _make_dummy('dummy_a')
    assert get_driver_class('dummy_a') is cls
    assert 'dummy_a' in available_drivers()


def test_duplicate_name_rejected():
    _make_dummy('dummy_b')
    with pytest.raises(ValueError, match="already registered"):
        _make_dummy('dummy_b')


def test_subclass_replaces_base_under_same_name():
    """An extension may ship a SUBCLASS of a built-in driver under the same name;
    the more specific class wins (that is how an extension adds q-mode to the
    public e-mode 'pyscf' driver)."""
    base = _make_dummy('dummy_sub')

    @register_driver
    class Extended(base):
        supported_cboa_modes = ('e-mode', 'q-mode')

    assert get_driver_class('dummy_sub') is Extended


def test_base_registering_after_subclass_does_not_downgrade():
    """Import order must not matter: once the subclass is registered, a later
    registration of the base keeps the subclass."""
    base = _make_dummy('dummy_order')

    @register_driver
    class Extended(base):
        pass

    register_driver(base)
    assert get_driver_class('dummy_order') is Extended


def test_unrelated_duplicate_still_rejected():
    """Two unrelated classes under one name is still an error (only the
    subclass relationship is allowed to override)."""
    _make_dummy('dummy_unrelated')
    with pytest.raises(ValueError, match="already registered"):
        _make_dummy('dummy_unrelated')


def test_unknown_driver_names_extension_package():
    with pytest.raises(ValueError, match="entry-point group"):
        get_driver_class('no_such_driver')


def test_default_capabilities():
    cls = _make_dummy('dummy_c')
    drv = cls(calc=type('C', (), {'p': None, 'pt': None})(), coord=None, cell=None)
    assert cls.supported_cboa_modes == ('e-mode',)
    assert drv.polarizability().shape == (3, 3)
    assert drv.polarizability_gradient(natom=2) is None
    assert drv.refresh_displaced() is None
