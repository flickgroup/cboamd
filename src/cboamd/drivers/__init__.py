from cboamd.drivers.base import Driver
from cboamd.drivers.registry import (
    register_driver, get_driver_class, available_drivers)

__all__ = ["Driver", "register_driver", "get_driver_class", "available_drivers"]

# built-in drivers register on package import
from cboamd.drivers import deepmd, nep            # noqa: F401,E402

# pyscf is an optional extra: a missing install must not break importing the
# drivers package (the driver is then simply absent from available_drivers()).
try:
    from cboamd.drivers import pyscf_driver       # noqa: F401,E402
except ImportError:
    pass
