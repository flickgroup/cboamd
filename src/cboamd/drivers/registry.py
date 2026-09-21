"""Driver registry: maps driver names to Driver subclasses.

Built-in drivers register on package import; external packages expose extra
drivers through the 'cboamd.drivers' entry-point group (loading the entry
point imports the module, whose register_driver decorators run).
"""
from importlib.metadata import entry_points

_REGISTRY = {}
_ENTRY_POINTS_LOADED = False


def register_driver(cls):
    """Class decorator: register a Driver subclass under cls.name.

    A SUBCLASS of an already registered driver replaces it under the same name
    (that is how an extension package extends a built-in driver, e.g. adding
    q-mode to the public e-mode 'pyscf' driver); import order does not matter,
    the more specific class always wins. Two unrelated classes under one name
    remain an error.
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a non-empty 'name' attribute.")
    existing = _REGISTRY.get(cls.name)
    if existing is not None:
        if existing is cls or issubclass(existing, cls):
            return cls          # keep the more specific registration
        if not issubclass(cls, existing):
            raise ValueError(
                f"driver {cls.name!r} is already registered by {existing.__name__}.")
    _REGISTRY[cls.name] = cls
    return cls


def _load_entry_points():
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for ep in entry_points(group='cboamd.drivers'):
        ep.load()


def get_driver_class(name):
    # unconditionally (the flag makes it a no-op after the first call): an
    # installed extension may override a built-in name with a subclass, so the
    # entry points must be loaded even for names the built-ins already provide.
    _load_entry_points()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown driver {name!r}; available: {available_drivers()}. "
            "Extension drivers become available after installing the package "
            "that provides them through the 'cboamd.drivers' entry-point group.")
    return _REGISTRY[name]


def available_drivers():
    _load_entry_points()
    return sorted(_REGISTRY)
