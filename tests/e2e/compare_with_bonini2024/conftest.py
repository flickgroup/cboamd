"""Fixtures for the Rabi-splitting comparison (NEP vs explicit CBOA QEDFT).

The NEP polariton branches are read from data/ (generate_nep.py); the suite skips
cleanly if they are missing or the plotting deps are unavailable. All loading and
plotting lives in plot.py.
"""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "plot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def plotmod():
    pytest.importorskip("ase")          # dt conversion (and NEP recompute fallback)
    pytest.importorskip("scipy")        # scripts.infrared FFT
    pytest.importorskip("matplotlib")
    return _load("bonini_rabi_plot")


@pytest.fixture(scope="session")
def collected(plotmod):
    """Reference plus cached NEP branches; skip if a NEP cache is missing."""
    for name in ("nep_chi_included.txt", "nep_chi_neglected.txt",
                 "nep_chi_resonant.txt"):
        if not os.path.exists(os.path.join(plotmod.DATA, name)):
            pytest.skip(f"missing {name}; run generate_nep.py")
    return plotmod.collect()
