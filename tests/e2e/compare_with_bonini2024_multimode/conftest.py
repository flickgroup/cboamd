"""Fixtures for the multi-mode (7-harmonic) CO2 Rabi comparison (Bonini 2024 Fig. 3B).

The MD branch caches are generated offline (generate_branches.py) and cached in
data/; the quantitative comparison + figure tests skip if the NEP cache is missing.
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
    pytest.importorskip("ase")
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    pytest.importorskip("calorine")
    mod = _load("bonini_mm_plot")
    if not os.path.exists(os.path.join(mod.MODELS, "nep-energy.txt")):
        pytest.skip("NEP CO2 models not available")
    return mod


@pytest.fixture(scope="session")
def collected(plotmod):
    """ab initio Fig. 3B reference + cached NEP branches; skip if no cache."""
    if plotmod.load_cached_branches() is None:
        pytest.skip("missing NEP branch cache; run 'python generate_branches.py nep'")
    return plotmod.collect()
