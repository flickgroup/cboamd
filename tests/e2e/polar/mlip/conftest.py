"""Fixtures for the polar MLIP comparison (NEP, DeepMD, DeepMD-const-chi vs ab initio).

NEP is recomputed once per session; DeepMD (real + constant-chi) and the ab initio
reference are read
from data/. All data-loading and plotting lives in plot.py. Heavy deps are gated so
the suite skips cleanly without them.
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
    return _load("polar_mlip_plot")


@pytest.fixture(scope="session")
def collected(plotmod):
    """NEP, DeepMD, DeepMD-const-chi and ab initio references; skip if any is missing."""
    data = plotmod.collect()
    for lam in plotmod.LAMBDAS:
        for method in ("nep", "deepmd", "deepmd_const", "aimd"):
            if data[lam][method] is None:
                if method == "aimd":
                    pytest.skip(
                        f"missing aimd reference for lambda={lam}; the ab "
                        "initio (DFT) dipole trajectories (run with "
                        "polar=true) are distributed through the data "
                        "repository published with the li2026cboamd paper, "
                        f"place aimd_lam{lam}.dat under data/ to activate "
                        "this comparison"
                    )
                pytest.skip(f"missing {method} reference for lambda={lam}; "
                            f"cache it into data/{method}_lam{lam}.dat first")
    return data
