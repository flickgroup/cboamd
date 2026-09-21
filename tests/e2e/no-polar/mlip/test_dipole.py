"""Compare the CO2 dipole-moment trajectory of NEP, DeepMD and an ab initio reference
(no polarizability) for lambda = 0, 0.1, 0.3 over 1000 steps.

Different models of the same surface, started from the same geometry at rest, so
the check is a sanity comparison: every method produces a non-trivial oscillating
dipole, and NEP / DeepMD track the ab initio reference at early times (before the
distinct potentials dephase). Writes the overlay plot.
"""
import numpy as np


def test_mlip_dipole(collected, plotmod):
    for lam in plotmod.LAMBDAS:
        d = collected[lam]
        for method in ("nep", "deepmd", "aimd"):
            arr = d[method]
            assert np.std(arr) > 1e-4, f"{method} dipole has no oscillation at lambda={lam}"
        # NEP/DeepMD approximate the ab initio surface -> agree at early times
        n = min(40, len(d["nep"]), len(d["aimd"]), len(d["deepmd"]))
        for method in ("nep", "deepmd"):
            corr = np.corrcoef(d[method][:n], d["aimd"][:n])[0, 1]
            assert corr > 0.9, f"{method} vs ab initio early dipole corr {corr:.2f} at lambda={lam}"

    plotmod.dipole_figure(collected, plotmod.FIGDIR + "/nopolar_mlip_dipole.png")
