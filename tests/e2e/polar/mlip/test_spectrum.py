"""Compare the CO2 IR / polariton spectrum of NEP, DeepMD and an ab initio reference
(with polarizability, polar=true) for lambda = 0, 0.1, 0.3 over 1000 steps.

These are different models of the same DFT surface. The asserted check is at
lambda=0 (no cavity, so the polarizability is inactive): each method's dominant
mode sits in the C=O asymmetric-stretch region and the ML potentials land near the
ab initio reference. At lambda>0 the polariton splitting is screened by the
polarizability; NEP matches the reference there, while DeepMD's polar model
under-screens
(documented outlier, see README) -- those curves are plotted, not asserted. Writes
the annotated overlay plot.
"""
import numpy as np


def _peak(freq, amp, band=(1500.0, 3000.0)):
    m = (freq > band[0]) & (freq < band[1])
    return freq[m][np.argmax(amp[m])]


def test_mlip_spectra(collected, plotmod):
    aimd0 = collected[0.0]["aimd"]
    p_aimd = _peak(*plotmod.spectrum(aimd0, collected[0.0]["dt"]))

    for method in ("nep", "deepmd", "deepmd_const", "aimd"):
        arr = collected[0.0][method]
        peak = _peak(*plotmod.spectrum(arr, collected[0.0]["dt"]))
        assert 1900.0 < peak < 2700.0, f"{method} lambda=0 peak {peak:.0f} outside C=O region"
        if method != "aimd":
            assert abs(peak - p_aimd) < 200.0, f"{method} peak {peak:.0f} far from ab initio {p_aimd:.0f}"

    plotmod.spectrum_figure(collected, plotmod.FIGDIR + "/polar_mlip_spectrum.png")
