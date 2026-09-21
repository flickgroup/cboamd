"""Rabi-splitting comparison of NEP against the explicit CBOA QEDFT reference.

Checks the physics of the published comparison figure: including the molecular
polarizability (chi) makes NEP track the QEDFT polaritons, while neglecting it
lets the upper polariton blow up with the coupling.
"""
import numpy as np


def test_chi_included_tracks_qedft(collected, plotmod):
    """With chi included, NEP reproduces the QEDFT lower/upper polaritons."""
    assert np.max(np.abs(collected["incl_lower"] - collected["ref_lower"])) < 150.0
    assert np.max(np.abs(collected["incl_upper"] - collected["ref_upper"])) < 150.0


def test_chi_neglected_blows_up(collected, plotmod):
    """Neglecting chi pushes the upper polariton well above QEDFT at strong
    coupling, and above the chi-included branch."""
    assert collected["negl_upper"][-1] > collected["ref_upper"][-1] + 250.0
    assert collected["negl_upper"][-1] > collected["incl_upper"][-1] + 200.0


def test_lower_polariton_redshifts(collected, plotmod):
    """Both NEP variants red-shift the lower polariton as the coupling grows."""
    for key in ("incl_lower", "negl_lower"):
        assert collected[key][-1] < collected[key][0] - 200.0


def test_resonant_omega_stays_on_resonance(collected, plotmod):
    """chi in the dynamics, but the bare cavity frequency un-screened per lambda
    (omega_bare = omega_c*sqrt(1+lambda^2 chi)) so the dynamics' own
    1/(1+lambda^2 chi) screening leaves the EFFECTIVE frequency on resonance: the
    polariton pair then stays centred on the bare mode, whereas the un-adjusted
    chi-included case is detuned by the screening and slides the pair down."""
    res_c = 0.5 * (collected["res_lower"] + collected["res_upper"])
    incl_c = 0.5 * (collected["incl_lower"] + collected["incl_upper"])
    # on-resonance: the centre barely moves across the coupling range
    assert np.max(np.abs(res_c - res_c[0])) < 40.0
    # contrast: the un-adjusted chi-included centre slides down markedly, so the
    # adjusted variant ends far above it
    assert incl_c[0] - incl_c[-1] > 300.0
    assert res_c[-1] > incl_c[-1] + 300.0


def test_resonant_is_growing_anticrossing(collected, plotmod):
    """The on-resonance variant is a well-formed anti-crossing: the Rabi splitting
    grows with the coupling from ~0 to several hundred cm^-1."""
    split = collected["res_upper"] - collected["res_lower"]
    assert split[-1] > 400.0
    assert split[-1] > split[1]


def test_writes_figure(collected, plotmod):
    plotmod.rabi_figure(collected, plotmod.FIGDIR + "/compare_with_bonini2024_rabi.png")


def test_writes_spectrum_figure(plotmod):
    """Stacked dipole-spectra figure (intensities) from the cached spectra."""
    import os
    import pytest
    if not os.path.exists(os.path.join(plotmod.DATA, "nep_spectra.npz")):
        pytest.skip("missing nep_spectra.npz; run generate_nep.py")
    plotmod.spectrum_figure(plotmod.FIGDIR + "/compare_with_bonini2024_spectra.png")
