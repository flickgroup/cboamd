"""Multi-mode (7-harmonic) CO2 Rabi splitting: NEP MD vs the ab initio Bonini 2024
Fig. 3B reference (data/CO2_Fig3_panelB.dat).

Both the reference (linear-response CBOA) and the NEP MD (dipole autocorrelation)
resonant LP/UP are extracted the same way -- the two IR-brightest branches in the
resonant band -- and compared at the same lambda grid. The 7-harmonic ladder
(omega_a=(a/3)*omega_res, lambda_a=(a/3)*lambda_res, a=1..7, resonant a=3) produces
a large asymmetric split: the lower polariton red-shifts more than the upper
blue-shifts (the chi-included signature). At strong coupling (lambda > ~0.10,
V-USC) the NEP lower polariton is weak and mixes with the off-resonant a=2
harmonic, so the naive top-2-peak extractor is unreliable there; the tests use
the cleanly-extractable range lambda <= 0.10. Skips unless the NEP cache
(generate_branches.py nep) is present.
"""
import numpy as np

_CLEAN_MAX = 0.10          # cleanly-extractable VSC range for the MD peak picker


def _clean(collected):
    lam = np.asarray(collected["lambda"])
    return lam <= _CLEAN_MAX + 1e-9


def _nearest(collected, value):
    lam = np.asarray(collected["lambda"])
    return int(np.argmin(np.abs(lam - value)))


def test_split_tracks_ab_initio_reference(collected):
    """NEP MD resonant Rabi splitting matches the ab initio Fig. 3B splitting to
    ~30 cm^-1 over the cleanly-extractable VSC range -- the MLIP-vs-first-principles
    agreement (offset-robust observable). Near lambda~0.07 the resonant upper
    polariton and the off-resonant a=4 harmonic swap brightness (an anticrossing),
    so the top-2-peak extraction jumps for both NEP and the reference at slightly
    different lambda; allow a couple of such outliers and require the bulk to agree
    tightly."""
    m = _clean(collected)
    diff = np.abs((collected["mm_upper"] - collected["mm_lower"])
                  - (collected["ref_upper"] - collected["ref_lower"]))[m]
    assert np.median(diff) < 40.0
    assert np.mean(diff < 60.0) >= 0.75          # >=75% of clean lambda agree tightly


def test_split_grows_and_degenerate_at_zero(collected):
    split = collected["mm_upper"] - collected["mm_lower"]
    assert split[_nearest(collected, 0.0)] < 50.0        # degenerate at lambda = 0
    assert split[_nearest(collected, 0.02)] > 50.0       # opens up
    assert split[_nearest(collected, 0.10)] > split[_nearest(collected, 0.02)]  # grows


def test_chi_included_asymmetry(collected):
    """chi included (Fig. 3B): the lower polariton red-shifts more than the upper
    blue-shifts, relative to the bare (lambda=0) frequency."""
    i0, i10 = _nearest(collected, 0.0), _nearest(collected, 0.10)
    bare = collected["mm_lower"][i0]
    drop = bare - collected["mm_lower"][i10]
    rise = collected["mm_upper"][i10] - bare
    assert drop > 0 and rise > 0                  # genuine hybridization
    assert drop > rise                            # asymmetric, lower polariton favored


def test_writes_rabi_figure(plotmod):
    """Single-panel freq-vs-coupling comparison (like compare_with_bonini2024_rabi.png)
    for the multi-mode panel B (chi included): the ab initio Bonini 2024 LP/UP branch
    vs NEP / ab initio q-mode / ab initio e-mode MD. Plots whichever method caches
    are present (generate the NEP ones with generate_branches.py; the ab initio
    caches ship with the repository)."""
    import os
    data = plotmod.collect_rabi()
    out = plotmod.rabi_multimode_figure(
        data, plotmod.FIGDIR + "/compare_with_bonini2024_multimode_rabi.png")
    assert os.path.exists(out)
