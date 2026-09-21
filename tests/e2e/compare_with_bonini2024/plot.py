"""Rabi-splitting comparison: NEP (chi included / chi neglected) vs explicit CBOA QEDFT.

For each cavity coupling lambda on the reference grid, the NEP CO2 trajectory
is run with the molecular polarizability term on (chi included) and off
(chi neglected); the lower/upper polariton frequencies are read off the
dipole_x spectrum and overlaid on the explicit CBOA QEDFT reference
(octopus, data/octopus_CO2_CBOA_FD.txt).

With chi included the upper polariton is screened and stays near the bare mode
(matching QEDFT); with chi neglected it blows up with lambda -- the physics the
figure is meant to show.

Standalone:

    cd tests/e2e/compare_with_bonini2024
    PYTHONPATH=<repo>/src:<repo> python plot.py

NEP polariton peaks are cached in data/nep_chi_{included,neglected}.txt
(generate_nep.py). The e2e test imports this module so loading and plotting live
in one place. Writes ../figures/compare_with_bonini2024_rabi.png.
"""
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# NEP models are shared with the mlip comparisons
MODELS = os.path.join(HERE, "..", "no-polar", "mlip", "models")


def _repo_root(d):
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root (pyproject.toml) not found above " + HERE)


REPO = _repo_root(HERE)
FIGDIR = os.path.join(REPO, "tests", "e2e", "figures")
PARAMS = json.load(open(os.path.join(HERE, "params.json")))
GEOM = os.path.join(REPO, PARAMS["geometry"])
BAND = tuple(PARAMS["band"])


def load_reference():
    """Explicit CBOA QEDFT (octopus). Rows: lambda, bend, lower-pol, upper-pol.

    Returns (lambdas, lower_polariton, upper_polariton); the bend row is unused
    here (the dipole_x spectrum only carries the asymmetric-stretch polaritons)."""
    ref = np.loadtxt(os.path.join(HERE, PARAMS["reference"]))
    return ref[0], ref[2], ref[3]


LAMBDAS = load_reference()[0]


def dt_au():
    """Trajectory timestep in atomic units."""
    from ase import units
    from cboamd.atomic_constants import P_Ang, P_pe, P_Har
    return PARAMS["timestep_fs"] * units.fs * P_Ang * (P_pe * P_Har) ** 0.5


def run_nep(lam, polar, omega_au=None):
    """Run the NEP CO2 trajectory at coupling lambda with the polarizability term
    on/off; return dipole_x(t). `omega_au` overrides the bare cavity frequency
    (default PARAMS["omega_au"])."""
    from cboamd import pymd_ase
    nep = PARAMS["nep"]
    lam = float(lam)  # reference grid is a numpy array; keep the JSON config plain
    omega = PARAMS["omega_au"] if omega_au is None else float(omega_au)
    cfg = {
        "xyz_file": GEOM, "driver": "nep",
        "nep_pot": os.path.join(MODELS, nep["pot"]),
        "nep_dip": os.path.join(MODELS, nep["dip"]),
        "nep_pol": os.path.join(MODELS, nep["pol"]),
        "photons": lam > 0.0, "polar": polar, "polar_force": polar,
        "nphoton": 1, "omega_photon": [omega], "lambda_photon": [lam],
        "lambda_vector": [PARAMS["pol_vector"]], "steps": PARAMS["steps"],
        "timestep": PARAMS["timestep_fs"],
    }
    cwd = os.getcwd()
    d = tempfile.mkdtemp()
    os.chdir(d)
    try:
        json.dump(cfg, open("in.json", "w"))
        argv = sys.argv
        sys.argv = ["cboamd", "-i", "in.json"]
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            pymd_ase.main()
        sys.argv = argv
        dip = np.loadtxt("dipole.dat")
    finally:
        os.chdir(cwd)
    return dip[:, 2]


def spectrum(signal, dt):
    import scripts.infrared as infrared
    from cboamd.atomic_constants import P_cm1
    signal = np.asarray(signal)
    w, ft = infrared.fourier_transform(signal - signal.mean(), dt)
    return w * P_cm1, np.abs(ft)


def polariton_peaks(signal, dt, band=BAND):
    """Lower and upper polariton frequencies (cm^-1) from the dipole_x spectrum.

    Takes the two most intense local maxima in the band; at lambda=0 the branches
    are degenerate so a single peak is returned for both."""
    freq, amp = spectrum(signal, dt)
    m = (freq > band[0]) & (freq < band[1])
    f, a = freq[m], amp[m]
    loc = [k for k in range(1, len(a) - 1) if a[k] > a[k - 1] and a[k] > a[k + 1]]
    if not loc:
        loc = [int(np.argmax(a))]
    top = sorted(sorted(loc, key=lambda k: -a[k])[:2], key=lambda k: f[k])
    fr = [f[k] for k in top]
    return (fr[0], fr[0]) if len(fr) == 1 else (fr[0], fr[-1])


def compute_branches(polar):
    """Run NEP across the lambda grid and return (lower[], upper[]) polaritons."""
    dt = dt_au()
    lower, upper = [], []
    for lam in LAMBDAS:
        lo, up = polariton_peaks(run_nep(lam, polar), dt)
        lower.append(lo)
        upper.append(up)
    return np.array(lower), np.array(upper)


def cavity_chi_xx():
    """CO2 polarizability along the cavity polarization (x), from the NEP pol
    model at the input geometry, in a.u. -- the chi that screens the cavity mode
    (omega_eff^2 = omega_c^2 / (1 + lambda^2 * chi))."""
    from ase.io import read
    from calorine.calculators import CPUNEP
    atoms = read(GEOM, index=0)
    atoms.calc = CPUNEP(os.path.join(MODELS, PARAMS["nep"]["pol"]))
    pol = np.asarray(atoms.calc.get_polarizability())
    return float(pol[0, 0])


def compute_branches_resonant(chi=None):
    """Same chi-included (polar=True) dynamics, but the BARE cavity frequency is
    un-screened per lambda -- omega_bare = omega_c * sqrt(1 + lambda^2 * chi) --
    so that after the dynamics' own 1/(1+lambda^2 chi) screening the EFFECTIVE
    cavity frequency stays on the nominal resonance. Isolates the coupling-side
    effect of the polarizability from its detuning of the cavity mode.

    Returns (lower[], upper[], chi)."""
    if chi is None:
        chi = cavity_chi_xx()
    dt = dt_au()
    lower, upper = [], []
    for lam in LAMBDAS:
        omega_bare = PARAMS["omega_au"] * np.sqrt(1.0 + float(lam) ** 2 * chi)
        lo, up = polariton_peaks(run_nep(lam, True, omega_au=omega_bare), dt)
        lower.append(lo)
        upper.append(up)
    return np.array(lower), np.array(upper), chi


def load_cached(name):
    """Cached NEP branches as (lambdas, lower, upper) from data/<name>, or None."""
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    tab = np.loadtxt(path)
    return tab[0], tab[1], tab[2]


def collect():
    """Reference plus cached NEP branches (chi included / neglected / on-resonance).
    Recomputes a NEP variant only if its cache is missing."""
    lam, ref_lo, ref_up = load_reference()
    out = {"lambda": lam, "ref_lower": ref_lo, "ref_upper": ref_up}
    for name, key in (("nep_chi_included.txt", "incl"),
                      ("nep_chi_neglected.txt", "negl")):
        cached = load_cached(name)
        lo, up = (compute_branches(key == "incl") if cached is None
                  else (cached[1], cached[2]))
        out[key + "_lower"], out[key + "_upper"] = lo, up
    res = load_cached("nep_chi_resonant.txt")
    if res is None:
        lo, up, _ = compute_branches_resonant()
    else:
        _, lo, up = res
    out["res_lower"], out["res_upper"] = lo, up
    return out


def rabi_figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = data["lambda"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # (lower-key, upper-key, colour, label) -- order matches the reference figure
    series = [
        ("incl_lower", "incl_upper", "#2ca089", r"MLIP, $\chi$ included"),
        ("ref_lower", "ref_upper", "#c44e52", "explicit CBOA QEDFT"),
        ("negl_lower", "negl_upper", "#4c72b0", r"MLIP, $\chi$ neglected"),
        ("res_lower", "res_upper", "#dd8452",
         r"MLIP, $\chi$ incl., $\omega_c$ on resonance"),
    ]
    for lo_key, up_key, color, label in series:
        ax.plot(lam, data[lo_key], "o--", color=color, ms=5, lw=1.0, label=label)
        ax.plot(lam, data[up_key], "o--", color=color, ms=5, lw=1.0)  # upper branch, no label
    # bare cavity resonance (omega_c): the polaritons straddle it on resonance
    ax.axhline(PARAMS["omega_cm"], ls="--", color="0.4", lw=1.0, zorder=0,
               label=r"cavity resonance $\omega_c$")
    ax.set_xlabel(r"Coupling Strength $\lambda$")
    ax.set_ylabel(r"Frequency [cm$^{-1}$]")
    ax.set_xlim(-0.005, 0.305)
    ax.set_ylim(1000, 3500)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


SPEC_LAMBDAS = (0.0, 0.1, 0.2, 0.3)   # panels: outside-cavity + three couplings
SPEC_BAND = (1000.0, 3300.0)
CO2_CHI_XX = 21.8327   # a.u.; CO2 |-polarizability, == cavity_chi_xx() (NEP pol model)
# per lambda, the (tag, colour, label, polar) variants; omega handled below
SPEC_VARIANTS = [
    ("incl", "#2ca089", r"$\chi$ included", True),
    ("negl", "#4c72b0", r"$\chi$ neglected", False),
    ("res", "#dd8452", r"$\chi$ incl., $\omega_c$ on res.", True),
]
SPEC_CACHE = os.path.join(DATA, "nep_spectra.npz")


def compute_spectra(lambdas=SPEC_LAMBDAS, band=SPEC_BAND):
    """Run NEP once per (lambda, variant) and return (freq_band, {key: amp}); keys
    are f'{lam:.2f}_{tag}' (tag 'bare' at lambda=0, else incl/negl/res)."""
    dt = dt_au()
    chi = cavity_chi_xx()
    freq_band, out = None, {}
    for lam in lambdas:
        jobs = ([("bare", False, PARAMS["omega_au"])] if lam == 0.0 else
                [("incl", True, PARAMS["omega_au"]),
                 ("negl", False, PARAMS["omega_au"]),
                 ("res", True, PARAMS["omega_au"] * np.sqrt(1.0 + lam ** 2 * chi))])
        for tag, polar, omega in jobs:
            freq, amp = spectrum(run_nep(lam, polar, omega_au=omega), dt)
            m = (freq > band[0]) & (freq < band[1])
            freq_band = freq[m]
            out[f"{lam:.2f}_{tag}"] = amp[m]
    return freq_band, out


def save_spectra(path=SPEC_CACHE, lambdas=SPEC_LAMBDAS):
    """Cache the dipole spectra so the figure plots from disk (NEP runs once)."""
    freq, specs = compute_spectra(lambdas)
    np.savez(path, freq=freq, lambdas=np.array(lambdas, float), **specs)
    return path


def load_spectra(path=SPEC_CACHE):
    """Cached spectra as (freq, lambdas, {key: amp}), or None if absent."""
    if not os.path.exists(path):
        return None
    d = np.load(path)
    specs = {k: d[k] for k in d.files if k not in ("freq", "lambdas")}
    return d["freq"], d["lambdas"], specs


def spectrum_figure(path):
    """Dipole_x spectra (intensity vs wavenumber) for the NEP variants, stacked by
    coupling -- to see how each variant
    redistributes the oscillator strength between the polariton peaks. Reads the
    cached spectra (data/nep_spectra.npz); recomputes only if the cache is absent.

    Top panel (lambda=0) is outside-cavity (one bare spectrum); each coupling panel
    overlays chi included / neglected / (incl, omega_c on resonance). QEDFT LP/UP
    peaks are vertical dotted lines (QEDFT is peak positions, not a spectrum) and
    the cavity resonance omega_c a dashed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cached = load_spectra()
    if cached is None:
        save_spectra()
        cached = load_spectra()
    freq, lambdas, specs = cached

    fig, axes = plt.subplots(len(lambdas), 1, figsize=(6.4, 2.1 * len(lambdas)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, lam in zip(axes, lambdas):
        if lam == 0.0:
            ax.plot(freq, specs["0.00_bare"], color="k", lw=1.2, label="bare")
            ax.axvline(PARAMS["omega_cm"], ls="--", color="0.6", lw=0.8,
                       label=r"cavity $\omega_c$")
            ax.text(0.98, 0.92, rf"$\omega_c$ = {PARAMS['omega_cm']:.0f} cm$^{{-1}}$",
                    color="0.4", transform=ax.transAxes, ha="right", va="top",
                    fontsize=8)
        else:
            for tag, color, label, _polar in SPEC_VARIANTS:
                ax.plot(freq, specs[f"{lam:.2f}_{tag}"], color=color, lw=1.2,
                        label=(label if lam == lambdas[1] else None))
            ax.axvline(PARAMS["omega_cm"], ls="--", color="0.6", lw=0.8)
            # bare omega_c fed to the on-resonance variant (un-screened per lambda)
            omega_res = PARAMS["omega_cm"] * np.sqrt(1.0 + lam ** 2 * CO2_CHI_XX)
            ax.text(0.98, 0.92, rf"$\omega_c$(res, bare) = {omega_res:.0f} cm$^{{-1}}$",
                    color="#dd8452", transform=ax.transAxes, ha="right", va="top",
                    fontsize=8)
        ax.set_title("outside cavity" if lam == 0.0 else f"$\\lambda$ = {lam:.2f}",
                     fontsize=9)
        ax.set_ylabel("intensity")
        if lam in (0.0, lambdas[1]):
            ax.legend(fontsize=7, loc="upper left", frameon=False)
    axes[-1].set_xlabel(r"wavenumber (cm$^{-1}$)")
    fig.suptitle("CO$_2$ dipole spectra: NEP polariton intensities", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    data = collect()
    out = rabi_figure(data, os.path.join(FIGDIR, "compare_with_bonini2024_rabi.png"))
    print("wrote", out)
    out = spectrum_figure(os.path.join(FIGDIR, "compare_with_bonini2024_spectra.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
