"""Multi-mode (7-harmonic) CO2 Rabi comparison vs Bonini 2024 Fig. 3B.

Single-panel figure (like ../compare_with_bonini2024/compare_with_bonini2024_rabi.png):
frequency vs coupling for the harmonic ladder omega_a=(a/3)*omega_res,
lambda_a=(a/3)*lambda_res, a=1..7, resonant a=3, chi included (panel B). The ab
initio panel-B vibro-polariton branches are the reference; NEP / ab initio q-mode /
ab initio e-mode MD resonant LP/UP are overlaid. The ab initio dipole traces ship
pre-generated in data/ (their generator lives with the ab initio extension
package); NEP is regenerated here.

Standalone (generate the NEP branch caches, then the figure is made by the test):
    cd tests/e2e/compare_with_bonini2024_multimode
    PYTHONPATH=<repo>/src:<repo> python generate_branches.py nep
"""
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _repo_root(d):
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root (pyproject.toml) not found above " + HERE)


REPO = _repo_root(HERE)
FIGDIR = os.path.join(REPO, "tests", "e2e", "figures")
MODELS = os.path.join(REPO, "tests", "e2e", "no-polar", "mlip", "models")
PARAMS = json.load(open(os.path.join(HERE, "params.json")))
GEOM = os.path.join(REPO, PARAMS["geometry"])
MEV2CM = 8.0655442          # cm^-1 per meV


def dt_au():
    from ase import units
    from cboamd.atomic_constants import P_Ang, P_pe, P_Har
    return PARAMS["timestep_fs"] * units.fs * P_Ang * (P_pe * P_Har) ** 0.5


def harmonic_ladder(omega_res, lambda_res, n_harmonics, alpha_res):
    """omega_a = (a/alpha_res)*omega_res, lambda_a = (a/alpha_res)*lambda_res,
    a = 1..n_harmonics (Bonini 2024 Fig. 3 multi-mode setup)."""
    a = np.arange(1, n_harmonics + 1)
    return (a / alpha_res) * omega_res, (a / alpha_res) * lambda_res


def _run(driver, lambda_res, polar, n_harmonics, alpha_res, cboa_mode="e-mode"):
    """Run one CO2 MD trajectory; return dipole_x(t).

    cboa_mode='e-mode' (any driver) or 'q-mode' (self-consistent SCF; only drivers
    with a self-consistent electronic structure support it, and polar is then
    carried by the SCF density so the polar flag has no effect)."""
    omegas, lambdas = harmonic_ladder(
        PARAMS["omega_res_au"], float(lambda_res), n_harmonics, alpha_res)
    cfg = {
        "xyz_file": GEOM, "driver": driver, "cboa_mode": cboa_mode,
        "photons": lambda_res > 0.0, "polar": polar, "polar_force": polar,
        "nphoton": int(n_harmonics),
        "omega_photon": omegas.tolist(), "lambda_photon": lambdas.tolist(),
        "lambda_vector": [PARAMS["pol_vector"]] * int(n_harmonics),
        "steps": PARAMS["steps"], "timestep": PARAMS["timestep_fs"],
    }
    if driver == "nep":
        nep = PARAMS["nep"]
        cfg.update(nep_pot=os.path.join(MODELS, nep["pot"]),
                   nep_dip=os.path.join(MODELS, nep["dip"]),
                   nep_pol=os.path.join(MODELS, nep["pol"]))
    cwd, d = os.getcwd(), tempfile.mkdtemp()
    os.chdir(d)
    try:
        json.dump(cfg, open("in.json", "w"))
        argv = sys.argv
        sys.argv = ["cboamd", "-i", "in.json"]
        from cboamd import pymd_ase
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            pymd_ase.main()
        sys.argv = argv
        dip = np.loadtxt("dipole.dat")[:, 2]            # dipole_x
    finally:
        os.chdir(cwd)
    return dip


def spectrum(signal, dt):
    import scripts.infrared as infrared
    from cboamd.atomic_constants import P_cm1
    signal = np.asarray(signal)
    w, ft = infrared.fourier_transform(signal - signal.mean(), dt)
    return w * P_cm1, np.abs(ft)


def polariton_peaks(signal, dt, band):
    """Lower/upper polariton frequencies (cm^-1) from the dipole_x spectrum."""
    freq, amp = spectrum(signal, dt)
    m = (freq > band[0]) & (freq < band[1])
    f, a = freq[m], amp[m]
    loc = [k for k in range(1, len(a) - 1) if a[k] > a[k - 1] and a[k] > a[k + 1]]
    if not loc:
        loc = [int(np.argmax(a))]
    top = sorted(sorted(loc, key=lambda k: -a[k])[:2], key=lambda k: f[k])
    fr = [f[k] for k in top]
    return (fr[0], fr[0]) if len(fr) == 1 else (fr[0], fr[-1])


# --- Bonini 2024 Fig. 3 reference data (ab initio linear response) -----------
# Provided per panel as CO2_Fig3_panel{A,B,C,D}.dat, shape (Nlambda, 1 + 3*nb):
#   col0 = lambda ; cols 1..nb = frequency (meV) ; cols nb+1..2nb = photon
#   character ; cols 2nb+1..3nb = IR intensity. nb = 16 (multi A/B), 10 (single C/D).

def load_panel(panel):
    """Load a Fig. 3 reference panel: (lambda (N,), freqs_cm (N, nb),
    intensity (N, nb), character (N, nb)). Frequencies converted meV -> cm^-1."""
    arr = np.loadtxt(os.path.join(DATA, f"CO2_Fig3_panel{panel}.dat"))
    lam = arr[:, 0]
    nb = (arr.shape[1] - 1) // 3
    freqs = arr[:, 1:1 + nb] * MEV2CM
    character = arr[:, 1 + nb:1 + 2 * nb]
    intensity = arr[:, 1 + 2 * nb:1 + 3 * nb]
    return lam, freqs, intensity, character


def reference_branches(panel, band_cm, lambdas):
    """Resonant lower/upper polariton (cm^-1) vs lambda from a reference panel:
    at each lambda take the two IR-brightest vibro-polariton branches inside the
    band (the LP/UP doublet). Mirrors polariton_peaks on the ab initio data."""
    lam, fr, inten, _ = load_panel(panel)
    lower, upper = [], []
    for target in lambdas:
        j = int(np.argmin(np.abs(lam - target)))
        f, w = fr[j], inten[j]
        m = (f > band_cm[0]) & (f < band_cm[1])
        fb, wb = f[m], w[m]
        if fb.size < 2:
            lower.append(np.nan); upper.append(np.nan); continue
        pair = np.sort(fb[np.argsort(wb)[-2:]])
        lower.append(pair[0]); upper.append(pair[-1])
    return np.array(lower), np.array(upper)


# --- MD dipole trajectories, per (method, lambda) ----------------------------
# We save the raw dipole_x(t) per (method, lambda) as data/<tag>_lam<lambda>.dat
# (same convention as ../no-polar/mlip), so the spectrum, peak-picking and LP-UP
# extraction are all post-processing that NEVER requires re-running the MD. The
# dipole_x spectrum has several IR-active branches per lambda (resonant LP/UP, the
# mixed mid-branch, the off-resonant a=2/a=4 harmonics); the figure plots them all.
_METHOD_STYLE = {   # tag -> (color, marker, label)
    "nep": ("#2ca089", "o", "NEP (e-mode)"),
    "aimd_qmode": ("#c44e52", "s", "ab initio q-mode"),
    "aimd_emode": ("#4c72b0", "^", "ab initio e-mode"),
}


def dipole_path(tag, lam):
    return os.path.join(DATA, f"{tag}_lam{lam}.dat")


def save_dipole(tag, lam, dip):
    np.savetxt(dipole_path(tag, lam), dip,
               header=f"{tag} dipole_x(t) [a.u.]; lambda={lam}, "
                      f"{PARAMS['steps']} steps @ {PARAMS['timestep_fs']} fs, "
                      "7-harmonic resonant, chi included")


def method_lambdas(tag):
    """Lambda values with a saved dipole file for this method (sorted)."""
    import glob
    import re
    lams = []
    for path in glob.glob(os.path.join(DATA, f"{tag}_lam*.dat")):
        m = re.search(r"_lam(-?[0-9.]+)\.dat$", os.path.basename(path))
        if m:
            lams.append(float(m.group(1)))
    return sorted(lams)


def find_peaks(freq, amp, band, kmax=8, rel=0.02):
    """Significant IR peaks (cm^-1) in the band of a spectrum: local maxima with
    amplitude >= rel*max, up to kmax by amplitude, sorted by frequency. Pure
    post-processing -- peak-picking can change without re-running any MD."""
    freq, amp = np.asarray(freq), np.asarray(amp)
    m = (freq > band[0]) & (freq < band[1])
    f, a = freq[m], amp[m]
    if f.size == 0:
        return np.array([]), np.array([])
    loc = [i for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] > a[i + 1]]
    if not loc:
        loc = [int(np.argmax(a))]
    amax = max(a[i] for i in loc)
    loc = [i for i in loc if a[i] >= rel * amax]
    loc = sorted(sorted(loc, key=lambda i: -a[i])[:kmax], key=lambda i: f[i])
    return np.array([f[i] for i in loc]), np.array([a[i] for i in loc])


def load_cached_branches():
    return method_lambdas("nep") or None


def method_peaks(tag, band, kmax=8, rel=0.02):
    """All spectral peaks per lambda from the saved dipole files -> (lam, freq, amp)."""
    lams = method_lambdas(tag)
    if not lams:
        return None
    dt = dt_au()
    plam, pfreq, pamp = [], [], []
    for lam in lams:
        f, a = spectrum(np.loadtxt(dipole_path(tag, lam)), dt)
        fr, am = find_peaks(f, a, band, kmax=kmax, rel=rel)
        plam.extend([lam] * len(fr)); pfreq.extend(fr); pamp.extend(am)
    return np.array(plam), np.array(pfreq), np.array(pamp)


def collect():
    """NEP LP/UP (two brightest peaks in the resonant band, from the saved NEP
    dipole files) + ab initio reference (panel B) at the same lambdas."""
    lams = method_lambdas("nep")
    dt = dt_au()
    lam_ok, lo, up = [], [], []
    for lam in lams:
        f, a = spectrum(np.loadtxt(dipole_path("nep", lam)), dt)
        fr, _ = find_peaks(f, a, PARAMS["band_cm"], kmax=2)
        lam_ok.append(lam)
        if fr.size == 0:
            lo.append(np.nan); up.append(np.nan)
        else:
            lo.append(fr.min()); up.append(fr.max())
    lam_ok = np.array(lam_ok)
    ref_lo, ref_up = reference_branches("B", PARAMS["band_cm"], lam_ok)
    return {"lambda": lam_ok, "mm_lower": np.array(lo), "mm_upper": np.array(up),
            "ref_lower": ref_lo, "ref_upper": ref_up}


# --- single-panel comparison figure (panel B: multi, chi included) -----------

def collect_rabi():
    """ab initio panel B (all branches) + all MD spectral peaks per available method
    (from the saved dipole files -- no MD rerun needed to change extraction)."""
    lam, freqs, intensity, character = load_panel("B")
    out = {"ref_panel": (lam, freqs, intensity, character), "methods": {}}
    for tag in _METHOD_STYLE:
        peaks = method_peaks(tag, PARAMS["band_peaks_cm"])   # default kmax=8, rel=0.02
        if peaks is not None:
            out["methods"][tag] = peaks
    return out


def rabi_multimode_figure(data, path):
    """Single-panel freq-vs-coupling comparison for the multi-mode (7-harmonic,
    resonant) CO2, chi included (Bonini 2024 panel B): ab initio vibro-polariton
    branches as uniform-size dots coloured by photon character, with ALL MD spectral
    peaks per method overlaid (uniform-size coloured markers, no character)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    lam, freqs, _intensity, character = data["ref_panel"]
    ywin = (1150.0, 3400.0)          # resonant polaritons + bend and a=2/a=4 harmonics
    L = np.broadcast_to(lam[:, None], freqs.shape)
    sc = ax.scatter(L.ravel(), freqs.ravel(), c=character.ravel(), cmap="plasma",
                    vmin=0, vmax=1, s=6, linewidths=0, zorder=1,
                    label="VASP Bonini2024")
    for tag, (color, mk, label) in _METHOD_STYLE.items():
        if tag not in data["methods"]:
            continue
        lm, fr, _am = data["methods"][tag]          # every MD spectral peak
        ax.scatter(lm, fr, marker=mk, s=28, facecolor=color, edgecolor="k",
                   linewidths=0.5, label=label, zorder=3)
    ax.set_xlabel(r"coupling $\lambda$")
    ax.set_ylabel(r"$\hbar\omega$ [cm$^{-1}$]")
    ax.set_title(r"CO$_2$ multi-mode (7 harmonics, resonant), $\chi$ included")
    ax.set_xlim(-0.005, float(lam.max()) + 0.005)
    ax.set_ylim(*ywin)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.colorbar(sc, ax=ax, label="photon character (ab initio reference)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
