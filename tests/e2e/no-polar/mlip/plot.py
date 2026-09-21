"""Build the no-polar MLIP comparison figures (NEP and DeepMD vs an ab initio reference).

Standalone:

    cd tests/e2e/no-polar/mlip
    PYTHONPATH=<repo>/src:<repo> python plot.py

NEP is recomputed (calorine), DeepMD is read from data/ (cache it with
generate_deepmd.py in an env that has ase + deepmd-kit), and the ab initio
reference is read from data/ when a cached copy is present. Writes
../../figures/nopolar_mlip_dipole.png and nopolar_mlip_spectrum.png. The e2e tests
import this module so data-loading and plotting live in one place.
"""
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")


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
LAMBDAS = PARAMS["lambdas"]

# plot order / styling: (method-key, colour, linestyle, label)
SERIES = [
    ("aimd", "C1", "-", "ab initio (DFT)"),
    ("deepmd", "k", "--", "DeepMD"),
    ("nep", "C0", "-", "NEP"),
]
BAND = (1500.0, 3400.0)   # wide enough for the lambda=0.3 upper polariton


def run_nep(lam):
    """Run the NEP CO2 trajectory at coupling lambda; return (dipole_x, dt)."""
    from cboamd import pymd_ase
    nep = PARAMS["nep"]
    cfg = {
        "xyz_file": GEOM, "driver": "nep",
        "nep_pot": os.path.join(MODELS, nep["pot"]),
        "nep_dip": os.path.join(MODELS, nep["dip"]),
        "nep_pol": os.path.join(MODELS, nep["pol"]),
        "photons": lam > 0.0, "polar": False, "polar_force": False,
        "nphoton": 1, "omega_photon": [PARAMS["omega_au"]], "lambda_photon": [lam],
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
    return dip[:, 2], dip[1, 1] - dip[0, 1]


def load_ref(method, lam):
    """Cached dipole_x(t) for every method, read from data/."""
    path = os.path.join(DATA, f"{method}_lam{lam}.dat")
    return np.loadtxt(path) if os.path.exists(path) else None


def spectrum(signal, dt):
    import scripts.infrared as infrared
    from cboamd.atomic_constants import P_cm1
    signal = np.asarray(signal)
    w, ft = infrared.fourier_transform(signal - signal.mean(), dt)
    return w * P_cm1, np.abs(ft)


def dt_au():
    """Trajectory timestep in atomic units (same for every cached method)."""
    from ase import units
    from cboamd.atomic_constants import P_Ang, P_pe, P_Har
    return PARAMS["timestep_fs"] * units.fs * P_Ang * (P_pe * P_Har) ** 0.5


def collect():
    """Load NEP, DeepMD and ab initio references for every lambda (NEP is recomputed
    only if its cached file is missing). References that don't exist are None."""
    dt = dt_au()
    data = {}
    for lam in LAMBDAS:
        nep = load_ref("nep", lam)
        if nep is None:
            nep, _ = run_nep(lam)
        data[lam] = {"dt": dt, "nep": nep,
                     "deepmd": load_ref("deepmd", lam), "aimd": load_ref("aimd", lam)}
    return data


def find_peaks(freq, amp, n, band=BAND):
    m = (freq > band[0]) & (freq < band[1])
    f, a = freq[m], amp[m]
    loc = [k for k in range(1, len(a) - 1) if a[k] > a[k - 1] and a[k] > a[k + 1]]
    if not loc:
        loc = [int(np.argmax(a))]
    top = sorted(sorted(loc, key=lambda k: -a[k])[:n], key=lambda k: f[k])
    return [(f[k], a[k]) for k in top]


def _panels(title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(LAMBDAS), 1, figsize=(8, 9), sharex=True)
    fig.suptitle(title)
    return plt, fig, np.atleast_1d(axes)


def dipole_figure(data, path):
    plt, fig, axes = _panels("CO$_2$ dipole moment (no polarizability): NEP & DeepMD vs ab initio")
    for ax, lam in zip(axes, LAMBDAS):
        d = data[lam]
        t = np.arange(len(d["nep"])) * PARAMS["timestep_fs"]
        for key, color, ls, label in SERIES:
            arr = d[key]
            if arr is None:
                continue
            ax.plot(t[:len(arr)], arr, ls, color=color,
                    lw=1.1 if key == "deepmd" else 0.8, label=label)
        ax.set_ylabel("dipole$_x$ (a.u.)")
        ax.set_title(f"$\\lambda$ = {lam}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (fs)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def spectrum_figure(data, path):
    plt, fig, axes = _panels("CO$_2$ IR / polariton spectrum (no polarizability): NEP & DeepMD vs ab initio")
    for ax, lam in zip(axes, LAMBDAS):
        d = data[lam]
        for key, color, ls, label in SERIES:
            arr = d[key]
            if arr is None:
                continue
            freq, amp = spectrum(arr, d["dt"])
            m = (freq > BAND[0]) & (freq < BAND[1])
            ax.plot(freq[m], amp[m], ls, color=color, lw=1.3, label=label)
            for pf, pa in find_peaks(freq, amp, n=2 if lam > 0 else 1):
                ax.annotate(f"{pf:.0f}", (pf, pa), color=color, fontsize=7,
                            ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
        ax.set_ylabel("IR intensity (arb.)")
        ax.set_title(f"$\\lambda$ = {lam}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    data = collect()
    dipole_figure(data, os.path.join(FIGDIR, "nopolar_mlip_dipole.png"))
    spectrum_figure(data, os.path.join(FIGDIR, "nopolar_mlip_spectrum.png"))
    print("wrote nopolar_mlip_{dipole,spectrum}.png to", os.path.abspath(FIGDIR))


if __name__ == "__main__":
    main()
