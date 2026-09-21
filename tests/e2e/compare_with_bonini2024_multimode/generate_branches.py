"""Save MD dipole_x(t) trajectories (7-harmonic CO2) for the single-panel Rabi
comparison figure (test_multimode_rabi.py / rabi_multimode_figure).

Each method writes one dipole file per lambda, data/<tag>_lam<lambda>.dat (same
convention as ../no-polar/mlip). Spectra / peak-picking / LP-UP extraction are all
done at plot time from these files, so changing the extraction never requires
re-running the MD. A lambda is only (re)run if its file is missing.

    cd tests/e2e/compare_with_bonini2024_multimode
    PYTHONPATH=<repo>/src:<repo> python generate_branches.py nep          # minutes

Only the NEP branches are generated here (seconds/lambda). The ab initio q-mode
and e-mode dipole traces (data/aimd_{qmode,emode}_lam*.dat) ship pre-generated:
they need a self-consistent electronic-structure driver, which lives with the ab
initio extension package, and cost hours per lambda.
"""
import os
import sys

import numpy as np

import plot as P

SPEC = {"nep": ("nep", "e-mode")}

# NEP is cheap -> use a fine lambda grid for a dense curve.
NEP_LAMBDAS = np.round(np.arange(0.0, float(P.PARAMS["lambdas"][-1]) + 1e-9, 0.01), 3).tolist()


def _lambdas(tag):
    return NEP_LAMBDAS


def generate(tag):
    driver, cboa_mode = SPEC[tag]
    lambdas = _lambdas(tag)
    print(f"generating {tag} ({driver}, {cboa_mode}): "
          f"{len(lambdas)} lambdas x {P.PARAMS['steps']} steps...")
    for i, value in enumerate(lambdas, 1):
        out = P.dipole_path(tag, value)
        if os.path.exists(out):                   # only (re)run missing lambdas
            print(f"  lambda={value:.3f}: present, skipping ({i}/{len(lambdas)})")
            continue
        dip = P._run(driver, value, True, P.PARAMS["n_harmonics"],
                     P.PARAMS["alpha_res"], cboa_mode)
        P.save_dipole(tag, value, dip)            # raw dipole_x(t); extraction is
        print(f"  lambda={value:.3f}: saved {os.path.basename(out)} "  # post-processing
              f"({i}/{len(lambdas)})")


def main():
    args = sys.argv[1:] or ["all"]
    tags = list(SPEC) if args == ["all"] else args
    os.makedirs(P.DATA, exist_ok=True)
    for tag in tags:
        if tag not in SPEC:
            raise SystemExit(f"unknown tag {tag!r}; choose from {list(SPEC)} or 'all'")
        generate(tag)


if __name__ == "__main__":
    main()
