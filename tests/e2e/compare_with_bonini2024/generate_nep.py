"""Cache the NEP polariton branches for the Rabi-splitting comparison.

Run in an environment with ase + calorine:

    cd tests/e2e/compare_with_bonini2024
    PYTHONPATH=<repo>/src:<repo> python generate_nep.py

For every lambda on the reference grid it runs the NEP CO2 trajectory
(PARAMS["steps"] steps) twice -- with the polarizability term included and
neglected -- extracts the lower/upper polariton frequencies from the dipole_x
spectrum, and writes data/nep_chi_included.txt and data/nep_chi_neglected.txt
(each a 3xN table: lambda, lower, upper). A variant is recomputed only if its file
is missing.
"""
import os

import numpy as np

import plot as P  # shared run + extraction logic


def _save(polar, name):
    out = os.path.join(P.DATA, name)
    if os.path.exists(out):
        print("present, skipping:", name)
        return
    tag = "included" if polar else "neglected"
    print(f"generating NEP (chi {tag}), {len(P.LAMBDAS)} lambdas x {P.PARAMS['steps']} steps...")
    lower, upper = P.compute_branches(polar)
    np.savetxt(out, np.vstack([P.LAMBDAS, lower, upper]),
               header=f"NEP polariton branches, chi {tag} [cm^-1]; rows: lambda, lower, upper; "
                      f"omega={P.PARAMS['omega_cm']} cm-1, {P.PARAMS['steps']} steps @ "
                      f"{P.PARAMS['timestep_fs']} fs")
    print("saved", name)


def _save_resonant():
    out = os.path.join(P.DATA, "nep_chi_resonant.txt")
    if os.path.exists(out):
        print("present, skipping: nep_chi_resonant.txt")
        return
    print(f"generating NEP (chi included, omega_c on resonance), "
          f"{len(P.LAMBDAS)} lambdas x {P.PARAMS['steps']} steps...")
    lower, upper, chi = P.compute_branches_resonant()
    np.savetxt(out, np.vstack([P.LAMBDAS, lower, upper]),
               header="NEP polariton branches, chi INCLUDED with the bare cavity "
                      "frequency un-screened per lambda "
                      f"(omega_bare = {P.PARAMS['omega_cm']}*sqrt(1+lambda^2*chi), "
                      f"chi_xx = {chi:.4f} a.u.) so the dynamics' 1/(1+lambda^2 chi) "
                      "screening leaves the effective frequency on resonance [cm^-1]; "
                      "rows: lambda, lower, upper")
    print("saved nep_chi_resonant.txt")


def _save_spectra_cache():
    out = os.path.join(P.DATA, "nep_spectra.npz")
    if os.path.exists(out):
        print("present, skipping: nep_spectra.npz")
        return
    print(f"generating NEP dipole spectra for lambdas {P.SPEC_LAMBDAS} "
          f"x variants, {P.PARAMS['steps']} steps...")
    P.save_spectra(out)
    print("saved nep_spectra.npz")


def main():
    os.makedirs(P.DATA, exist_ok=True)
    _save(True, "nep_chi_included.txt")
    _save(False, "nep_chi_neglected.txt")
    _save_resonant()
    _save_spectra_cache()


if __name__ == "__main__":
    main()
