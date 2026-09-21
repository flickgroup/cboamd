import warnings

import numpy as np


class photon_mode:
    """One or more cavity photon modes (array-based; mirrors the reference implementation).

    Storage conventions match the reference implementation:
      omega, lam : np.ndarray shape (nmodes,)
      pol_vec    : np.ndarray shape (3, nmodes) -- column m is mode m's polarization
    Added here for MD (not in the reference): qa, pa, fa, ea, shape (nmodes,).

    The reference convention is followed, but the normalization is made
    *idempotent*: a scalar, a (1,) array, or a (nmodes,) array all map to
    (nmodes,), and ``pol_vec`` accepts 1-D (3,) [single mode], (3, nmodes), or
    (nmodes, 3) and always stores (3, nmodes). This is required because the q-mode
    finite-difference helpers reconstruct a photon_mode from an *existing* one's
    arrays, e.g. ``photon_mode(pt.nmodes, pt.omega, pt.lam, pt.pol_vec, ...)``; the
    reference's literal ``np.array([omega])`` would nest those to (1, 1) and crash.
    Scalar ``qa``/``pa`` broadcast across modes.
    """

    def __init__(self, nmodes, omega, lam, pol_vec, exc=0., number=0., qa=0, pa=0):
        self.nmodes = int(nmodes)
        # omega/lam are per-mode physical parameters -> must supply exactly nmodes
        # entries (no silent broadcast). qa/pa are initial conditions -> a scalar
        # broadcasts to all modes.
        self.omega = self._as_mode_array(omega, "omega", broadcast=False)
        self.lam = self._as_mode_array(lam, "lam", broadcast=False)
        self.pol_vec = self._as_pol_array(pol_vec)
        self.qa = self._as_mode_array(qa, "qa", broadcast=True)
        self.pa = self._as_mode_array(pa, "pa", broadcast=True)
        self.fa = np.zeros(self.nmodes)          # photon force, per mode
        self.ea = np.zeros(self.nmodes)          # field amplitude, per mode
        self.exc = exc
        self.number = number

    def __len__(self):
        return self.nmodes

    def __str__(self):
        return (f"photon_mode(nmodes={self.nmodes}, omega={self.omega}, "
                f"lam={self.lam})")

    def _as_mode_array(self, value, name, broadcast=False):
        # idempotent: scalar / (1,) / (nmodes,) all -> (nmodes,). Re-feeding an
        # existing (nmodes,) array is a no-op (no nesting). With broadcast=True a
        # lone scalar fills all modes (initial conditions); with broadcast=False a
        # length-1 input for nmodes > 1 is a length error (per-mode parameters).
        arr = np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)
        if arr.size == 1 and self.nmodes > 1 and broadcast:
            arr = np.full(self.nmodes, arr.item())
        if arr.shape != (self.nmodes,):
            raise ValueError(
                f"photon_mode {name!r} has length {arr.size}, expected {self.nmodes}.")
        return arr

    def _as_pol_array(self, pol_vec):
        # store (3, nmodes); accept 1-D (3,) [single mode], (3, nmodes), or
        # (nmodes, 3). The (3, 3) square case is read as (3, nmodes) (reference
        # convention), which is also what re-feeding an existing pt.pol_vec
        # produces -> idempotent.
        arr = np.asarray(pol_vec, dtype=float)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError(
                    f"pol_vec must be (3, {self.nmodes}); got {arr.shape}.")
            arr = arr.reshape(3, 1)
        elif arr.shape == (self.nmodes, 3) and arr.shape != (3, self.nmodes):
            arr = arr.T
        if arr.shape != (3, self.nmodes):
            raise ValueError(
                f"pol_vec must be (3, {self.nmodes}); got {np.asarray(pol_vec).shape}.")
        return self._normalize_pol(arr)

    @staticmethod
    def _normalize_pol(arr):
        # Each column (mode polarization) must be a UNIT vector: the CBOA
        # screening/force code (get_cboa_forces_bonini, and the single-mode
        # projected-response path) uses pol_vec directly, so a non-unit column
        # silently rescales the coupling by ||pol||. Normalize per column and warn
        # prominently if any column was not already unit. A zero-length column is
        # a degenerate (undefined) polarization -> hard error.
        norms = np.linalg.norm(arr, axis=0)          # (nmodes,)
        bad_norm = (norms == 0.0) | ~np.isfinite(norms)
        if np.any(bad_norm):
            bad_modes = np.flatnonzero(bad_norm).tolist()
            raise ValueError(
                f"pol_vec has a zero-length or non-finite column (mode(s) {bad_modes}); "
                "a photon polarization must be a nonzero finite direction.")
        # rtol=0: the warning contract is strict -- anything beyond atol of unit
        # length is reported, not silently absorbed by isclose's default rtol.
        off = ~np.isclose(norms, 1.0, rtol=0, atol=1e-8)
        if not np.any(off):
            return arr        # already unit: return unchanged (bit-stable on re-feed)
        bad = {int(m): float(norms[m]) for m in np.flatnonzero(off)}
        warnings.warn(
            "\n" + "!" * 72 + "\n"
            "photon_mode: pol_vec was NOT unit-normalized and has been "
            "normalized.\n"
            f"  rescaled mode(s) -> original |pol|: {bad}\n"
            "  The polarization direction is unchanged; its LENGTH is now 1.\n"
            "  (Coupling strength belongs in `lam`, not in the length of "
            "pol_vec.)\n"
            + "!" * 72,
            stacklevel=3,
        )
        out = np.array(arr, copy=True)
        out[:, off] = arr[:, off] / norms[off]        # rescale only the off columns
        return out
