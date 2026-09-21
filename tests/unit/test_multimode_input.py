"""Unit tests for build_photon_mode (Task 4): construct a (possibly multi-mode)
photon_mode from the JSON input dict, with JSON-serializable qa0/pa0."""
import json

import numpy as np
import pytest

from cboamd.pymd_ase import build_photon_mode


def _jdata(**kw):
    base = dict(nphoton=2, omega_photon=[0.0111, 0.0111],
                lambda_photon=[0.05, 0.05],
                lambda_vector=[[1, 0, 0], [0, 1, 0]])
    base.update(kw)
    return base


def test_build_two_modes():
    pt = build_photon_mode(_jdata(), 'e-mode')
    assert pt.nmodes == 2
    np.testing.assert_allclose(pt.omega, [0.0111, 0.0111])
    # JSON lambda_vector is (M,3); stored as (3,M) -- columns are the mode pols
    assert pt.pol_vec.shape == (3, 2)
    np.testing.assert_allclose(pt.pol_vec[:, 0], [1, 0, 0])
    np.testing.assert_allclose(pt.pol_vec[:, 1], [0, 1, 0])
    assert pt.cboa_mode == 'e-mode'
    np.testing.assert_allclose(pt.qa, [0.0, 0.0])


def test_qa0_scalar_broadcasts():
    pt = build_photon_mode(_jdata(qa0=0.3), 'e-mode')
    np.testing.assert_allclose(pt.qa0, [0.3, 0.3])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_photon_mode(_jdata(lambda_photon=[0.05]), 'e-mode')


def test_single_mode_still_works():
    pt = build_photon_mode(
        dict(nphoton=1, omega_photon=[0.01107], lambda_photon=[0.05],
             lambda_vector=[[1, 0, 0]]), 'q-mode')
    assert pt.nmodes == 1 and pt.cboa_mode == 'q-mode'
    assert pt.pol_vec.shape == (3, 1)


def test_qa0_pa0_json_serializable():
    pt = build_photon_mode(_jdata(qa0=0.3, pa0=[1.0, 2.0]), 'e-mode')
    json.dumps({'qa0': pt.qa0.tolist(), 'pa0': pt.pa0.tolist()})  # must not raise
