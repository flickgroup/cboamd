"""Two degenerate perpendicular cavity modes vs one axial mode on CO2 (Task 8).

With the CO2 axis along x, a single mode polarized along x and two degenerate
modes polarized along x and y give the same Rabi splitting of the asymmetric-
stretch peak (the y mode is IR-dark for an x-aligned linear molecule), confirming
multi-mode reduces correctly and the modes are independent under the matrix
screening. End-to-end through the real NEP MD pipeline.
"""


def test_two_mode_split_matches_single_mode(two_mode_collected):
    s1 = two_mode_collected["split_1mode"]
    s2 = two_mode_collected["split_2mode"]
    assert s1 > 200.0                                   # a real, resolved Rabi split
    assert abs(s2 - s1) < 80.0                           # degenerate y-mode is a spectator


def test_two_mode_trajectory_energy_bounded(two_mode_collected):
    # NVE: total-energy drift over the short trajectory stays small (stability guard)
    assert two_mode_collected["energy2_drift"] < 0.5     # eV
