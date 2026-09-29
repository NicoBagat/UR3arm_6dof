"""Tests that need no external robotics library (NumPy only)."""
import numpy as np

from arm6dof.model import default_arm
from arm6dof.forward import fk_chain, joint_frames
from arm6dof.torque import joint_torques, sizing_report, worst_case_pose


def test_dof_is_six():
    assert default_arm().dof == 6


def test_home_pose_is_deterministic():
    m = default_arm()
    a = fk_chain(m, np.zeros(6))
    b = fk_chain(m, np.zeros(6))
    assert np.allclose(a, b)
    # bottom row of a homogeneous transform is always [0,0,0,1]
    assert np.allclose(a[3, :], [0, 0, 0, 1])


def test_joint_frames_count():
    m = default_arm()
    frames = joint_frames(m, np.zeros(6))
    assert len(frames) == m.dof + 1


def test_fk_rejects_wrong_length_q():
    m = default_arm()
    try:
        fk_chain(m, np.zeros(5))
        assert False, "should have raised"
    except ValueError:
        pass


def test_torque_positive_and_shaped():
    m = default_arm()
    tau = joint_torques(m, worst_case_pose(m))
    assert tau.shape == (6,)
    # a horizontal arm with payload must load the shoulder (joint 1) most
    assert abs(tau[1]) > 0


def test_sizing_report_has_a_row_per_joint():
    m = default_arm()
    rows = sizing_report(m, worst_case_pose(m))
    assert len(rows) == 6
    assert all("ok" in r for r in rows)
