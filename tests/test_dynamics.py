"""Dynamics tests (NumPy only)."""
import numpy as np

from arm6dof.model import default_arm
from arm6dof.torque import worst_case_pose, joint_torques
from arm6dof.dynamics import (
    joint_inertia, dynamic_torques, sizing_report, DEFAULT_PEAK_ACCEL,
)


def test_inertia_nonnegative_and_shaped():
    m = default_arm()
    I = joint_inertia(m, worst_case_pose(m))
    assert I.shape == (6,)
    assert np.all(I >= 0)


def test_dynamic_reduces_to_static_at_zero_accel():
    """With no acceleration, dynamic torque == static gravity torque."""
    m = default_arm()
    q = worst_case_pose(m)
    dyn = dynamic_torques(m, q, peak_accel=0.0)
    stat = np.abs(joint_torques(m, q))
    assert np.allclose(dyn, stat)


def test_dynamic_exceeds_static_when_accelerating():
    m = default_arm()
    q = worst_case_pose(m)
    dyn = dynamic_torques(m, q, peak_accel=DEFAULT_PEAK_ACCEL)
    stat = np.abs(joint_torques(m, q))
    # at least one joint carrying distal mass must need more torque to move
    assert np.any(dyn > stat + 1e-9)


def test_sizing_report_shape():
    m = default_arm()
    rows = sizing_report(m, worst_case_pose(m), DEFAULT_PEAK_ACCEL)
    assert len(rows) == 6
    for r in rows:
        assert r["total_Nm"] >= r["static_Nm"] - 1e-9
        assert "ok" in r
