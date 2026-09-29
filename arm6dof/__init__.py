"""arm6dof -- kinematics & servo-sizing framework for a 6-axis servo arm.

Geometry lives in model.py (edit that one file to match your real hardware).
"""
from .model import ArmModel, JointSpec, LinkMass, default_arm  # noqa: F401
from .forward import fk_chain, joint_frames, to_rtb            # noqa: F401
from .inverse import ik, pose_error                            # noqa: F401
from .torque import joint_torques, sizing_report, worst_case_pose  # noqa: F401

__all__ = [
    "ArmModel", "JointSpec", "LinkMass", "default_arm",
    "fk_chain", "joint_frames", "to_rtb",
    "ik", "pose_error",
    "joint_torques", "sizing_report", "worst_case_pose",
]
