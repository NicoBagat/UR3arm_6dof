"""Static torque per joint -> servo sizing and payload feasibility.

This is the part off-the-shelf robotics libraries don't hand you and the part
that decides which AliExpress servos to actually buy. It answers: "at this pose,
holding this payload, how much torque does each joint's servo have to produce,
and is that within the servo's rated stall torque with a safety margin?"

Method (quasi-static, gravity only -- no acceleration):
  * place the arm at pose q, get every joint frame (forward.joint_frames)
  * gather all gravitational forces: each link lump mass and the tip payload,
    expressed in world coordinates
  * for each joint axis, sum the torque about that axis from every mass that is
    DISTAL to it (further out the chain). tau = sum( r x F ) . axis
This ignores dynamics (inertia, speed) so add a margin; it is the standard first
cut for picking servos.
"""
from __future__ import annotations

import numpy as np

from .forward import joint_frames
from .model import ArmModel, default_arm

# recommended headroom: servo rated stall torque should exceed the static
# holding torque by at least this factor (covers acceleration, friction,
# voltage sag, and stall-torque optimism in cheap servo datasheets)
SAFETY_FACTOR = 2.5


def _world_point(frame: np.ndarray, local_xyz) -> np.ndarray:
    p = np.ones(4)
    p[:3] = local_xyz
    return (frame @ p)[:3]


def joint_torques(model: ArmModel, q) -> np.ndarray:
    """Static holding torque (N.m) required at each joint at pose q."""
    frames = joint_frames(model, q)          # dof+1 frames, world-referenced
    g = np.array([0.0, 0.0, -model.gravity])

    # build the list of (world_position, world_force) gravity loads
    loads: list[tuple[np.ndarray, np.ndarray, int]] = []  # (pos, force, distal_of)
    for lm in model.link_masses:
        pos = _world_point(frames[lm.at_joint], lm.com)
        loads.append((pos, lm.mass * g, lm.at_joint))

    # payload at the tool tip (frame dof, offset by tool_offset)
    tip = _world_point(frames[model.dof], model.tool_offset)
    loads.append((tip, model.payload * g, model.dof))

    tau = np.zeros(model.dof)
    for j in range(model.dof):
        # joint j axis is the Z of frame j (in world coords)
        origin_j = frames[j][:3, 3]
        axis_j = frames[j][:3, 2]
        for pos, force, distal_of in loads:
            if distal_of > j:  # only masses further out load this joint
                r = pos - origin_j
                tau[j] += np.dot(np.cross(r, force), axis_j)
    return tau


def sizing_report(model: ArmModel, q) -> list[dict]:
    """Per-joint: required torque, servo rating, margin, pass/fail."""
    tau = np.abs(joint_torques(model, q))
    rows = []
    for j, spec in enumerate(model.joints):
        rated = spec.servo_stall_torque * spec.gear_ratio
        required = tau[j] * SAFETY_FACTOR
        rows.append(
            {
                "joint": spec.name,
                "servo": spec.servo_model,
                "static_Nm": round(float(tau[j]), 3),
                "required_with_margin_Nm": round(float(required), 3),
                "servo_rated_Nm": round(float(rated), 3),
                "ok": bool(rated >= required),
            }
        )
    return rows


def worst_case_pose(model: ArmModel) -> np.ndarray:
    """Arm stretched straight out horizontally -- the max-torque configuration.

    Shoulder and elbow at the angles that lay the arm flat along +X, which
    maximises the lever arm of every distal mass. This is what you size for.
    """
    q = np.zeros(model.dof)
    # with the reference DH, q2 offset is +pi/2 (arm points up at zero); rotate
    # shoulder so the upper arm is horizontal, keep elbow straight.
    q[1] = -np.pi / 2
    return q


if __name__ == "__main__":
    m = default_arm()
    q = worst_case_pose(m)
    print(f"{m.name}\nWorst-case (arm horizontal) servo sizing, "
          f"payload {m.payload*1000:.0f} g, safety x{SAFETY_FACTOR}:\n")
    hdr = f"{'joint':16} {'servo':12} {'static':>8} {'needed':>8} {'rated':>8}  ok"
    print(hdr)
    print("-" * len(hdr))
    for r in sizing_report(m, q):
        print(f"{r['joint']:16} {r['servo']:12} "
              f"{r['static_Nm']:8.3f} {r['required_with_margin_Nm']:8.3f} "
              f"{r['servo_rated_Nm']:8.3f}  {'PASS' if r['ok'] else 'FAIL'}")
