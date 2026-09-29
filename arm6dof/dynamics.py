"""Dynamic torque per joint -> servo sizing for MOTION, not just holding.

torque.py answers "can the servo HOLD the pose?". This answers "can it also
ACCELERATE the arm?" -- the term that actually decides whether the arm can move
at a useful speed, which cheap servos often fail even when they can hold static.

Full rigid-body dynamics is  tau = M(q).qdd + C(q,qd).qd + g(q). We compute a
principled, data-light estimate using ONLY the point-mass lumps already in
model.py (no inertia tensors required):

  * g(q)      -- the gravity term = torque.joint_torques (already exact for the
                 lumped model).
  * M(q).qdd  -- effective inertia each joint sees from every mass DISTAL to it,
                 times a user-supplied peak angular acceleration. For a point
                 mass at perpendicular distance r_perp from joint axis j,
                 its contribution to that joint's inertia is m * r_perp^2
                 (parallel-axis, point mass has no self-inertia).
  * C(q,qd).qd -- velocity/Coriolis term. Omitted here (small at hobby speeds,
                 and needs qd); the safety factor and a note cover it. For exact
                 numbers use roboticstoolbox.rne once links carry inertia.

This is the standard "gravity + inertia at peak accel" servo-sizing estimate.
Honest about what it drops; good enough to choose STS3215 vs a geared joint.
"""
from __future__ import annotations

import numpy as np

from .forward import joint_frames
from .model import ArmModel, default_arm
from .torque import joint_torques, _world_point, SAFETY_FACTOR

# default peak angular acceleration per joint (rad/s^2) if the caller gives none.
# ~8 rad/s^2 reaches 90 deg/s in ~0.2 s -- a brisk but realistic hobby move.
DEFAULT_PEAK_ACCEL = 8.0


def joint_inertia(model: ArmModel, q) -> np.ndarray:
    """Effective inertia (kg.m^2) each joint sees from all distal masses at q.

    Diagonal approximation of M(q): for joint j, sum m * r_perp^2 over every
    mass further out the chain, where r_perp is the distance from the mass to
    joint j's axis line.
    """
    frames = joint_frames(model, q)

    # (world_pos, mass, distal_of_joint_index)
    masses: list[tuple[np.ndarray, float, int]] = []
    for lm in model.link_masses:
        masses.append((_world_point(frames[lm.at_joint], lm.com), lm.mass,
                       lm.at_joint))
    masses.append((_world_point(frames[model.dof], model.tool_offset),
                   model.payload, model.dof))

    inertia = np.zeros(model.dof)
    for j in range(model.dof):
        origin_j = frames[j][:3, 3]
        axis_j = frames[j][:3, 2]
        axis_j = axis_j / (np.linalg.norm(axis_j) or 1.0)
        for pos, m, distal_of in masses:
            if distal_of > j and m > 0:
                r = pos - origin_j
                # perpendicular distance from the mass to the joint axis line
                r_perp = np.linalg.norm(r - np.dot(r, axis_j) * axis_j)
                inertia[j] += m * r_perp ** 2
    return inertia


def dynamic_torques(model: ArmModel, q, peak_accel: float = DEFAULT_PEAK_ACCEL):
    """Total torque (N.m) per joint = |gravity| + inertia * peak_accel.

    peak_accel is a scalar rad/s^2 applied to every joint (worst case: all
    joints commanded to peak accel at once). Returns absolute magnitudes.
    """
    g_term = np.abs(joint_torques(model, q))
    inertia = joint_inertia(model, q)
    accel_term = inertia * float(peak_accel)
    return g_term + accel_term


def sizing_report(model: ArmModel, q, peak_accel: float = DEFAULT_PEAK_ACCEL):
    """Per-joint dynamic sizing: static, inertia, accel torque, total, verdict."""
    g_term = np.abs(joint_torques(model, q))
    inertia = joint_inertia(model, q)
    accel_term = inertia * float(peak_accel)
    total = g_term + accel_term

    rows = []
    for j, spec in enumerate(model.joints):
        rated = spec.servo_stall_torque * spec.gear_ratio
        required = total[j] * SAFETY_FACTOR
        rows.append({
            "joint": spec.name,
            "servo": spec.servo_model,
            "static_Nm": round(float(g_term[j]), 3),
            "inertia_kgm2": round(float(inertia[j]), 4),
            "accel_Nm": round(float(accel_term[j]), 3),
            "total_Nm": round(float(total[j]), 3),
            "required_with_margin_Nm": round(float(required), 3),
            "servo_rated_Nm": round(float(rated), 3),
            "ok": bool(rated >= required),
        })
    return rows


if __name__ == "__main__":
    from .torque import worst_case_pose
    m = default_arm()
    q = worst_case_pose(m)
    pa = DEFAULT_PEAK_ACCEL
    print(f"{m.name}\nDYNAMIC sizing (gravity + inertia*{pa} rad/s^2), "
          f"payload {m.payload*1000:.0f} g, safety x{SAFETY_FACTOR}:\n")
    hdr = (f"{'joint':16} {'static':>7} {'inertia':>8} {'accel':>7} "
           f"{'total':>7} {'need':>7} {'rated':>7}  ok")
    print(hdr); print("-" * len(hdr))
    for r in sizing_report(m, q, pa):
        print(f"{r['joint']:16} {r['static_Nm']:7.3f} {r['inertia_kgm2']:8.4f} "
              f"{r['accel_Nm']:7.3f} {r['total_Nm']:7.3f} "
              f"{r['required_with_margin_Nm']:7.3f} {r['servo_rated_Nm']:7.3f}  "
              f"{'PASS' if r['ok'] else 'FAIL'}")
