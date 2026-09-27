"""Inverse kinematics: desired tool pose -> joint angles.

Strategy:
  * Primary path uses roboticstoolbox's numerical solver `ikine_LM`
    (Levenberg-Marquardt), seeded from a start guess and constrained to the
    model's joint limits. Robust for a general 6R arm and needs no hand-derived
    algebra.
  * Because our arm has a SPHERICAL WRIST (axes 4,5,6 intersect), a closed-form
    analytic solution is also possible (Pieper). That is a later optimisation;
    `analytic_ik` is stubbed with the decomposition steps documented so the
    structure is ready when we want deterministic, multi-solution IK.

Every result is validated by round-tripping through forward.fk_chain and
checking the position/orientation error, so a silent bad solve can't slip
through.
"""
from __future__ import annotations

import numpy as np

from .forward import fk_chain, to_rtb
from .model import ArmModel, default_arm


def pose_error(model: ArmModel, q, target: np.ndarray) -> tuple[float, float]:
    """(position error [m], orientation error [rad]) of q vs a target pose."""
    achieved = fk_chain(model, q)
    dp = np.linalg.norm(achieved[:3, 3] - target[:3, 3])

    R_err = achieved[:3, :3].T @ target[:3, :3]
    cos_ang = (np.trace(R_err) - 1.0) / 2.0
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    dtheta = float(np.arccos(cos_ang))
    return float(dp), dtheta


def ik(model: ArmModel, target: np.ndarray, q0=None,
       pos_tol: float = 1e-4, rot_tol: float = 1e-3):
    """Solve IK for a 4x4 target pose. Returns (q, ok, (pos_err, rot_err)).

    `ok` is True only when the round-trip error is within tolerance AND the
    solution respects joint limits.
    """
    robot = to_rtb(model)  # raises a clear error if rtb missing

    from spatialmath import SE3
    Tep = SE3(target, check=False)

    if q0 is None:
        q0 = np.zeros(model.dof)

    sol = robot.ikine_LM(Tep, q0=q0, joint_limits=True)
    q = np.asarray(sol.q, dtype=float)

    dp, dr = pose_error(model, q, target)
    lo, hi = model.q_limits()
    within_limits = bool(np.all(q >= np.array(lo) - 1e-6) and
                         np.all(q <= np.array(hi) + 1e-6))
    ok = bool(sol.success and dp <= pos_tol and dr <= rot_tol and within_limits)
    return q, ok, (dp, dr)


def analytic_ik(model: ArmModel, target: np.ndarray):  # pragma: no cover
    """Closed-form IK for the spherical-wrist arm (Pieper decomposition).

    NOT YET IMPLEMENTED -- documented so the analytic path can be filled in:
      1. wrist centre  = target_position - d6 * target_z_axis
      2. solve q1,q2,q3 from the wrist-centre position (planar 2-link geometry
         plus base rotation) -> up to 4 arm configurations
      3. solve q4,q5,q6 from the residual orientation R_3_6 = R_0_3^T @ R_target
         -> two wrist flips each
    Yields up to 8 solutions; pick the reachable one nearest q0.
    """
    raise NotImplementedError(
        "analytic_ik is a documented stub; use ik() (numerical) for now."
    )


if __name__ == "__main__":
    m = default_arm()
    # take a known pose from FK, perturb the seed, and see if IK recovers it
    q_true = np.array([0.3, 0.4, -0.6, 0.2, 0.5, -0.1])
    target = fk_chain(m, q_true)
    try:
        q, ok, err = ik(m, target, q0=np.zeros(m.dof))
        print("IK ok:", ok, "pos_err(m):", round(err[0], 6),
              "rot_err(rad):", round(err[1], 6))
        print("q solved:", np.round(q, 4))
    except ImportError as e:
        print(e)
