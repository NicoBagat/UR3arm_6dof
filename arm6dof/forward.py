"""Forward kinematics.

Two implementations that MUST agree (see tests/):

1. `fk_chain`   -- our own DH matrix product, NumPy only, no external deps.
                   This is the reference we trust and test against.
2. `to_rtb`     -- builds a roboticstoolbox DHRobot from the same model, so we
                   get its IK, Jacobian, plotting and collision tools for free.

Keeping both lets us cross-check the library against first principles and lets
the unit tests run even on a machine where roboticstoolbox is not installed.
"""
from __future__ import annotations

import numpy as np

from .model import ArmModel, default_arm
from .transforms import compose, dh_transform


def fk_chain(model: ArmModel, q) -> np.ndarray:
    """Tool-tip pose (4x4) for joint vector q, from our own DH product.

    Includes the tool offset (flange -> TCP) as a final translation.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (model.dof,):
        raise ValueError(f"q must have {model.dof} elements, got {q.shape}")

    T = np.eye(4)
    for (a, alpha, d, theta_off), qi in zip(model.dh_table(), q):
        T = T @ dh_transform(a, alpha, d, theta_off + qi)

    tool = np.eye(4)
    tool[:3, 3] = model.tool_offset
    return T @ tool


def joint_frames(model: ArmModel, q) -> list[np.ndarray]:
    """Cumulative transform of every joint frame (frame 0..dof), for viz/torque.

    Returns dof+1 transforms: the base (identity) followed by each joint frame.
    """
    q = np.asarray(q, dtype=float)
    frames = [np.eye(4)]
    T = np.eye(4)
    for (a, alpha, d, theta_off), qi in zip(model.dh_table(), q):
        T = T @ dh_transform(a, alpha, d, theta_off + qi)
        frames.append(T.copy())
    return frames


def to_rtb(model: ArmModel):
    """Build a roboticstoolbox DHRobot from the model (lazy import).

    Raises a clear error if roboticstoolbox isn't installed, so the NumPy path
    still works standalone.
    """
    try:
        import roboticstoolbox as rtb
        from roboticstoolbox import RevoluteDH
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "roboticstoolbox is not installed. Run: pip install -r requirements.txt"
        ) from exc

    links = [
        RevoluteDH(a=j.a, alpha=j.alpha, d=j.d, offset=j.theta_offset,
                   qlim=[j.q_min, j.q_max])
        for j in model.joints
    ]
    from spatialmath import SE3
    tool = SE3(*model.tool_offset)
    return rtb.DHRobot(links, name=model.name, tool=tool)


if __name__ == "__main__":
    m = default_arm()
    home = np.zeros(m.dof)
    pose = fk_chain(m, home)
    print(f"{m.name}\nDOF: {m.dof}")
    print("Tool tip at home pose (x, y, z) m:", np.round(pose[:3, 3], 4))
