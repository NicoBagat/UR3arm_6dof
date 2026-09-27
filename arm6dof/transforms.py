"""Small, dependency-light homogeneous-transform helpers (NumPy only).

Even though roboticstoolbox supplies its own SE3 type, we keep these so
torque.py and our own tests can reason about frames without pulling the whole
library into every unit test. A 4x4 homogeneous transform packs a 3x3 rotation
and a 3x1 translation:

    [ R  p ]
    [ 0  1 ]
"""
from __future__ import annotations

import numpy as np


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard (proximal) DH homogeneous transform for one joint.

    theta already includes the joint variable q plus the model's theta_offset.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,      sa,       ca,      d],
            [0.0,     0.0,      0.0,    1.0],
        ]
    )


def rotation_of(T: np.ndarray) -> np.ndarray:
    return T[:3, :3]


def translation_of(T: np.ndarray) -> np.ndarray:
    return T[:3, 3]


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Chain transforms left-to-right: compose(A, B, C) == A @ B @ C."""
    out = np.eye(4)
    for T in transforms:
        out = out @ T
    return out
