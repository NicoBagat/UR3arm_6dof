"""Quick 3D visualisation of the arm at a joint configuration (matplotlib).

Uses only forward.joint_frames + matplotlib so it works without
roboticstoolbox. For richer interactive plots (Swift, collision, trajectories)
call forward.to_rtb(model).plot(q) instead.
"""
from __future__ import annotations

import numpy as np

from .forward import joint_frames
from .model import ArmModel, default_arm


def plot(model: ArmModel, q, show: bool = True, save_path: str | None = None):
    import matplotlib.pyplot as plt  # lazy import

    frames = joint_frames(model, q)
    pts = np.array([f[:3, 3] for f in frames])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", lw=3, ms=6, label="links")

    # draw each joint's local axes as short RGB triads
    L = 0.04
    for f in frames:
        o = f[:3, 3]
        for k, c in enumerate("rgb"):
            ax.plot([o[0], o[0] + L * f[0, k]],
                    [o[1], o[1] + L * f[1, k]],
                    [o[2], o[2] + L * f[2, k]], c)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"{model.name}\nq = {np.round(q, 3)}")
    reach = sum(abs(j.a) + abs(j.d) for j in model.joints) + 0.1
    ax.set_xlim(-reach, reach); ax.set_ylim(-reach, reach); ax.set_zlim(0, reach)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    m = default_arm()
    plot(m, np.array([0.3, -0.6, 0.8, 0.0, 0.4, 0.0]))
