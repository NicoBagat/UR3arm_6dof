"""IK round-trip test. Skips automatically if roboticstoolbox isn't installed."""
import numpy as np
import pytest

from arm6dof.model import default_arm
from arm6dof.forward import fk_chain

rtb = pytest.importorskip(
    "roboticstoolbox",
    reason="roboticstoolbox not installed; run pip install -r requirements.txt",
)

from arm6dof.inverse import ik  # noqa: E402


@pytest.mark.parametrize("q_true", [
    np.array([0.0, -0.5, 0.7, 0.0, 0.3, 0.0]),
    np.array([0.4, -0.3, 0.5, 0.2, -0.4, 0.6]),
    np.array([-0.6, 0.2, -0.8, 0.5, 0.5, -0.3]),
])
def test_ik_recovers_fk_pose(q_true):
    """A pose produced by FK must be solvable back by IK to that pose."""
    m = default_arm()
    target = fk_chain(m, q_true)
    q, ok, err = ik(m, target, q0=np.zeros(6))
    assert ok, f"IK failed: pos_err={err[0]:.2e} rot_err={err[1]:.2e}"
    # the recovered pose matches the target (joint values themselves may differ
    # because a 6R arm has multiple solutions -- pose equality is what matters)
    assert np.allclose(fk_chain(m, q)[:3, 3], target[:3, 3], atol=1e-3)
