"""Single source of truth for the 6-DOF arm geometry, masses and joint limits.

All lengths in metres, masses in kilograms, angles in radians unless a name
says otherwise. This module holds NO kinematics maths -- it only *describes*
the arm. forward.py / inverse.py / torque.py consume it.

The arm is a 6R (six revolute joints) manipulator with a spherical wrist:
joints 4, 5, 6 axes intersect at a single point (the wrist centre). That
property is what gives inverse.py a clean closed-form solution.

Frame convention: standard (proximal) Denavit-Hartenberg. Each joint i is
described by (a_i, alpha_i, d_i, theta_i):
    a     link length      : distance along X_i from Z_i to Z_{i+1}
    alpha link twist        : angle about X_i from Z_i to Z_{i+1}
    d     link offset       : distance along Z_i from X_{i-1} to X_i
    theta joint variable    : angle about Z_i from X_{i-1} to X_i  (the DOF)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, radians


@dataclass(frozen=True)
class JointSpec:
    """One revolute joint: its fixed DH row plus limits and servo data.

    theta here is the DH *offset* added to the commanded joint angle q, so the
    zero pose can be defined independently of where the servo reads zero.
    """

    name: str
    a: float          # link length (m)
    alpha: float      # link twist (rad)
    d: float          # link offset (m)
    theta_offset: float = 0.0   # constant DH theta added to q (rad)

    q_min: float = -pi          # software joint limit, lower (rad)
    q_max: float = pi           # software joint limit, upper (rad)

    servo_model: str = ""       # e.g. "STS3215"
    servo_stall_torque: float = 0.0   # N.m at rated voltage
    gear_ratio: float = 1.0     # >1 means external reduction on top of the servo


@dataclass(frozen=True)
class LinkMass:
    """A point-mass lump for static torque analysis.

    Position is the mass centre expressed in the frame of `at_joint` (the frame
    that moves with that joint), in metres. Good enough for servo sizing; swap
    for a full inertia tensor later if you do dynamics.
    """

    at_joint: int             # index 0..5 whose frame the com is expressed in
    mass: float               # kg
    com: tuple[float, float, float]  # centre of mass in that joint's frame (m)


@dataclass(frozen=True)
class ArmModel:
    name: str
    joints: tuple[JointSpec, ...]
    link_masses: tuple[LinkMass, ...] = field(default_factory=tuple)
    payload: float = 0.0                       # kg carried at the tool tip
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # flange->TCP (m)
    gravity: float = 9.80665                   # m/s^2

    @property
    def dof(self) -> int:
        return len(self.joints)

    def dh_table(self) -> list[tuple[float, float, float, float]]:
        """(a, alpha, d, theta_offset) rows, one per joint, in order."""
        return [(j.a, j.alpha, j.d, j.theta_offset) for j in self.joints]

    def q_limits(self) -> tuple[list[float], list[float]]:
        lo = [j.q_min for j in self.joints]
        hi = [j.q_max for j in self.joints]
        return lo, hi


# ---------------------------------------------------------------------------
# Reference design: STS3215 / SO-ARM class, sized for a 500 g tip payload.
#
# Link lengths and masses below are PLACEHOLDERS in the right ballpark for a
# ~500 mm reach printed arm. They exist so the whole pipeline (FK/IK/torque)
# runs and is testable TODAY. Replace each number with a measured value once
# the frame is chosen -- model.py is the only file you must edit to do so.
# ---------------------------------------------------------------------------

STS3215_STALL = 3.0   # N.m (~30 kgf.cm at 12 V), datasheet figure

REFERENCE_ARM = ArmModel(
    name="arm6dof-reference (STS3215 class, placeholder geometry)",
    joints=(
        # base yaw
        JointSpec("J1_base", a=0.0,   alpha=pi / 2, d=0.12, theta_offset=0.0,
                  q_min=radians(-170), q_max=radians(170),
                  servo_model="STS3215", servo_stall_torque=STS3215_STALL),
        # shoulder pitch  -- carries the whole arm, the heavy joint
        JointSpec("J2_shoulder", a=0.25, alpha=0.0, d=0.0, theta_offset=pi / 2,
                  q_min=radians(-90), q_max=radians(90),
                  servo_model="STS3215x2", servo_stall_torque=2 * STS3215_STALL),
        # elbow pitch
        JointSpec("J3_elbow", a=0.22, alpha=0.0, d=0.0, theta_offset=0.0,
                  q_min=radians(-150), q_max=radians(150),
                  servo_model="STS3215", servo_stall_torque=STS3215_STALL),
        # wrist roll  (spherical wrist starts here: axes 4,5,6 intersect)
        JointSpec("J4_wrist_roll", a=0.0, alpha=pi / 2, d=0.0, theta_offset=0.0,
                  q_min=radians(-170), q_max=radians(170),
                  servo_model="STS3215", servo_stall_torque=STS3215_STALL),
        # wrist pitch
        JointSpec("J5_wrist_pitch", a=0.0, alpha=-pi / 2, d=0.0, theta_offset=0.0,
                  q_min=radians(-110), q_max=radians(110),
                  servo_model="STS3215", servo_stall_torque=STS3215_STALL),
        # wrist yaw / flange
        JointSpec("J6_flange", a=0.0, alpha=0.0, d=0.06, theta_offset=0.0,
                  q_min=radians(-170), q_max=radians(170),
                  servo_model="STS3215", servo_stall_torque=STS3215_STALL),
    ),
    link_masses=(
        LinkMass(at_joint=1, mass=0.30, com=(0.125, 0.0, 0.0)),  # upper arm
        LinkMass(at_joint=2, mass=0.25, com=(0.110, 0.0, 0.0)),  # forearm
        LinkMass(at_joint=4, mass=0.15, com=(0.0, 0.0, 0.03)),   # wrist cluster
    ),
    payload=0.5,                       # <-- the 500 g target
    tool_offset=(0.0, 0.0, 0.08),      # 80 mm gripper reach past the flange
)


def default_arm() -> ArmModel:
    return REFERENCE_ARM
