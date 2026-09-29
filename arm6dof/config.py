"""Save / load the user-editable arm configuration as JSON.

The GUI lets you tune the parts of the model that are design decisions rather
than fixed maths: the DH distances (a, d) per joint, the external gear ratio
per joint, the TCP payload, and the home pose. This module persists exactly
those fields and re-applies them onto the base model, immutably.

Two destinations, both plain JSON:
  * the DEFAULT file (config_default.json next to this module) -- loaded
    automatically by default_arm() at startup when present, so "Save as
    default" makes the current tuning the one the app opens with.
  * any named file the user picks via "Save as..." -- for keeping alternative
    designs side by side and reloading them later.

Nothing here computes kinematics; it only describes overrides. model.py stays
the source of the fixed geometry (twists, joint limits, servo stall torque).
"""
from __future__ import annotations

import json
from dataclasses import replace
from math import degrees, radians
from pathlib import Path

from .model import ArmModel

DEFAULT_CONFIG = Path(__file__).with_name("config_default.json")

SCHEMA = 1


def config_from_model(model: ArmModel) -> dict:
    """Extract the editable fields from a model into a plain dict."""
    return {
        "schema": SCHEMA,
        "name": model.name,
        "payload_g": round(model.payload * 1000.0, 3),
        "home_deg": list(model.home_q_deg()),
        "joints": [
            {
                "name": j.name,
                "a_mm": round(j.a * 1000.0, 3),
                "d_mm": round(j.d * 1000.0, 3),
                "alpha_deg": round(degrees(j.alpha), 3),
                "theta_offset_deg": round(degrees(j.theta_offset), 3),
                "gear_ratio": round(j.gear_ratio, 3),
            }
            for j in model.joints
        ],
    }


def apply_config(model: ArmModel, cfg: dict) -> ArmModel:
    """Return a NEW model = base model with the config's editable fields applied.

    Unknown / missing fields fall back to the base model's value, so a partial
    or older config still loads. Joints are matched by position, guarded by dof.
    """
    joints = list(model.joints)
    cfg_joints = cfg.get("joints", [])
    new_joints = []
    for i, j in enumerate(joints):
        cj = cfg_joints[i] if i < len(cfg_joints) else {}
        new_joints.append(replace(
            j,
            a=float(cj.get("a_mm", j.a * 1000.0)) / 1000.0,
            d=float(cj.get("d_mm", j.d * 1000.0)) / 1000.0,
            alpha=radians(float(cj.get("alpha_deg", degrees(j.alpha)))),
            theta_offset=radians(float(
                cj.get("theta_offset_deg", degrees(j.theta_offset)))),
            gear_ratio=float(cj.get("gear_ratio", j.gear_ratio)),
        ))

    home_deg = cfg.get("home_deg")
    home = tuple(float(v) for v in home_deg) if home_deg and \
        len(home_deg) == model.dof else model.home_deg

    return replace(
        model,
        joints=tuple(new_joints),
        payload=float(cfg.get("payload_g", model.payload * 1000.0)) / 1000.0,
        home_deg=home,
    )


def save_config(model: ArmModel, path: str | Path) -> Path:
    """Write the model's editable config to a JSON file. Returns the path."""
    p = Path(path)
    p.write_text(json.dumps(config_from_model(model), indent=2), encoding="utf-8")
    return p


def load_config(path: str | Path) -> dict:
    """Read a config JSON file into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_default(model: ArmModel) -> ArmModel:
    """Apply config_default.json onto the base model if it exists, else return
    the base model unchanged. Safe to call at startup."""
    if DEFAULT_CONFIG.exists():
        try:
            return apply_config(model, load_config(DEFAULT_CONFIG))
        except Exception:
            return model
    return model
