# UR3_6axis_Desk

A desktop **6-axis robotic arm** built from hobby serial-bus servos, plus
**`arm6dof`**, a Python framework that works out the arm's kinematics and
**servo sizing** in simulation *before any parts are bought*. The target is a
**500 g payload** at the tool tip.

> The UR3 form factor is only the inspiration for the name. This is an
> independent, hobby-scale build. It is not a Universal Robots or ABB product.

---

## Why the software comes first

You can't choose the servos until you know how much torque each joint must
produce. You can't know that until the arm's geometry exists as a formal model.
So the build order is deliberate:

```
model geometry → forward kinematics → static + dynamic torque sizing
              → BUY SERVOS → inverse kinematics → control → hardware
```

This repository covers the **software and analysis layers**. The sizing tools
already show that a bare STS3215-class servo build does **not** hold 500 g at
full reach. Catching that before buying parts is the reason the framework
exists.

---

## Features

- **One geometry model** (`arm6dof/model.py`) drives everything: the DH table,
  link masses, joint limits, servo ratings, gear ratios, payload, tool offset
  and home pose.
- **Forward kinematics** with a NumPy-only DH product. It can also export the
  arm to a `roboticstoolbox` `DHRobot`.
- **Inverse kinematics** uses a Levenberg–Marquardt solver. Every result is
  checked by running it back through forward kinematics and comparing against
  tolerances and joint limits.
- **Static sizing** answers "can each servo *hold* this pose?"
- **Dynamic sizing** answers "can each servo also *accelerate* the arm?" It
  adds gravity to inertia × peak acceleration.
- **Presets**: the original placeholder arm (`custom`) and an **IRB 1300-7/1.4**
  geometry template.
- **Interactive GUI**:
  - joint sliders and a live 3D view with frame triads for each joint
  - tool-tip (TCP) pose readout
  - a live editor for DH geometry, payload and gear ratios
  - a PASS/FAIL table for each joint, in static or dynamic mode
- **JSON config save/load** keeps different designs side by side. You can also
  set one as the default the app opens with.

---

## Project layout

| Path | Role |
|---|---|
| `arm6dof/model.py` | **Single source of truth.** `JointSpec`, `LinkMass`, `ArmModel`, the `REFERENCE_ARM` and `IRB1300_7_14` presets, the `PRESETS` registry and `default_arm()`. |
| `arm6dof/config.py` | Saves and loads the editable fields as JSON: a/d/α/θ₀ per joint, gear ratios, payload and home pose. Loads `config_default.json` at startup. |
| `arm6dof/transforms.py` | Homogeneous-transform helpers (NumPy only). |
| `arm6dof/forward.py` | Forward kinematics (`fk_chain`, `joint_frames`) and the `to_rtb()` exporter to roboticstoolbox. |
| `arm6dof/inverse.py` | Numerical IK (`ik`), `pose_error`, and a documented stub for analytic IK using Pieper's method. |
| `arm6dof/torque.py` | Static gravity torque per joint, `sizing_report` and `worst_case_pose`. |
| `arm6dof/dynamics.py` | Effective joint inertia and dynamic torque (gravity + inertia × peak acceleration), with its own sizing report. |
| `arm6dof/viz.py` | Standalone 3D matplotlib plot of a configuration. |
| `arm6dof/gui.py` | Tkinter control panel (see [GUI](#gui)). |
| `arm6dof/theme.py` | GUI design tokens: palette, grid and fonts. |
| `tests/` | Tests for FK, FK↔IK round-trips, and static and dynamic torque. |

---

## Quick start

Requires **Python 3.10+**. The GUI also needs Tkinter, which comes with the
python.org installers. On Debian and Ubuntu, run `sudo apt install python3-tk`.

```bash
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
python -m arm6dof.forward     # tool-tip pose at the home configuration
python -m arm6dof.torque      # static sizing table at the worst-case (horizontal) pose
python -m arm6dof.dynamics    # dynamic sizing table (gravity + inertia at 8 rad/s²)
python -m arm6dof.inverse     # IK round-trip demo (needs roboticstoolbox)
python -m arm6dof.viz         # static 3D plot
python -m arm6dof.gui         # interactive control panel
pytest -q                     # tests
```

These commands all use `default_arm()`. That is the **IRB 1300-7/1.4
template**, with `arm6dof/config_default.json` applied on top if that file
exists.

Dependencies (`requirements.txt`): `numpy`, `matplotlib`,
`roboticstoolbox-python` (only needed for IK and `to_rtb`) and `pytest`.

---

## Model and conventions

- **Units:** metres, kilograms and radians inside the model. The GUI and JSON
  config use millimetres, grams and degrees.
- **Frames:** standard (proximal) **Denavit–Hartenberg**. Each joint is
  `(a, α, d, θ)`, where θ = commanded `q` + `theta_offset`.
- **Kinematic structure:** 6R with a **spherical wrist**, meaning axes 4, 5 and
  6 meet at one point. That is what makes a closed-form IK possible later.
- **Masses:** point-mass lumps (`LinkMass`), each placed in a joint's frame.
  This is enough for servo sizing. Full inertia tensors are a later refinement.
- **Servo rating:** `servo_stall_torque × gear_ratio`. A `gear_ratio` above 1
  models an external reduction.

### Presets

| Preset | Geometry | Notes |
|---|---|---|
| `custom` (`REFERENCE_ARM`) | ~500 mm reach, STS3215-class | The original **placeholder** geometry. Realistic proportions, but not measured. |
| `IRB 1300-7/1.4` (default) | ~1.4 m reach | **Reconstructed** standard DH. ABB does not publish DH tables. The structure and α pattern are correct and the proportions are close. Masses and servos stay STS3215-class. This is a **geometry template to scale down**, not a model of the industrial robot. |

> ⚠️ Neither preset is measured hardware. Every sizing number depends on real
> link lengths, masses and centres of mass (see [Plan](#plan)).

---

## Servo sizing

**Static** (`torque.py`): for each joint, it sums `(r × F) · axis` over every
gravity load further out along the arm. Those loads are the link masses and the
payload at the tool tip. A joint **passes** when:

```
servo_stall_torque × gear_ratio  ≥  |τ_static| × SAFETY_FACTOR   (2.5)
```

**Dynamic** (`dynamics.py`): it adds an inertia term to the gravity term:

```
τ_total = |g(q)| + I_eff(q) · peak_accel        I_eff = Σ m · r⊥²  (distal masses)
```

The default peak acceleration is **8 rad/s²**, which reaches 90°/s in about
0.2 s. The Coriolis and velocity term is left out on purpose because it is
small at hobby speeds. Use `roboticstoolbox.rne` once the links carry real
inertia tensors.

A `FAIL` row means that joint needs one of three fixes: a stronger servo, an
external gear reduction, or a shorter reach.

---

## GUI

`python -m arm6dof.gui` opens a dark control panel with these parts:

- **Scan**: a live 3D view of the arm. Each joint frame is drawn as a triad
  (X red, Y green, Z blue) and the TCP is marked with a red ✕.
- **Joints**: one slider per joint, limited to that joint's range, plus a
  **Home** button that returns to the model's `home_deg` pose.
- **DH geometry**: a **preset picker** and an editor row per joint for a/d (mm),
  α (−90/0/90/180°) and θ₀ (deg). The arm and the sizing table update live.
  - **Save as default** writes `arm6dof/config_default.json`. The file is
    gitignored because it holds per-machine settings.
  - **Save as… / Load…** write and read any named JSON config.
- **TCP**: tool-tip position in mm.
- **Payload**: a slider from 0 to 2000 g.
- **Servo sizing**: a PASS/FAIL table for the current pose. Click the mode
  label to switch between **static** and **dynamic**. Dynamic mode adds a
  peak-acceleration slider (0–40 rad/s²). The table also has a **gear ratio**
  spinbox for each joint.

The GUI styling uses Cormorant Garamond Italic if `theme.py` finds it on the
local machine. Otherwise it falls back to Georgia, so the app runs without the
font.

### Config file format

```json
{
  "schema": 1,
  "name": "IRB 1300-7/1.4 (reconstructed standard DH, geometry template)",
  "payload_g": 500.0,
  "home_deg": [0, 0, 0, 0, 0, 0],
  "joints": [
    {"name": "J1_base", "a_mm": 150.0, "d_mm": 544.0,
     "alpha_deg": -90.0, "theta_offset_deg": 0.0, "gear_ratio": 1.0}
  ]
}
```

Only these fields can be overridden. Joint limits, servo models and link masses
always come from `model.py`. Joints are matched by position, and any missing
field keeps the base model's value.

---

## Tests

```bash
pytest -q
```

- **`test_kinematics.py`**: DOF count, FK determinism, a valid bottom row for
  the homogeneous transform, frame count, and rejection of a `q` with the wrong
  length. It also checks that the static torque results have the right shape
  and one sizing row per joint.
- **`test_dynamics.py`**:
  - effective inertia is never negative
  - dynamic torque equals static torque at zero acceleration
  - dynamic torque is greater than static when accelerating
  - total torque ≥ static torque for every joint
- **`test_ik_roundtrip.py`**: poses produced by FK are solved back by IK. These
  tests **skip automatically** if `roboticstoolbox` is not installed.

The kinematics and dynamics tests need only NumPy.

---

## Status of the analysis

These results come from the placeholder **`custom`** geometry at the horizontal
worst-case pose, with a 500 g payload and a ×2.5 safety factor:

| Joint | Static hold | Dynamic (8 rad/s²) | Bare STS3215 (30 kgf·cm) |
|---|---|---|---|
| Base J1 | PASS | **FAIL** (rotational inertia) | insufficient |
| Shoulder J2 (2× STS3215) | **FAIL** | **FAIL** | needs gearing or a bigger servo |
| Elbow J3 | **FAIL** | **FAIL** | needs gearing or a bigger servo |
| Wrist J4–J6 | PASS | PASS | ok |

**Conclusion so far:** the lower three joints need more than a bare STS3215.
The IRB 1300 template reaches much further, so it puts even more load on those
joints. Run `python -m arm6dof.dynamics`, or use the GUI's gear-ratio column,
to see the numbers for any geometry.

---

## Plan

### ✅ Done

- Kinematics framework (FK, numerical IK, transforms) for a 6R arm with a
  spherical wrist.
- Static **and** dynamic servo-sizing calculators, both driven by one geometry
  model.
- Interactive GUI with live PASS/FAIL for each joint, and live editing of DH
  geometry, payload and gear ratios.
- A preset registry (`custom`, `IRB 1300-7/1.4`) and JSON config save/load.
- Tests for the geometry-first layers.

### ⏳ Open decisions

1. **Physical frame.** Choose between a printed frame (SO-ARM or UR-style STL
   set) and a metal-bracket kit. This fixes the real link lengths and masses.
2. **Real geometry.** Measure link lengths, masses and centres of mass, then
   replace the placeholders.
3. **Servo per joint.** A bare STS3215 fails the lower joints at 500 g. For
   each joint, pick a bigger servo, an external **gear reduction**, or a
   shorter reach.
4. **Wrist geometry.** Confirm that the last three axes really intersect.
   Otherwise, stay on numerical IK.
5. **Controller.** Options are Arduino + PCA9685 (PWM), ESP32 (wireless), or a
   Raspberry Pi (Python and IK on board). Serial-bus servos change this choice
   compared with PWM servos.
6. **Power budget.** Size the power supply for the combined stall current of
   all servos.
7. **Gripper / end effector.** Decide its type and weight, and how much of the
   payload it uses.

### ▶️ Next actions

- [ ] Choose the frame and measure its geometry. Enter the values through the
      GUI, then **Save as default**, or add a new preset in `model.py`.
- [ ] Adjust servos and gear ratios until **every joint passes the dynamic
      check** at 500 g and a realistic peak acceleration.
- [ ] Write a concrete **shopping list** that passes the sizing check: servos,
      driver board, power supply, frame and gripper.
- [ ] Implement `analytic_ik` (Pieper's method) once the wrist geometry is
      confirmed.
- [ ] Add a `hardware/` layer that turns IK joint angles into servo commands
      (STS3215: 0–4095 counts; PWM: microsecond pulses).
- [ ] Optional: add the Coriolis term via `roboticstoolbox.rne` once the links
      carry real inertia tensors.
- [ ] Add tests for the `config.py` round-trip and the preset registry.

---
