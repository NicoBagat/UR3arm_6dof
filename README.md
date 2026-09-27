# UR3_6axis_Desk

A desktop **6-axis robotic arm** built from AliExpress serial-bus servos, with a
Python framework for kinematics and **servo sizing** that is validated in
simulation *before any parts are bought*. Target: **500 g payload** at the tool
tip.

> Named after the UR3 form factor as inspiration. This is an independent,
> hobby-scale build — not a Universal Robots product.

---

## Why the software comes first

You cannot choose the servos until you know the torque each joint must produce,
and you cannot know that until the arm's geometry exists as a formal model. So
the build order is deliberate:

```
model geometry → forward kinematics → torque/dynamics sizing
              → BUY SERVOS → inverse kinematics → control → hardware
```

This repo currently covers the **software/analysis layers**. The torque and
dynamics tools already show that a bare STS3215-class servo build does **not**
hold 500 g at full reach — which is exactly the kind of decision this framework
exists to catch early.

---

## What's here now (`arm6dof/`)

| Module | Role |
|---|---|
| `model.py` | **Single source of truth** — DH table, link masses, joint limits, payload, tool offset. Edit this one file to match real hardware. |
| `transforms.py` | Homogeneous-transform helpers (NumPy only). |
| `forward.py` | Forward kinematics: joint angles → tool pose. Own DH product **and** a roboticstoolbox model builder. |
| `inverse.py` | Inverse kinematics: tool pose → joint angles (numerical LM solver; analytic spherical-wrist path documented for later). |
| `torque.py` | **Static** torque per joint → servo sizing (can it *hold* the pose?). |
| `dynamics.py` | **Dynamic** torque = gravity + inertia × peak acceleration (can it *move*?). |
| `viz.py` | 3D matplotlib plot of the arm at a configuration. |
| `gui.py` | Interactive control panel: joint sliders, live 3D view, tool-pose readout, live static/dynamic servo-sizing table. |
| `theme.py` | GUI styling tokens (ported from a separate design project; degrades gracefully if its fonts are absent). |
| `tests/` | FK determinism, FK↔IK round-trip, static + dynamic torque sanity. 10 tests, all passing. |

### Coordinate convention
Standard (proximal) **Denavit–Hartenberg**. The arm is a **6R with a spherical
wrist** (axes 4-5-6 intersect), which is what makes a clean closed-form IK
possible later.

> ⚠️ **The link lengths and masses in `model.py` are realistic PLACEHOLDERS**
> (~500 mm reach, STS3215-class servos), chosen so the whole pipeline runs today.
> They are **not measured hardware values yet** — see "Plan: still to define".

---

## Quick start

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1     Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt

python -m arm6dof.forward     # forward kinematics at the home pose
python -m arm6dof.torque      # static servo-sizing table
python -m arm6dof.dynamics    # dynamic (gravity + inertia) sizing table
python -m arm6dof.gui         # interactive control panel
pytest -q                     # tests (IK tests auto-skip without roboticstoolbox)
```

---

## Plan

### ✅ Defined / done
- Kinematic framework (FK, IK, transforms) with a spherical-wrist 6R model.
- Static **and** dynamic servo-sizing calculators driven by one geometry model.
- Interactive GUI showing live pass/fail per joint as you pose the arm.
- Test suite for the geometry-first layers.

### ⏳ Still to define (open decisions)
1. **Physical frame** — printed (SO-ARM / UR-style STL set) vs. metal-bracket
   kit. This fixes the real link lengths and link masses.
2. **Real geometry** — measure link lengths, masses and centres of mass; replace
   the placeholders in `model.py`. Every sizing number depends on this.
3. **Servo selection per joint** — the current STS3215 assumption FAILS the
   lower joints at 500 g. Decide per joint between: a bigger servo, an external
   **gear reduction** (`gear_ratio` in `model.py`), or a shorter reach.
4. **Wrist geometry** — confirm the last three axes truly intersect (spherical
   wrist) so the analytic IK is valid; otherwise stay on numerical IK.
5. **Controller** — Arduino + PCA9685 (PWM), ESP32 (wireless), or Raspberry Pi
   (Python/IK on-board). Serial-bus servos (STS3215) change this vs. PWM.
6. **Power budget** — size the supply for combined stall current (a real
   failure point on multi-servo arms).
7. **Gripper / end effector** — type, weight, and its contribution to payload.

### ▶️ Next actions (in order)
- [ ] Choose the frame (decision #1) and measure geometry → update `model.py`.
- [ ] Iterate `model.py` (servos + gear ratios) until **every joint PASSES the
      dynamic check** at 500 g and a realistic peak acceleration.
- [ ] Produce a concrete AliExpress **shopping list** that survives the dynamic
      sizing check (servos, driver board, power supply, frame/kit, gripper).
- [ ] Fill in `analytic_ik` (Pieper decomposition) for deterministic,
      multi-solution IK once wrist geometry is confirmed.
- [ ] Add a `hardware/` layer mapping IK joint angles → servo commands
      (STS3215: 0–4095 counts; PWM: microsecond pulses).
- [ ] Optional: add the Coriolis/velocity dynamics term via
      `roboticstoolbox.rne` once links carry real inertia tensors.

---

## Status of the analysis (with placeholder geometry)

At the horizontal worst-case pose, 500 g payload, ×2.5 safety factor:

| Joint | Static hold | Dynamic (8 rad/s²) | STS3215 (30 kgf·cm) |
|---|---|---|---|
| Base J1 | PASS | **FAIL** (rotational inertia) | insufficient |
| Shoulder J2 | **FAIL** | **FAIL** | needs gearing / bigger servo |
| Elbow J3 | **FAIL** | **FAIL** | needs gearing / bigger servo |
| Wrist J4–J6 | PASS | PASS | ok |

**Conclusion so far:** the lower three joints need more than a bare STS3215.
Resolving that (decision #3) is the immediate engineering task.
