# arm6dof

Kinematics and servo-sizing framework for a **6-axis servo robotic arm**
targeting a **500 g tip payload**. Geometry-first and hardware-agnostic: the
whole pipeline (forward kinematics, inverse kinematics, static torque / servo
sizing) runs and is testable in simulation *before any parts are bought*.

## Why this exists first

You cannot pick the AliExpress servos until you know the torque each joint must
hold, and you cannot know that until you have the arm's geometry as a formal
model. So the order is: **model -> forward kinematics -> torque sizing -> buy
servos -> inverse kinematics -> control**. This repo builds the software layers;
`model.py` is the one file you edit as the real hardware firms up.

## Layers

| File | Role |
|------|------|
| `arm6dof/model.py` | **Single source of truth.** DH table, link masses, joint limits, payload, tool offset. Edit this to match real hardware. |
| `arm6dof/transforms.py` | Homogeneous-transform helpers (NumPy only). |
| `arm6dof/forward.py` | Forward kinematics: joint angles -> tool pose. Own DH product **and** a roboticstoolbox model builder. |
| `arm6dof/inverse.py` | Inverse kinematics: tool pose -> joint angles (numerical LM solver; analytic spherical-wrist path documented for later). |
| `arm6dof/torque.py` | **Static torque per joint -> servo sizing.** Decides which servos you buy. |
| `arm6dof/viz.py` | 3D plot of the arm at a configuration (matplotlib). |
| `tests/` | FK determinism, FK<->IK round-trip, torque sanity. |

## Coordinate convention

Standard (proximal) Denavit-Hartenberg. Each joint = `(a, alpha, d, theta)`.
The arm is a **6R with a spherical wrist** (axes 4,5,6 intersect), which is what
makes a clean closed-form IK possible later.

> The link lengths and masses in `model.py` today are **realistic placeholders**
> (~500 mm reach, STS3215-class servos). They make the pipeline runnable now.
> Replace each with a measured value once the frame is chosen.

## Setup (Windows PowerShell)

```powershell
cd "$env:OneDrive\Desktop\1 - projects\arm6dof"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
# forward kinematics at the home pose
python -m arm6dof.forward

# servo sizing at the worst-case (arm horizontal) pose -- the buy/no-buy table
python -m arm6dof.torque

# IK round-trip demo (needs roboticstoolbox)
python -m arm6dof.inverse

# tests (IK tests auto-skip if roboticstoolbox isn't installed)
pytest -q
```

## The servo-sizing question

`python -m arm6dof.torque` prints, for the arm stretched horizontal with the
payload at the tip, how much torque each joint must hold vs. its servo's rated
stall torque, with a 2.5x safety factor. A `FAIL` row means that joint needs a
stronger servo or an external gear reduction (`gear_ratio` in `model.py`), or a
shorter link. This is the table that turns into a shopping list.

## Next steps

1. Choose the physical frame (printed SO-ARM-class vs metal kit) and measure
   real link lengths + masses; update `model.py`.
2. Re-run `torque.py` and iterate servo choice until every joint is `PASS`.
3. Fill in `analytic_ik` for deterministic multi-solution IK (optional).
4. Add a `hardware/` layer mapping IK joint angles to servo commands
   (STS3215: 0-4095 counts; PWM: microsecond pulses).
