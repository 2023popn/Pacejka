# Pacejka

Calculate the forces a tire generates with the Pacejka "magic formula".

Describe a tire by its peak friction, cornering and longitudinal stiffness, and how its grip changes with load. Then ask for the lateral and longitudinal force at any vertical load, slip angle and slip ratio. Everything works on single numbers or on whole numpy arrays, so a full slip-angle sweep or a load-versus-slip grid is one call.

## Installation

```bash
pip install pacejka
```

The only dependency is numpy. To run the plotting tests too, install the `test` extra (see [Development](#development)).

The package is installed as `pacejka` and imported as `Pacejka`:

```python
from Pacejka import Tire
```

## Quick start

```python
from Pacejka import Tire

tire = Tire.from_macro_properties(
    "Hoosier R55A",
    mu_base=1.6,                      # peak friction coefficient at the reference load
    cornering_stiffness_per_deg=25.0, # lateral force per degree of slip angle (small angles)
    longitudinal_stiffness=2000.0,    # longitudinal force per unit slip ratio (small slip)
    load_sensitivity_n=0.9,           # how much grip per unit load falls as load rises
    f_z0=150.0,                       # reference load the stiffnesses are quoted at
)

fy, fx = tire.calculate_forces(f_z=150.0, alpha=4.0, kappa=0.05, alpha_unit="deg")
print(f"lateral = {fy:.1f}, longitudinal = {fx:.1f}")
```

```
lateral = 94.8, longitudinal = 94.3
```

`calculate_forces` returns `(lateral_force, longitudinal_force)`. Forces come out in whatever unit you use for load, so pounds in gives pounds-force out and newtons in gives newtons out.

If you only need the lateral force:

```python
fy = tire.calculate_lateral_force(150.0, 4.0, alpha_unit="deg")
```

## Units

| Input | Unit |
|---|---|
| `f_z` (vertical load) | any force unit; forces come back in the same unit |
| `alpha` (slip angle) | radians or degrees, see below |
| `kappa` (slip ratio) | dimensionless, `1.0` is 100 % slip |
| `cornering_stiffness_per_deg` / `_per_rad` | force per degree / radian of slip angle |
| `longitudinal_stiffness` | force per unit slip ratio |

### Slip angle: radians or degrees

State the unit of `alpha` explicitly with `alpha_unit="rad"` or `alpha_unit="deg"`. This is the safest option.

If you leave `alpha_unit` out, the unit is detected: the angle is taken to be radians unless any value has a magnitude above π/2 (about 1.57), in which case the **whole** input is treated as degrees. A sweep from −15 to 15 is therefore read correctly as degrees, but small angles in degrees are not (an angle of `0.5` looks like radians). When in doubt, pass `alpha_unit`.

```python
tire.calculate_lateral_force(150.0, 4.0)                       # detected: 4.0 is degrees
tire.calculate_lateral_force(150.0, 0.5)                       # detected: 0.5 is radians -> 239.8
tire.calculate_lateral_force(150.0, 0.5, alpha_unit="deg")     # explicitly degrees       -> 12.5
```

### Cornering stiffness: per degree or per radian

`from_macro_properties` takes either `cornering_stiffness_per_deg` or `cornering_stiffness_per_rad` (exactly one). The `Tire` constructor itself works in radians only.

## Sweeps and grids with arrays

Pass arrays for any of `f_z`, `alpha` and `kappa`. numpy broadcasting combines them.

A slip-angle sweep at one load:

```python
import numpy as np

alpha = np.linspace(-15, 15, 7)
fy = tire.calculate_lateral_force(150.0, alpha, alpha_unit="deg")
print(np.round(fy, 1))
```

```
[-223.1 -188.7 -115.2    0.   115.2  188.7  223.1]
```

A grid with one row per load and one column per slip angle:

```python
loads = np.array([[100.0], [150.0], [200.0]])   # a column
alpha = np.array([2.0, 5.0, 10.0])               # a row
grid = tire.calculate_lateral_force(loads, alpha, alpha_unit="deg")
print(grid.shape)
print(np.round(grid, 1))
```

```
(3, 3)
[[ 34.2  80.  131. ]
 [ 49.3 115.2 188.7]
 [ 63.9 149.2 244.5]]
```

Plotting a sweep with matplotlib:

```python
import matplotlib.pyplot as plt

alpha = np.linspace(-30, 30, 400)
for f_z in (50, 100, 150, 200, 250):
    plt.plot(alpha, tire.calculate_lateral_force(f_z, alpha, alpha_unit="deg"), label=f"{f_z}")
plt.xlabel("Slip angle (deg)")
plt.ylabel("Lateral force")
plt.legend()
plt.show()
```

A scalar in gives a scalar out. Lists work as well as arrays.

> When `friction_circle=False`, each force has the shape of its own inputs (lateral from `f_z` and `alpha`, longitudinal from `f_z` and `kappa`). With the friction circle on (the default), both forces have the full broadcast shape.

## The friction circle

By default, `calculate_forces` limits the combined force so it cannot exceed the tire's peak force at that load. If the lateral and longitudinal forces together are too large, both are scaled down by the same factor, which keeps their direction.

```python
for circle in (False, True):
    fy, fx = tire.calculate_forces(150.0, 8.0, 0.3, friction_circle=circle, alpha_unit="deg")
    print(f"friction_circle={circle}: fy={fy:.1f} fx={fx:.1f} total={np.hypot(fx, fy):.1f}")
```

```
friction_circle=False: fy=164.9 fx=238.5 total=290.0
friction_circle=True: fy=136.5 fx=197.4 total=240.0
```

The limit here is 240, which is the peak force at this load (`tire.calculate_d(150.0)`). Pass `friction_circle=False` for the pure, independent lateral and longitudinal forces.

## Building a tire

**From measurable properties** (most common). B is derived from your stiffnesses at each load, and grip changes with load:

```python
Tire.from_macro_properties("name", mu_base=1.6, cornering_stiffness_per_deg=25.0,
                           longitudinal_stiffness=2000.0, load_sensitivity_n=0.9, f_z0=150.0)
```

`load_sensitivity_n` of `1.0` means grip per unit load is constant. Lower values mean it falls as load increases (`0.9` is roughly a 10 % drop per doubling of load). `f_z0` is the load at which the stiffnesses and `mu_base` apply.

**From fixed magic formula constants.** You supply B, C and E directly, and there is no load sensitivity (peak force is `mu_base * f_z`). Lateral values default to the longitudinal ones if you leave them out:

```python
fixed_tire = Tire.from_constants("Fixed", mu_base=1.5, b_x=8.0, c_x=1.4, e_x=0.1)
fy, fx = fixed_tire.calculate_forces(200.0, 0.1, 0.05, alpha_unit="rad")
```

**Directly** with `Tire(...)`, which takes cornering stiffness per radian and the other values by keyword.

## Looking at the coefficients

`coefficients_at(f_z)` returns the full set of magic formula coefficients for each axis at a given load, as `(longitudinal, lateral)`:

```python
longitudinal, lateral = tire.coefficients_at(150.0)
print(lateral)
```

```
MagicFormulaCoefficients(b=4.263078832818626, c=1.4, d=240.0, e=-0.1)
```

These are immutable and freshly built on every call. A `Tire` never changes when you ask it for forces, so one tire can safely be shared between threads. The fixed settings that belong to the tire are on `tire.longitudinal` and `tire.lateral`, for example `AxisShape(c=1.4, e=-0.1, b=None)`.

## How it works

For each axis, with `x` the slip angle in radians (lateral) or the slip ratio (longitudinal):

```
F = D * sin( C * arctan( B*x - E * (B*x - arctan(B*x)) ) )

D = mu_peak * (Fz / Fz0) ** (n - 1) * Fz          peak force at load Fz
B = stiffness * (Fz / Fz0) ** n / (C * D)         from the small-slip stiffness
```

`C` (shape) and `E` (curvature) default to 1.4 and 0.1 (longitudinal) or -0.1 (lateral). D is the same for both axes. A load of zero or less produces no force.

## Limitations

- These are the pure-slip magic formulas combined with a friction circle, not the full combined-slip Pacejka formulation. Lateral force does not depend on slip ratio except through the friction circle limit.
- No camber, inflation pressure, speed, temperature or aligning moment.
- The effective friction coefficient follows a power law in load, so it grows without bound as load approaches zero. Peak force still goes to zero.
- Tire properties are your inputs. The package does not ship measured tire data, so check your coefficients against your own data.

## Development

```bash
git clone https://github.com/2023popn/Pacejka
cd Pacejka
pip install -e ".[test]"
pytest
```

The tests include **sweeps**: plots of lateral force against slip angle, longitudinal force against slip ratio, load sensitivity, the friction circle, and a load-versus-slip grid. They are saved as PNG files in `sweep_output/` (or the folder given by `--sweep-dir`) so you can look at the curves, and each also checks the shape of what it plots.

```bash
pytest --sweep-dir=plots      # save the sweep plots somewhere else
pytest -m "not sweep"         # skip the plotting tests
```

The sweep tests need matplotlib and are skipped if it is not installed.

## License

MIT. See `LICENSE`.
