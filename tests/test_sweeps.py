"""
Parameter sweeps that save plots to the sweep folder.

Each test draws a figure for you to look at (see ``--sweep-dir``) and also checks
that the curves have the expected shape, so a broken model fails here too.
Skipped automatically if matplotlib is not installed; deselect with ``-m "not sweep"``.
"""

import numpy as np
import pytest

from conftest import F_Z0, LOAD_SENSITIVITY, MU

pytestmark = pytest.mark.sweep

FORCE_UNIT = "lbf"
LOADS = [50.0, 100.0, 150.0, 200.0, 250.0]


def test_lateral_force_vs_slip_angle(r55a, plt, save_sweep):
    alpha = np.linspace(-30, 30, 601)
    fig, ax = plt.subplots(figsize=(9, 5))

    peaks = []
    for f_z in LOADS:
        fy = r55a.calculate_lateral_force(f_z, alpha, alpha_unit="deg")
        ax.plot(alpha, fy, linewidth=2, label=f"$F_z$ = {f_z:g} {FORCE_UNIT}")
        peaks.append(fy.max())

        assert fy == pytest.approx(-fy[::-1])  # odd in slip angle
        assert fy.max() == pytest.approx(r55a.calculate_d(f_z), rel=1e-3)

    ax.set(
        title="Lateral force vs. slip angle",
        xlabel="Slip angle α (deg)",
        ylabel=f"Lateral force $F_y$ ({FORCE_UNIT})",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    save_sweep(fig, "lateral_force_vs_slip_angle")

    assert peaks == sorted(peaks)  # more load, more peak force


def test_longitudinal_force_vs_slip_ratio(r55a, plt, save_sweep):
    kappa = np.linspace(-0.6, 0.6, 601)
    fig, ax = plt.subplots(figsize=(9, 5))

    for f_z in LOADS:
        _, fx = r55a.calculate_forces(f_z, 0.0, kappa, friction_circle=False)
        ax.plot(kappa, fx, linewidth=2, label=f"$F_z$ = {f_z:g} {FORCE_UNIT}")

        assert fx == pytest.approx(-fx[::-1])
        assert fx.max() == pytest.approx(r55a.calculate_d(f_z), rel=1e-3)

    ax.set(
        title="Longitudinal force vs. slip ratio",
        xlabel="Slip ratio κ (1 = 100 %)",
        ylabel=f"Longitudinal force $F_x$ ({FORCE_UNIT})",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    save_sweep(fig, "longitudinal_force_vs_slip_ratio")


def test_load_sensitivity(r55a, plt, save_sweep):
    loads = np.linspace(0, 600, 601)  # includes zero load
    peak_force = r55a.calculate_d(loads)
    with np.errstate(invalid="ignore", divide="ignore"):
        effective_mu = np.where(loads > 0, peak_force / np.where(loads > 0, loads, 1.0), np.nan)

    fig, (ax_force, ax_mu) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_force.plot(loads, peak_force, linewidth=2, label="model peak force")
    ax_force.plot(loads, MU * loads, "--", color="gray", label="constant μ (no load sensitivity)")
    ax_force.axvline(F_Z0, color="black", linewidth=0.8, linestyle=":")
    ax_force.set(
        title="Peak force vs. load",
        xlabel=f"Vertical load $F_z$ ({FORCE_UNIT})",
        ylabel=f"Peak force ({FORCE_UNIT})",
    )
    ax_force.legend()
    ax_force.grid(True, linestyle="--", alpha=0.5)

    ax_mu.plot(loads, effective_mu, linewidth=2, color="crimson")
    ax_mu.axvline(F_Z0, color="black", linewidth=0.8, linestyle=":", label="reference load")
    ax_mu.set(
        title="Effective friction coefficient vs. load",
        xlabel=f"Vertical load $F_z$ ({FORCE_UNIT})",
        ylabel="Peak force / load",
    )
    ax_mu.legend()
    ax_mu.grid(True, linestyle="--", alpha=0.5)
    save_sweep(fig, "load_sensitivity")

    assert peak_force[0] == 0.0
    assert np.all(np.diff(peak_force) > 0)  # more load, more force
    assert np.all(np.diff(effective_mu[1:]) < 0)  # but less grip per unit load
    assert effective_mu[int(F_Z0)] == pytest.approx(MU)
    assert peak_force[600] / peak_force[300] == pytest.approx(2.0**LOAD_SENSITIVITY)


def test_friction_circle_diagram(r55a, plt, save_sweep):
    f_z = 150.0
    radius = r55a.calculate_d(f_z)
    kappa = np.linspace(-0.6, 0.6, 400)
    alphas_deg = [2, 5, 8, 12, 20]

    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="black", linewidth=1.5)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(alphas_deg)))
    outside = False
    for alpha_deg, color in zip(alphas_deg, colors):
        raw_fy, raw_fx = r55a.calculate_forces(
            f_z, alpha_deg, kappa, friction_circle=False, alpha_unit="deg"
        )
        fy, fx = r55a.calculate_forces(f_z, alpha_deg, kappa, friction_circle=True, alpha_unit="deg")
        raw_fy = np.broadcast_to(raw_fy, raw_fx.shape)  # no circle: forces keep their own shapes
        ax.plot(raw_fx, raw_fy, "--", color=color, alpha=0.45)
        ax.plot(fx, fy, color=color, linewidth=2, label=f"α = {alpha_deg}°")

        assert np.all(np.hypot(fx, fy) <= radius * (1 + 1e-12))
        outside = outside or np.any(np.hypot(raw_fx, raw_fy) > radius * 1.01)

    ax.set(
        title=f"Friction circle at $F_z$ = {f_z:g} {FORCE_UNIT}\n(dashed = without the circle)",
        xlabel=f"Longitudinal force $F_x$ ({FORCE_UNIT})",
        ylabel=f"Lateral force $F_y$ ({FORCE_UNIT})",
        aspect="equal",
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")
    save_sweep(fig, "friction_circle_diagram")

    assert outside  # the unclipped curves really do leave the circle


def test_lateral_force_load_grid(r55a, plt, save_sweep):
    loads = np.linspace(0, 500, 101)[:, None]  # column of loads
    alpha = np.linspace(-30, 30, 241)  # row of slip angles
    grid = r55a.calculate_lateral_force(loads, alpha, alpha_unit="deg")

    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.contourf(alpha, loads[:, 0], grid, levels=25, cmap="RdBu_r")
    fig.colorbar(mesh, label=f"Lateral force $F_y$ ({FORCE_UNIT})")
    ax.set(
        title="Lateral force over load and slip angle",
        xlabel="Slip angle α (deg)",
        ylabel=f"Vertical load $F_z$ ({FORCE_UNIT})",
    )
    save_sweep(fig, "lateral_force_load_grid")

    assert grid.shape == (101, 241)
    assert np.all(grid[0] == 0.0)  # no load, no force
    ten_degrees = grid[:, np.argmin(np.abs(alpha - 10.0))]
    assert np.all(np.diff(ten_degrees) > 0)  # at a fixed angle, more load gives more force
