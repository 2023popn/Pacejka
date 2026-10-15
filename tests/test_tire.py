"""Behavior tests for the Tire model."""

import copy
import dataclasses
import threading
import warnings

import numpy as np
import pytest

from pacejka import AxisShape, MagicFormulaCoefficients, Tire

from conftest import CORNERING_PER_DEG, F_Z0, LOAD_SENSITIVITY, LONGITUDINAL, MU

DEG_PER_RAD = 180.0 / np.pi
CORNERING_PER_RAD = CORNERING_PER_DEG * DEG_PER_RAD


def expected_d(f_z):
    """Peak force from the load sensitivity formula, written out independently."""
    return MU * (f_z / F_Z0) ** (LOAD_SENSITIVITY - 1) * f_z


def magic_formula(x, b, c, d, e):
    """The magic formula, written out independently of the library."""
    bx = b * x
    return d * np.sin(c * np.arctan(bx - e * (bx - np.arctan(bx))))


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_per_degree_and_per_radian_stiffness_agree():
    by_deg = Tire.from_macro_properties(
        "a", 1.6, cornering_stiffness_per_deg=25.0, longitudinal_stiffness=10.0
    )
    by_rad = Tire.from_macro_properties(
        "a", 1.6, cornering_stiffness_per_rad=25.0 * DEG_PER_RAD, longitudinal_stiffness=10.0
    )
    assert by_deg.calculate_forces(180, 0.1, 0.05) == pytest.approx(
        by_rad.calculate_forces(180, 0.1, 0.05), rel=1e-12
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            dict(
                cornering_stiffness_per_deg=25,
                cornering_stiffness_per_rad=1432,
                longitudinal_stiffness=10,
            ),
            "not both",
        ),
        (dict(longitudinal_stiffness=10), "per degree or per radian"),
        (dict(cornering_stiffness_per_rad=1432), "longitudinal_stiffness"),
    ],
)
def test_from_macro_properties_validates_its_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Tire.from_macro_properties("a", 1.6, **kwargs)


def test_constructor_only_accepts_radians():
    with pytest.raises(TypeError):
        Tire("a", 1.6, cornering_stiffness_per_deg=25.0)


def test_from_constants_lateral_values_default_to_longitudinal():
    tire = Tire.from_constants("c", 1.5, b_x=8.0, c_x=1.4, e_x=0.1)
    assert tire.lateral == AxisShape(c=1.4, e=0.1, b=8.0)
    assert tire.longitudinal == AxisShape(c=1.4, e=0.1, b=8.0)


# ----------------------------------------------------------------------
# Coefficients
# ----------------------------------------------------------------------


@pytest.mark.parametrize("f_z", [50.0, 150.0, 300.0, 600.0])
def test_peak_factor_follows_load_sensitivity(r55a, f_z):
    assert r55a.calculate_d(f_z) == pytest.approx(expected_d(f_z), rel=1e-12)


def test_peak_force_at_reference_load_is_mu_times_load(r55a):
    assert r55a.calculate_d(F_Z0) == pytest.approx(MU * F_Z0)


def test_doubling_load_scales_peak_force_by_two_to_the_n(r55a):
    ratio = r55a.calculate_d(400.0) / r55a.calculate_d(200.0)
    assert ratio == pytest.approx(2.0**LOAD_SENSITIVITY)


@pytest.mark.parametrize("f_z", [60.0, 150.0, 400.0])
def test_small_slip_slope_equals_requested_stiffness(r55a, f_z):
    scaling = (f_z / F_Z0) ** LOAD_SENSITIVITY
    slip = 1e-6

    lateral_slope = r55a.calculate_lateral_force(f_z, slip, alpha_unit="rad") / slip
    assert lateral_slope == pytest.approx(CORNERING_PER_RAD * scaling, rel=1e-6)

    _, longitudinal = r55a.calculate_forces(f_z, 0.0, slip, friction_circle=False)
    assert longitudinal / slip == pytest.approx(LONGITUDINAL * scaling, rel=1e-6)


def test_lateral_force_matches_independent_formula(r55a):
    f_z, alpha_deg = 150.0, 5.0
    d = expected_d(f_z)
    b = CORNERING_PER_RAD / (1.4 * d)
    expected = magic_formula(np.radians(alpha_deg), b, c=1.4, d=d, e=-0.1)
    assert r55a.calculate_lateral_force(f_z, alpha_deg, alpha_unit="deg") == pytest.approx(
        expected, rel=1e-12
    )


def test_from_constants_has_constant_friction_and_given_b(constant_tire):
    for f_z in (50.0, 200.0, 500.0):
        longitudinal, lateral = constant_tire.coefficients_at(f_z)
        assert longitudinal.d == pytest.approx(1.5 * f_z)
        assert lateral.d == pytest.approx(1.5 * f_z)
        assert longitudinal.b == 8.0
        assert lateral.b == 9.0


# ----------------------------------------------------------------------
# Shape of the curves
# ----------------------------------------------------------------------


def test_forces_are_odd_functions_of_slip(r55a):
    slip = np.linspace(0.01, 0.5, 40)
    assert r55a.calculate_lateral_force(150, -slip, alpha_unit="rad") == pytest.approx(
        -r55a.calculate_lateral_force(150, slip, alpha_unit="rad")
    )
    _, fx_neg = r55a.calculate_forces(150, 0.0, -slip, friction_circle=False)
    _, fx_pos = r55a.calculate_forces(150, 0.0, slip, friction_circle=False)
    assert fx_neg == pytest.approx(-fx_pos)


def test_force_never_exceeds_peak_and_reaches_it(r55a):
    alpha = np.linspace(0.0, 1.5, 3000)
    fy = r55a.calculate_lateral_force(150, alpha, alpha_unit="rad")
    d = expected_d(150.0)
    assert fy.max() <= d * (1 + 1e-12)
    assert fy.max() == pytest.approx(d, rel=1e-4)


# ----------------------------------------------------------------------
# Friction circle
# ----------------------------------------------------------------------


def combined_sweep(tire, loads, friction_circle):
    alpha = np.linspace(-0.5, 0.5, 41)[None, :, None]
    kappa = np.linspace(-0.5, 0.5, 41)[None, None, :]
    f_z = np.asarray(loads, dtype=float)[:, None, None]
    fy, fx = tire.calculate_forces(f_z, alpha, kappa, friction_circle, alpha_unit="rad")
    return np.hypot(fx, fy), f_z


def test_friction_circle_limits_combined_force_to_peak(r55a):
    magnitude, f_z = combined_sweep(r55a, [50.0, 150.0, 300.0], friction_circle=True)
    assert np.all(magnitude <= expected_d(f_z) * (1 + 1e-12))


def test_without_the_circle_combined_force_can_exceed_peak(r55a):
    magnitude, f_z = combined_sweep(r55a, [50.0, 150.0, 300.0], friction_circle=False)
    assert np.any(magnitude > expected_d(f_z) * 1.01)


def test_friction_circle_keeps_force_direction(r55a):
    fy_raw, fx_raw = r55a.calculate_forces(150, 0.4, 0.4, friction_circle=False, alpha_unit="rad")
    fy, fx = r55a.calculate_forces(150, 0.4, 0.4, friction_circle=True, alpha_unit="rad")
    assert np.hypot(fx_raw, fy_raw) > expected_d(150.0)  # the circle really did clip
    assert fx / fy == pytest.approx(fx_raw / fy_raw)


def test_friction_circle_leaves_small_forces_alone(r55a):
    with_circle = r55a.calculate_forces(150, 0.01, 0.01, friction_circle=True, alpha_unit="rad")
    without = r55a.calculate_forces(150, 0.01, 0.01, friction_circle=False, alpha_unit="rad")
    assert with_circle == pytest.approx(without, rel=1e-12)


def test_friction_circle_uses_load_sensitive_peak(r55a):
    # Direct call: demand far beyond the limit gets scaled onto the circle edge
    fx, fy = r55a.friction_circle(1e6, 0.0, 300.0)
    assert fx == pytest.approx(expected_d(300.0))
    assert fy == 0.0


# ----------------------------------------------------------------------
# Zero and negative load
# ----------------------------------------------------------------------


@pytest.mark.parametrize("f_z", [0, 0.0, -10.0, -1e-9, np.float64(-10.0)])
@pytest.mark.parametrize("friction_circle", [True, False])
def test_unloaded_tire_makes_no_force_and_no_warnings(r55a, constant_tire, f_z, friction_circle):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for tire in (r55a, constant_tire):
            assert tire.calculate_forces(f_z, 0.1, 0.1, friction_circle) == (0.0, 0.0)
            assert tire.calculate_lateral_force(f_z, 0.1) == 0.0
            assert tire.calculate_d(f_z) == 0.0


def test_unloaded_points_inside_an_array_make_no_force(r55a):
    loads = np.array([-5.0, 0.0, 100.0, 200.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fy, fx = r55a.calculate_forces(loads, 0.1, 0.1, alpha_unit="rad")
    assert fy[0] == fy[1] == fx[0] == fx[1] == 0.0
    assert fy[2] > 0 and fy[3] > fy[2]


# ----------------------------------------------------------------------
# Arrays and broadcasting
# ----------------------------------------------------------------------


@pytest.mark.parametrize("tire_name", ["r55a", "constant_tire"])
def test_array_call_matches_a_loop_of_scalar_calls(request, tire_name):
    tire = request.getfixturevalue(tire_name)
    f_z = np.array([-5.0, 0.0, 40.0, 150.0, 400.0])
    alpha = np.linspace(-0.4, 0.4, 5)
    kappa = np.linspace(-0.5, 0.5, 5)

    fy, fx = tire.calculate_forces(f_z, alpha, kappa, alpha_unit="rad")
    expected = np.array(
        [
            tire.calculate_forces(float(z), float(a), float(k), alpha_unit="rad")
            for z, a, k in zip(f_z, alpha, kappa)
        ]
    )
    assert fy == pytest.approx(expected[:, 0], rel=1e-12, abs=0)
    assert fx == pytest.approx(expected[:, 1], rel=1e-12, abs=0)


def test_loads_and_slip_angles_broadcast_into_a_grid(r55a):
    loads = np.array([[100.0], [200.0], [300.0]])
    alpha = np.linspace(-0.3, 0.3, 7)

    grid = r55a.calculate_lateral_force(loads, alpha, alpha_unit="rad")
    assert grid.shape == (3, 7)
    assert grid[1, 5] == pytest.approx(
        r55a.calculate_lateral_force(200.0, alpha[5], alpha_unit="rad")
    )

    fy, fx = r55a.calculate_forces(loads, alpha, 0.1, alpha_unit="rad")
    assert fy.shape == fx.shape == (3, 7)


def test_friction_circle_gives_both_forces_the_full_broadcast_shape(r55a):
    loads = np.array([[100.0], [200.0]])
    fy, fx = r55a.calculate_forces(loads, np.linspace(-0.3, 0.3, 7), np.linspace(-0.2, 0.2, 7))
    assert fy.shape == fx.shape == (2, 7)


def test_scalar_input_gives_scalar_output(r55a):
    assert np.ndim(r55a.calculate_lateral_force(150, 0.1)) == 0
    assert all(np.ndim(v) == 0 for v in r55a.calculate_forces(150, 0.1, 0.1))
    assert np.ndim(r55a.calculate_d(150)) == 0
    assert all(np.ndim(c.d) == 0 for c in r55a.coefficients_at(150))


def test_lists_and_zero_d_arrays_are_accepted(r55a):
    expected = r55a.calculate_forces(150, np.array([0.1, 0.2]), np.array([0.05, 0.06]))
    from_lists = r55a.calculate_forces(150, [0.1, 0.2], [0.05, 0.06])
    for got, want in zip(from_lists, expected):
        assert got == pytest.approx(want)
    assert r55a.calculate_forces(np.array(150.0), 0.1, 0.05) == pytest.approx(
        r55a.calculate_forces(150.0, 0.1, 0.05)
    )


def test_mismatched_shapes_raise(r55a):
    with pytest.raises(ValueError):
        r55a.calculate_forces(np.ones(3), np.ones(4), 0.1)


def test_empty_arrays_give_empty_results(r55a):
    assert r55a.calculate_lateral_force(150, np.array([])).shape == (0,)


# ----------------------------------------------------------------------
# Slip angle units
# ----------------------------------------------------------------------


def test_explicit_degrees_match_radians(r55a):
    alpha_deg = np.linspace(-10, 10, 21)
    assert r55a.calculate_lateral_force(150, alpha_deg, alpha_unit="deg") == pytest.approx(
        r55a.calculate_lateral_force(150, np.radians(alpha_deg), alpha_unit="rad")
    )


def test_unit_is_detected_from_the_whole_sweep(r55a):
    degrees = np.linspace(-15, 15, 31)  # most values look like radians, a few do not
    assert r55a.calculate_lateral_force(150, degrees) == pytest.approx(
        r55a.calculate_lateral_force(150, degrees, alpha_unit="deg")
    )
    radians = np.linspace(-0.3, 0.3, 31)
    assert r55a.calculate_lateral_force(150, radians) == pytest.approx(
        r55a.calculate_lateral_force(150, radians, alpha_unit="rad")
    )


def test_one_large_value_converts_the_whole_array(r55a):
    mixed = np.array([0.1, 20.0])
    assert r55a.calculate_lateral_force(150, mixed) == pytest.approx(
        r55a.calculate_lateral_force(150, mixed, alpha_unit="deg")
    )


def test_small_angles_in_degrees_need_the_explicit_unit(r55a):
    # 0.5 looks like radians to the detector, so degrees must be requested
    detected = r55a.calculate_lateral_force(150, 0.5)
    explicit = r55a.calculate_lateral_force(150, 0.5, alpha_unit="deg")
    assert detected == pytest.approx(r55a.calculate_lateral_force(150, 0.5, alpha_unit="rad"))
    assert explicit < detected


@pytest.mark.parametrize("bad_unit", ["degrees", "DEG", "radians", ""])
def test_invalid_unit_is_rejected(r55a, bad_unit):
    with pytest.raises(ValueError, match="alpha_unit"):
        r55a.calculate_lateral_force(150, 0.1, alpha_unit=bad_unit)
    with pytest.raises(ValueError, match="alpha_unit"):
        r55a.calculate_forces(150, 0.1, 0.1, alpha_unit=bad_unit)


# ----------------------------------------------------------------------
# State and immutability
# ----------------------------------------------------------------------


def test_force_calls_do_not_change_the_tire(r55a):
    before = copy.deepcopy(vars(r55a))
    loads = np.array([[50.0], [150.0]])
    r55a.calculate_forces(loads, np.linspace(-0.3, 0.3, 5), 0.1)
    r55a.calculate_lateral_force(150, 0.1)
    r55a.coefficients_at(loads)
    assert vars(r55a) == before


def test_coefficient_objects_are_frozen(r55a):
    longitudinal, _ = r55a.coefficients_at(150)
    assert isinstance(longitudinal, MagicFormulaCoefficients)
    with pytest.raises(dataclasses.FrozenInstanceError):
        longitudinal.d = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        r55a.lateral.c = 1.0


def test_coefficients_at_reports_fixed_and_load_dependent_values(r55a):
    longitudinal, lateral = r55a.coefficients_at(150.0)
    assert (longitudinal.c, longitudinal.e) == (1.4, 0.1)
    assert (lateral.c, lateral.e) == (1.4, -0.1)
    assert longitudinal.d == lateral.d == pytest.approx(MU * F_Z0)
    assert lateral.b == pytest.approx(CORNERING_PER_RAD / (1.4 * MU * F_Z0))


def test_one_tire_can_be_shared_between_threads(r55a):
    loads = (60.0, 400.0)
    expected = {
        z: r55a.calculate_forces(z, 0.1, 0.1, friction_circle=False, alpha_unit="rad")
        for z in loads
    }
    wrong = []

    def worker(f_z):
        for _ in range(1500):
            result = r55a.calculate_forces(f_z, 0.1, 0.1, friction_circle=False, alpha_unit="rad")
            if result != expected[f_z]:
                wrong.append(f_z)

    threads = [threading.Thread(target=worker, args=(z,)) for z in loads for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not wrong
