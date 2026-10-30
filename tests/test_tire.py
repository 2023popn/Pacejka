"""Behavior tests for the Tire model."""

import copy
import dataclasses
import threading
import warnings

import numpy as np
import pytest

from pacejka import (
    AxisShape,
    MagicFormulaCoefficients,
    PressureParams,
    PressureWithoutThermalWarning,
    ThermalParams,
    Tire,
    combine_factors,
    pressure_factors,
    thermal_factors,
)

from conftest import (
    CORNERING_PER_DEG,
    F_Z0,
    KART_CORNERING_PER_DEG,
    KART_F_Z0,
    KART_MU,
    LOAD_SENSITIVITY,
    LONGITUDINAL,
    MU,
    make_kart_tire,
)

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


# ----------------------------------------------------------------------
# Temperature and pressure
# ----------------------------------------------------------------------

KART_CORNERING_PER_RAD = KART_CORNERING_PER_DEG * DEG_PER_RAD

# Thermal and pressure settings whose initial conditions are the reference
# conditions, so a tire built with them starts out identical to a plain tire.
THERMAL_AT_REF = ThermalParams(initial_temperature_k=ThermalParams().reference_temperature_k)
PRESSURE_AT_REF = PressureParams(fill_pressure_pa=PressureParams().reference_pressure_pa)

LOADS = np.array([[150.0], [500.0], [900.0]])
SLIP_ANGLES = np.linspace(-0.3, 0.3, 13)
SLIP_RATIOS = np.linspace(-0.2, 0.2, 5)


def forces_grid(tire):
    return tire.calculate_forces(LOADS[:, :, None], SLIP_ANGLES[None, :, None], SLIP_RATIOS)


def small_slip_stiffness(tire, f_z=KART_F_Z0):
    """Lateral force per radian at a vanishingly small slip angle."""
    return tire.calculate_lateral_force(f_z, 1e-7, alpha_unit="rad") / 1e-7


def test_a_tire_without_state_reports_none_and_ignores_updates(kart_base):
    assert kart_base.temperature is None
    assert kart_base.pressure is None
    before = forces_grid(kart_base)
    kart_base.update(1.0, 500.0, 0.1, 0.05, 20.0)
    kart_base.reset()
    assert np.array_equal(forces_grid(kart_base), before)


def test_a_new_tire_starts_at_the_initial_conditions(kart_tire):
    assert kart_tire.temperature == ThermalParams().initial_temperature_k
    assert kart_tire.pressure == PressureParams().fill_pressure_pa


@pytest.mark.parametrize("builder", ["macro", "constants"])
def test_a_tire_at_its_reference_state_matches_a_plain_tire(builder):
    if builder == "macro":
        plain = make_kart_tire()
        modeled = make_kart_tire(THERMAL_AT_REF, PRESSURE_AT_REF)
    else:
        kwargs = dict(mu_base=1.5, b_x=8.0, c_x=1.4, e_x=0.1, b_y=9.0, c_y=1.3, e_y=-0.2)
        plain = Tire.from_constants("c", **kwargs)
        modeled = Tire.from_constants("c", thermal=THERMAL_AT_REF, pressure=PRESSURE_AT_REF, **kwargs)
    for plain_force, modeled_force in zip(forces_grid(plain), forces_grid(modeled)):
        assert modeled_force == pytest.approx(plain_force, rel=1e-12, abs=1e-12)


def test_a_cold_tire_has_less_peak_force_by_exactly_the_grip_factor(kart_base):
    thermal = ThermalParams()
    cold = make_kart_tire(thermal)  # starts at the (cold) initial temperature
    grip = thermal_factors(thermal.initial_temperature_k, thermal).grip
    assert grip < 1.0
    for f_z in (200.0, KART_F_Z0, 800.0):
        assert cold.calculate_d(f_z) == pytest.approx(kart_base.calculate_d(f_z) * grip)


def test_grip_and_stiffness_factors_act_independently_on_a_macro_tire():
    thermal = ThermalParams()
    tire = make_kart_tire(thermal)
    factors = thermal_factors(thermal.initial_temperature_k, thermal)
    assert factors.grip != pytest.approx(factors.cornering_stiffness)

    assert tire.calculate_d(KART_F_Z0) == pytest.approx(KART_MU * KART_F_Z0 * factors.grip)
    assert small_slip_stiffness(tire) == pytest.approx(
        KART_CORNERING_PER_RAD * factors.cornering_stiffness, rel=1e-5
    )


def test_grip_and_stiffness_factors_act_independently_on_a_constant_b_tire():
    thermal = ThermalParams()
    plain = Tire.from_constants("c", mu_base=1.5, b_x=8.0, c_x=1.4, e_x=0.1)
    tire = Tire.from_constants("c", mu_base=1.5, b_x=8.0, c_x=1.4, e_x=0.1, thermal=thermal)
    factors = thermal_factors(thermal.initial_temperature_k, thermal)

    assert tire.calculate_d(400.0) == pytest.approx(plain.calculate_d(400.0) * factors.grip)
    assert small_slip_stiffness(tire, 400.0) == pytest.approx(
        small_slip_stiffness(plain, 400.0) * factors.cornering_stiffness, rel=1e-5
    )


def test_pressure_changes_the_forces(kart_base):
    with pytest.warns(PressureWithoutThermalWarning):
        high = make_kart_tire(
            pressure=PressureParams(fill_pressure_pa=140_000.0, reference_pressure_pa=80_000.0)
        )
    factors = pressure_factors(140_000.0, PressureParams())
    assert factors.cornering_stiffness > 1.0
    assert small_slip_stiffness(high) > small_slip_stiffness(kart_base)
    assert high.calculate_d(KART_F_Z0) == pytest.approx(
        kart_base.calculate_d(KART_F_Z0) * factors.grip
    )


def test_force_queries_never_change_the_state(kart_tire):
    before = (kart_tire.temperature, kart_tire.pressure)
    forces_grid(kart_tire)
    kart_tire.calculate_lateral_force(KART_F_Z0, 0.1)
    kart_tire.coefficients_at(LOADS)
    assert (kart_tire.temperature, kart_tire.pressure) == before


def test_friction_circle_limit_follows_the_current_grip(kart_tire):
    # A demand far beyond the limit is cut back to exactly the peak force at this state
    fy, fx = kart_tire.calculate_forces(KART_F_Z0, 0.3, 0.3)
    assert np.hypot(fx, fy) == pytest.approx(kart_tire.calculate_d(KART_F_Z0))


def test_sliding_heats_the_tire_and_raises_its_pressure(kart_tire):
    start = (kart_tire.temperature, kart_tire.pressure)
    for _ in range(100):
        kart_tire.update(0.1, KART_F_Z0, 0.1, 0.05, 20.0)
    assert kart_tire.temperature > start[0]
    assert kart_tire.pressure > start[1]


def test_a_rolling_tire_without_slip_does_not_heat(kart_tire):
    for _ in range(50):
        kart_tire.update(0.1, KART_F_Z0, 0.0, 0.0, 20.0)
    assert kart_tire.temperature == pytest.approx(ThermalParams().initial_temperature_k)


def test_a_hot_tire_without_slip_cools_toward_ambient():
    hot = make_kart_tire(ThermalParams(initial_temperature_k=360.0))
    for _ in range(100):
        previous = hot.temperature
        hot.update(0.5, KART_F_Z0, 0.0, 0.0, 20.0, ambient=293.15)
        assert 293.15 < hot.temperature < previous


def test_grip_changes_as_the_tire_warms_up(kart_tire):
    cold_peak = kart_tire.calculate_d(KART_F_Z0)
    for _ in range(600):
        kart_tire.update(0.1, KART_F_Z0, 0.1, 0.05, 20.0)
    assert kart_tire.calculate_d(KART_F_Z0) > cold_peak  # the placeholder tire starts below optimum


def test_update_uses_the_same_conditions_as_calculate_forces(kart_tire):
    # Same step, spelled with degrees and radians, gives the same temperature
    other = kart_tire.copy()
    kart_tire.update(1.0, KART_F_Z0, 5.0, 0.05, 20.0, alpha_unit="deg")
    other.update(1.0, KART_F_Z0, np.radians(5.0), 0.05, 20.0, alpha_unit="rad")
    assert kart_tire.temperature == pytest.approx(other.temperature, rel=1e-12)


def test_reversing_speed_heats_like_forward_speed(kart_tire):
    other = kart_tire.copy()
    kart_tire.update(1.0, KART_F_Z0, 0.1, 0.05, 20.0)
    other.update(1.0, KART_F_Z0, 0.1, 0.05, -20.0)
    assert other.temperature == pytest.approx(kart_tire.temperature, rel=1e-12)


@pytest.mark.parametrize("name", ["dt", "f_z", "alpha", "kappa", "speed"])
def test_update_rejects_arrays(kart_tire, name):
    args = dict(dt=0.1, f_z=KART_F_Z0, alpha=0.1, kappa=0.05, speed=20.0)
    args[name] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match=name):
        kart_tire.update(**args)


def test_reset_returns_the_tire_to_its_initial_state(kart_tire):
    initial = (kart_tire.temperature, kart_tire.pressure)
    for _ in range(100):
        kart_tire.update(0.1, KART_F_Z0, 0.1, 0.05, 20.0)
    kart_tire.reset()
    assert (kart_tire.temperature, kart_tire.pressure) == initial


def test_a_copy_has_its_own_state(kart_tire):
    for _ in range(50):
        kart_tire.update(0.1, KART_F_Z0, 0.1, 0.05, 20.0)
    clone = kart_tire.copy()
    assert (clone.temperature, clone.pressure) == (kart_tire.temperature, kart_tire.pressure)

    before = kart_tire.temperature
    for _ in range(50):
        clone.update(0.1, KART_F_Z0, 0.1, 0.05, 20.0)
    assert kart_tire.temperature == before
    assert clone.temperature > before


def test_copying_a_stateless_tire_works(kart_base):
    clone = kart_base.copy()
    assert clone is not kart_base
    assert np.array_equal(forces_grid(clone), forces_grid(kart_base))


def test_thermal_only_tire_has_no_pressure(kart_base):
    tire = make_kart_tire(ThermalParams())
    assert tire.pressure is None
    tire.update(1.0, KART_F_Z0, 0.1, 0.05, 20.0)
    assert tire.temperature > ThermalParams().initial_temperature_k


def test_pressure_only_tire_holds_its_pressure_and_ignores_updates():
    params = PressureParams(fill_pressure_pa=100_000.0)
    with pytest.warns(PressureWithoutThermalWarning):
        tire = make_kart_tire(pressure=params)
    forces_before = forces_grid(tire)
    for _ in range(20):
        tire.update(1.0, KART_F_Z0, 0.1, 0.05, 20.0)
    assert tire.temperature is None
    assert tire.pressure == 100_000.0
    assert np.array_equal(forces_grid(tire), forces_before)


@pytest.mark.parametrize("builder", ["init", "macro", "constants"])
def test_the_pressure_only_warning_points_at_the_users_line(builder):
    params = PressureParams()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if builder == "init":
            Tire("a", 1.6, pressure=params)
        elif builder == "macro":
            Tire.from_macro_properties(
                "a", 1.6, cornering_stiffness_per_deg=25.0, longitudinal_stiffness=2000.0, pressure=params
            )
        else:
            Tire.from_constants("a", 1.5, b_x=8.0, c_x=1.4, e_x=0.1, pressure=params)
    assert len(caught) == 1
    assert issubclass(caught[0].category, PressureWithoutThermalWarning)
    assert caught[0].filename == __file__


def test_load_sensitivity_exponent_is_not_pushed_above_one():
    # n is 0.99, and a high enough pressure would scale it past 1 without the cap
    params = PressureParams(fill_pressure_pa=300_000.0, load_sensitivity_slope=0.5)
    with pytest.warns(PressureWithoutThermalWarning):
        tire = make_kart_tire(load_sensitivity_n=0.99, pressure=params)
    assert tire.calculate_d(2 * KART_F_Z0) <= 2 * tire.calculate_d(KART_F_Z0) * (1 + 1e-12)


def test_state_does_not_disturb_array_calls():
    tire = make_kart_tire(ThermalParams(), PressureParams())
    grid = forces_grid(tire)
    for i, f_z in enumerate(LOADS[:, 0]):
        for j, alpha in enumerate(SLIP_ANGLES):
            for k, kappa in enumerate(SLIP_RATIOS):
                scalar = tire.calculate_forces(float(f_z), float(alpha), float(kappa))
                assert scalar[0] == pytest.approx(grid[0][i, j, k], rel=1e-12, abs=1e-9)
                assert scalar[1] == pytest.approx(grid[1][i, j, k], rel=1e-12, abs=1e-9)


def test_combined_factors_are_what_the_tire_applies():
    thermal, pressure = ThermalParams(), PressureParams()
    tire = make_kart_tire(thermal, pressure)
    cold_pressure = pressure.fill_pressure_pa
    factors = combine_factors(
        thermal_factors(thermal.initial_temperature_k, thermal),
        pressure_factors(cold_pressure, pressure),
    )
    assert tire.calculate_d(KART_F_Z0) == pytest.approx(KART_MU * KART_F_Z0 * factors.grip)
