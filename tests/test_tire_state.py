"""Tests for the temperature and pressure model (pacejka.tire_state)."""

import dataclasses
import math
import threading
import warnings

import numpy as np
import pytest

from pacejka.tire_state import (
    DEFAULT_AMBIENT_K,
    FACTOR_MAX,
    FACTOR_MIN,
    IDENTITY,
    STANDARD_ATMOSPHERE_PA,
    PressureParams,
    PressureWithoutThermalWarning,
    ThermalParams,
    TireFactors,
    TireSnapshot,
    TireState,
    combine_factors,
    gauge_pressure_from_temperature,
    next_temperature_k,
    pressure_factors,
    slip_power_w,
    thermal_factors,
)

THERMAL = ThermalParams()
PRESSURE = PressureParams()


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(reference_temperature_k=0.0),
        dict(optimal_temperature_k=-5.0),
        dict(grip_width_k=0.0),
        dict(heat_capacity_j_per_k=-1.0),
        dict(initial_temperature_k=0.0),
        dict(grip_floor=0.0),
        dict(grip_floor=1.5),
        dict(heating_fraction=-0.1),
        dict(heating_fraction=1.1),
        dict(convection_w_per_k=-1.0),
        dict(convection_speed_coeff=-1.0),
    ],
)
def test_invalid_thermal_params_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ThermalParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(reference_pressure_pa=0.0),
        dict(optimal_pressure_pa=-1.0),
        dict(fill_pressure_pa=-1.0),
        dict(grip_curvature=-0.1),
    ],
)
def test_invalid_pressure_params_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PressureParams(**kwargs)


def test_params_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        THERMAL.grip_width_k = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        PRESSURE.fill_pressure_pa = 1.0


# ----------------------------------------------------------------------
# Factors
# ----------------------------------------------------------------------


def test_all_factors_are_one_at_the_reference_conditions():
    assert thermal_factors(THERMAL.reference_temperature_k, THERMAL) == IDENTITY
    assert pressure_factors(PRESSURE.reference_pressure_pa, PRESSURE) == IDENTITY


def test_factors_are_one_at_reference_even_when_reference_is_not_the_optimum():
    thermal = ThermalParams(reference_temperature_k=320.0, optimal_temperature_k=350.0)
    pressure = PressureParams(reference_pressure_pa=60_000.0, optimal_pressure_pa=90_000.0)
    assert thermal_factors(320.0, thermal) == IDENTITY
    assert pressure_factors(60_000.0, pressure) == IDENTITY


def test_grip_peaks_at_the_optimal_temperature():
    thermal = ThermalParams(reference_temperature_k=320.0, optimal_temperature_k=350.0)
    temperatures = np.linspace(280.0, 420.0, 281)
    grip = thermal_factors(temperatures, thermal).grip
    assert temperatures[np.argmax(grip)] == pytest.approx(350.0, abs=0.5)
    assert grip.max() > 1.0  # better than at the (cooler) reference


def test_grip_rises_toward_the_optimum_and_falls_after_it():
    cold = np.linspace(280.0, THERMAL.optimal_temperature_k, 50)
    hot = np.linspace(THERMAL.optimal_temperature_k, 420.0, 50)
    assert np.all(np.diff(thermal_factors(cold, THERMAL).grip) > 0)
    assert np.all(np.diff(thermal_factors(hot, THERMAL).grip) < 0)


def test_far_from_optimum_grip_settles_at_the_floor():
    assert thermal_factors(150.0, THERMAL).grip == pytest.approx(THERMAL.grip_floor, rel=1e-6)


def test_stiffness_falls_as_temperature_rises():
    temperatures = np.linspace(280.0, 400.0, 20)
    factors = thermal_factors(temperatures, THERMAL)
    assert np.all(np.diff(factors.cornering_stiffness) < 0)
    assert np.all(np.diff(factors.longitudinal_stiffness) < 0)


def test_temperature_leaves_load_sensitivity_alone():
    assert thermal_factors(300.0, THERMAL).load_sensitivity == 1.0


def test_grip_peaks_at_the_optimal_pressure():
    pressure = PressureParams(reference_pressure_pa=60_000.0, optimal_pressure_pa=90_000.0)
    pressures = np.linspace(20_000.0, 200_000.0, 361)
    grip = pressure_factors(pressures, pressure).grip
    assert pressures[np.argmax(grip)] == pytest.approx(90_000.0, abs=500.0)


def test_cornering_stiffness_rises_with_pressure():
    factors = pressure_factors(np.linspace(40_000.0, 160_000.0, 20), PRESSURE)
    assert np.all(np.diff(factors.cornering_stiffness) > 0)


def test_pressure_leaves_longitudinal_stiffness_alone():
    assert pressure_factors(120_000.0, PRESSURE).longitudinal_stiffness == 1.0


@pytest.mark.parametrize("function, params", [(thermal_factors, THERMAL), (pressure_factors, PRESSURE)])
def test_extreme_inputs_stay_within_the_factor_limits(function, params):
    for value in (1.0, 1e-3, 1e3, 1e6, 1e9):
        factors = function(value, params)
        for field in dataclasses.fields(TireFactors):
            assert FACTOR_MIN <= getattr(factors, field.name) <= FACTOR_MAX


def test_factor_functions_accept_arrays_and_match_scalar_calls():
    temperatures = np.linspace(280.0, 400.0, 7)
    pressures = np.linspace(40_000.0, 160_000.0, 7)
    for array_factors, scalar_function, values, params in (
        (thermal_factors(temperatures, THERMAL), thermal_factors, temperatures, THERMAL),
        (pressure_factors(pressures, PRESSURE), pressure_factors, pressures, PRESSURE),
    ):
        for i, value in enumerate(values):
            scalar = scalar_function(float(value), params)
            for field in dataclasses.fields(TireFactors):
                name = field.name
                expected = getattr(scalar, name)
                got = getattr(array_factors, name)
                got = got[i] if isinstance(got, np.ndarray) else got
                assert got == pytest.approx(expected, rel=1e-12)


def test_scalar_input_gives_plain_float_factors():
    factors = thermal_factors(300.0, THERMAL)
    assert all(type(getattr(factors, f.name)) is float for f in dataclasses.fields(TireFactors))


def test_combine_factors_multiplies_field_by_field():
    a = TireFactors(grip=0.9, cornering_stiffness=1.1)
    b = TireFactors(grip=0.8, longitudinal_stiffness=1.2, load_sensitivity=1.05)
    combined = combine_factors(a, b)
    assert combined.grip == pytest.approx(0.72)
    assert combined.cornering_stiffness == pytest.approx(1.1)
    assert combined.longitudinal_stiffness == pytest.approx(1.2)
    assert combined.load_sensitivity == pytest.approx(1.05)


def test_combine_factors_of_nothing_is_the_identity():
    assert combine_factors() == IDENTITY


def test_combine_factors_clamps_the_product():
    low = TireFactors(grip=0.3)
    high = TireFactors(grip=1.9)
    assert combine_factors(low, low).grip == FACTOR_MIN
    assert combine_factors(high, high).grip == FACTOR_MAX


def test_factors_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        IDENTITY.grip = 2.0


# ----------------------------------------------------------------------
# Gas law
# ----------------------------------------------------------------------


def test_pressure_at_the_fill_temperature_is_the_fill_pressure():
    assert gauge_pressure_from_temperature(293.15, 293.15, 80_000.0) == pytest.approx(80_000.0)


def test_gas_law_uses_absolute_pressure_and_temperature():
    expected = (80_000.0 + STANDARD_ATMOSPHERE_PA) * 343.15 / 293.15 - STANDARD_ATMOSPHERE_PA
    assert gauge_pressure_from_temperature(343.15, 293.15, 80_000.0) == pytest.approx(expected)
    # Doubling the absolute temperature doubles the absolute pressure
    hot = gauge_pressure_from_temperature(586.3, 293.15, 80_000.0)
    assert hot + STANDARD_ATMOSPHERE_PA == pytest.approx(2 * (80_000.0 + STANDARD_ATMOSPHERE_PA))


def test_a_hotter_tire_has_more_pressure_and_a_colder_one_less():
    assert gauge_pressure_from_temperature(330.0, 293.15, 80_000.0) > 80_000.0
    assert gauge_pressure_from_temperature(270.0, 293.15, 80_000.0) < 80_000.0


def test_gas_law_accepts_arrays():
    temperatures = np.array([280.0, 300.0, 340.0])
    result = gauge_pressure_from_temperature(temperatures, 293.15, 80_000.0)
    assert result.shape == (3,)
    assert np.all(np.diff(result) > 0)


@pytest.mark.parametrize("temperature, fill", [(0.0, 293.15), (-10.0, 293.15), (300.0, 0.0)])
def test_gas_law_rejects_non_positive_temperatures(temperature, fill):
    with pytest.raises(ValueError):
        gauge_pressure_from_temperature(temperature, fill, 80_000.0)


# ----------------------------------------------------------------------
# Sliding power
# ----------------------------------------------------------------------


def test_pure_lateral_slip_power_is_force_times_lateral_sliding_speed():
    power = slip_power_w(f_x=0.0, f_y=1000.0, slip_ratio=0.0, slip_angle_rad=0.1, speed_ms=20.0)
    assert power == pytest.approx(1000.0 * 20.0 * math.sin(0.1))


def test_pure_longitudinal_slip_power_is_force_times_longitudinal_sliding_speed():
    power = slip_power_w(f_x=800.0, f_y=0.0, slip_ratio=0.1, slip_angle_rad=0.0, speed_ms=20.0)
    assert power == pytest.approx(800.0 * 0.1 * 20.0)


def test_rolling_without_slip_makes_no_heat():
    assert slip_power_w(500.0, 500.0, 0.0, 0.0, 30.0) == 0.0


def test_standing_still_makes_no_heat():
    assert slip_power_w(500.0, 500.0, 0.1, 0.1, 0.0) == 0.0


def test_slip_power_is_never_negative_and_ignores_signs():
    positive = slip_power_w(300.0, 700.0, 0.05, 0.08, 15.0)
    negative = slip_power_w(-300.0, -700.0, -0.05, -0.08, -15.0)
    assert positive > 0
    assert negative == pytest.approx(positive)


def test_slip_power_broadcasts_arrays():
    speeds = np.array([5.0, 10.0, 20.0])
    power = slip_power_w(100.0, 400.0, 0.02, 0.05, speeds)
    assert power.shape == (3,)
    assert power == pytest.approx(power[0] * speeds / speeds[0])


# ----------------------------------------------------------------------
# Temperature dynamics
# ----------------------------------------------------------------------

AMBIENT = 293.15


def step(temperature, dt, power=0.0, speed=0.0, ambient=AMBIENT, params=THERMAL):
    return next_temperature_k(temperature, dt, power, speed, ambient, params)


def test_zero_time_step_changes_nothing():
    assert step(330.0, 0.0, power=2000.0, speed=20.0) == 330.0


def test_a_hot_tire_cools_toward_ambient_without_overshooting():
    # At standstill the time constant is C / h = 400 s, so run for 15 time constants
    temperature = 360.0
    for _ in range(6000):
        new = step(temperature, 1.0)
        assert AMBIENT <= new < temperature
        temperature = new
    assert temperature == pytest.approx(AMBIENT, abs=1e-3)


def test_a_cold_tire_warms_up_to_the_steady_temperature():
    power, speed = 2000.0, 20.0
    loss = THERMAL.convection_w_per_k + THERMAL.convection_speed_coeff * speed
    steady = AMBIENT + THERMAL.heating_fraction * power / loss

    temperature = AMBIENT
    for _ in range(5000):
        new = step(temperature, 0.5, power, speed)
        assert temperature <= new <= steady + 1e-9
        temperature = new
    assert temperature == pytest.approx(steady, abs=1e-3)


def test_more_power_gives_a_hotter_steady_state():
    def settle(power):
        temperature = AMBIENT
        for _ in range(3000):
            temperature = step(temperature, 1.0, power, 15.0)
        return temperature

    assert settle(500.0) < settle(1500.0) < settle(4000.0)


def test_more_speed_cools_the_tire_more():
    def settle(speed):
        temperature = AMBIENT
        for _ in range(3000):
            temperature = step(temperature, 1.0, 1500.0, speed)
        return temperature

    assert settle(30.0) < settle(10.0)


def test_two_half_steps_equal_one_full_step():
    one = step(310.0, 2.0, power=1500.0, speed=12.0)
    two = step(step(310.0, 1.0, power=1500.0, speed=12.0), 1.0, power=1500.0, speed=12.0)
    assert two == pytest.approx(one, rel=1e-12)


def test_a_huge_time_step_is_stable_and_lands_on_the_steady_state():
    power, speed = 2000.0, 20.0
    loss = THERMAL.convection_w_per_k + THERMAL.convection_speed_coeff * speed
    steady = AMBIENT + THERMAL.heating_fraction * power / loss
    assert step(AMBIENT, 1e9, power, speed) == pytest.approx(steady)


def test_without_any_cooling_heat_only_accumulates():
    params = ThermalParams(convection_w_per_k=0.0, convection_speed_coeff=0.0)
    expected = 300.0 + params.heating_fraction * 1000.0 * 10.0 / params.heat_capacity_j_per_k
    assert step(300.0, 10.0, power=1000.0, speed=20.0, params=params) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [dict(dt=-1.0), dict(power=-1.0), dict(speed=-1.0), dict(ambient=0.0)],
)
def test_invalid_step_inputs_are_rejected(kwargs):
    args = dict(dt=1.0, power=0.0, speed=0.0, ambient=AMBIENT)
    args.update(kwargs)
    with pytest.raises(ValueError):
        step(300.0, args["dt"], args["power"], args["speed"], args["ambient"])


def test_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError):
        step(0.0, 1.0)


# ----------------------------------------------------------------------
# TireState
# ----------------------------------------------------------------------


def test_a_state_needs_thermal_or_pressure_parameters():
    with pytest.raises(ValueError):
        TireState()


def test_thermal_and_pressure_start_at_the_initial_values():
    state = TireState(THERMAL, PRESSURE)
    assert state.temperature_k == THERMAL.initial_temperature_k
    assert state.pressure_pa == PRESSURE.fill_pressure_pa
    assert state.is_dynamic


def test_thermal_only_has_a_temperature_and_no_pressure():
    state = TireState(THERMAL)
    assert state.temperature_k == THERMAL.initial_temperature_k
    assert state.pressure_pa is None
    assert state.factors() == thermal_factors(THERMAL.initial_temperature_k, THERMAL)


def test_pressure_only_warns_and_has_a_pressure_and_no_temperature():
    with pytest.warns(PressureWithoutThermalWarning, match="constant"):
        state = TireState(None, PRESSURE)
    assert state.temperature_k is None
    assert state.pressure_pa == PRESSURE.fill_pressure_pa
    assert not state.is_dynamic
    assert state.factors() == pressure_factors(PRESSURE.fill_pressure_pa, PRESSURE)


def test_the_pressure_only_warning_points_at_the_callers_line():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TireState(None, PRESSURE)
    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_the_warning_is_a_user_warning_that_can_be_silenced():
    assert issubclass(PressureWithoutThermalWarning, UserWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", PressureWithoutThermalWarning)
        TireState(None, PRESSURE)  # would raise if the filter did not apply


def test_other_configurations_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TireState(THERMAL)
        TireState(THERMAL, PRESSURE)


def test_pressure_only_state_ignores_updates():
    with pytest.warns(PressureWithoutThermalWarning):
        state = TireState(None, PRESSURE)
    before = state.snapshot
    state.update(10.0, 5000.0, 30.0)
    assert state.snapshot == before


def test_update_heats_the_tire_and_raises_its_pressure():
    state = TireState(THERMAL, PRESSURE)
    start = state.snapshot
    state.update(5.0, 2000.0, 20.0)
    assert state.temperature_k > start.temperature_k
    assert state.pressure_pa > start.pressure_pa


def test_pressure_always_matches_the_gas_law_for_the_current_temperature():
    state = TireState(THERMAL, PRESSURE)
    for _ in range(50):
        state.update(0.5, 1500.0, 18.0)
        expected = gauge_pressure_from_temperature(
            state.temperature_k, THERMAL.initial_temperature_k, PRESSURE.fill_pressure_pa
        )
        assert state.pressure_pa == pytest.approx(expected)


def test_factors_follow_the_state():
    state = TireState(THERMAL, PRESSURE)
    cold = state.factors()
    for _ in range(100):
        state.update(0.5, 2000.0, 20.0)
    assert state.factors() != cold
    snapshot = state.snapshot
    expected = combine_factors(
        thermal_factors(snapshot.temperature_k, THERMAL),
        pressure_factors(snapshot.pressure_pa, PRESSURE),
    )
    assert state.factors() == expected


def test_thermal_only_state_has_thermal_factors_only():
    state = TireState(THERMAL)
    state.update(5.0, 2000.0, 20.0)
    assert state.factors() == thermal_factors(state.temperature_k, THERMAL)


def test_reset_returns_to_the_initial_state():
    state = TireState(THERMAL, PRESSURE)
    initial = state.snapshot
    state.update(30.0, 3000.0, 25.0)
    assert state.snapshot != initial
    state.reset()
    assert state.snapshot == initial


def test_copy_is_independent_of_the_original():
    state = TireState(THERMAL, PRESSURE)
    state.update(5.0, 2000.0, 20.0)
    clone = state.copy()
    assert clone.snapshot == state.snapshot

    clone.update(20.0, 3000.0, 25.0)
    assert clone.snapshot != state.snapshot
    clone.reset()
    assert state.temperature_k > THERMAL.initial_temperature_k  # untouched by the clone's reset


def test_copying_a_pressure_only_state_does_not_warn_again():
    with pytest.warns(PressureWithoutThermalWarning):
        state = TireState(None, PRESSURE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state.copy()


def test_snapshots_are_frozen():
    snapshot = TireState(THERMAL, PRESSURE).snapshot
    assert isinstance(snapshot, TireSnapshot)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.temperature_k = 1.0


def test_a_reader_always_sees_a_matching_temperature_and_pressure():
    state = TireState(THERMAL, PRESSURE)
    mismatches = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            snapshot = state.snapshot
            expected = gauge_pressure_from_temperature(
                snapshot.temperature_k, THERMAL.initial_temperature_k, PRESSURE.fill_pressure_pa
            )
            if snapshot.pressure_pa != pytest.approx(expected):
                mismatches.append(snapshot)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    for i in range(3000):
        if i % 500 == 0:
            state.reset()
        state.update(0.05, 2500.0, 22.0)
    done.set()
    for thread in threads:
        thread.join()
    assert not mismatches


def test_default_ambient_is_twenty_degrees_celsius():
    assert DEFAULT_AMBIENT_K == pytest.approx(293.15)
