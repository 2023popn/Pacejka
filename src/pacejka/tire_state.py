"""Tire temperature and pressure, and how they change grip.

This module is standalone: it never imports ``Tire``. It does two jobs.

* **Factors.** Turn a temperature and a pressure into multipliers
  (``TireFactors``) for the tire parameters the magic formula is built from:
  peak friction, the two stiffnesses and the load sensitivity. Every factor is
  1.0 at the reference temperature and pressure, which are the conditions your
  base tire parameters were measured at.
* **Dynamics.** Track a single lumped tire temperature through time, and derive
  the pressure from it with the gas law (``TireState``).

``Tire`` combines the two: it asks its ``TireState`` for the current factors and
applies them inside its own calculations.

Units are SI throughout: kelvin, pascals (gauge pressure), seconds, newtons,
metres per second, watts, joules.

Every default in ``ThermalParams`` and ``PressureParams`` is a PLACEHOLDER chosen
to give plausible behavior for a small kart tire. None of them come from
measured data. Replace them with your own before trusting the results.
"""

import copy as _copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

# Inputs can be single numbers or arrays; results take the shape of the inputs
FloatOrArray = Union[float, np.ndarray]

# Atmospheric pressure used to convert between gauge and absolute pressure
STANDARD_ATMOSPHERE_PA = 101_325.0

ZERO_CELSIUS_K = 273.15

# Ambient temperature used when none is given (20 degrees C)
DEFAULT_AMBIENT_K = ZERO_CELSIUS_K + 20.0

# Every factor is limited to this range, so extreme inputs cannot remove grip
# entirely or make a tire implausibly grippy.
FACTOR_MIN = 0.2
FACTOR_MAX = 2.0


class PressureWithoutThermalWarning(UserWarning):
    """Pressure parameters were given without thermal parameters."""


def _as_input(value) -> FloatOrArray:
    """Return a plain float for a single value, or a float array otherwise."""
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array


def _clamp(value: FloatOrArray) -> FloatOrArray:
    """Limit a factor to [FACTOR_MIN, FACTOR_MAX]."""
    if isinstance(value, float):
        return min(max(value, FACTOR_MIN), FACTOR_MAX)
    return np.clip(value, FACTOR_MIN, FACTOR_MAX)


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalParams:
    """
    Settings for the lumped (single temperature) thermal model of one tire.

    All defaults are placeholders, not measured values.

    Attributes:
        reference_temperature_k: Temperature your base tire parameters were
            measured at. All thermal factors are 1.0 here.
        optimal_temperature_k: Temperature of peak grip.
        grip_width_k: Width (standard deviation, in kelvin) of the grip peak.
        grip_floor: Grip far from the optimum, as a fraction of peak grip
            (between 0 and 1).
        stiffness_temp_coeff: Fractional change in stiffness per kelvin above
            the reference temperature (negative: the tire softens as it heats).
        heat_capacity_j_per_k: Lumped heat capacity of the tire.
        heating_fraction: Fraction of the sliding power that heats the tire
            (between 0 and 1).
        convection_w_per_k: Heat loss per kelvin above ambient at standstill.
        convection_speed_coeff: Extra heat loss per kelvin above ambient for
            each metre per second of speed (W / K / (m/s)).
        initial_temperature_k: Temperature at construction and after ``reset``.
    """

    reference_temperature_k: float = 343.15
    optimal_temperature_k: float = 343.15
    grip_width_k: float = 25.0
    grip_floor: float = 0.6
    stiffness_temp_coeff: float = -0.002
    heat_capacity_j_per_k: float = 2000.0
    heating_fraction: float = 0.5
    convection_w_per_k: float = 5.0
    convection_speed_coeff: float = 1.5
    initial_temperature_k: float = DEFAULT_AMBIENT_K

    def __post_init__(self):
        for name in (
            "reference_temperature_k",
            "optimal_temperature_k",
            "grip_width_k",
            "heat_capacity_j_per_k",
            "initial_temperature_k",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if not 0 < self.grip_floor <= 1:
            raise ValueError(f"grip_floor must be in (0, 1], got {self.grip_floor!r}")
        if not 0 <= self.heating_fraction <= 1:
            raise ValueError(f"heating_fraction must be in [0, 1], got {self.heating_fraction!r}")
        for name in ("convection_w_per_k", "convection_speed_coeff"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} cannot be negative, got {getattr(self, name)!r}")


@dataclass(frozen=True)
class PressureParams:
    """
    Settings for how tire pressure changes grip. All pressures are gauge, in pascals.

    All defaults are placeholders, not measured values.

    Attributes:
        reference_pressure_pa: Pressure your base tire parameters were measured
            at. All pressure factors are 1.0 here.
        optimal_pressure_pa: Pressure of peak grip.
        grip_curvature: How quickly grip falls away from the optimal pressure.
            Grip drops by ``grip_curvature * x**2`` (as a fraction), where x is
            the pressure's distance from the optimum relative to the optimum.
        stiffness_sensitivity: Relative change in cornering stiffness per
            relative change in pressure.
        load_sensitivity_slope: Relative change in the load sensitivity
            exponent per relative change in pressure.
        fill_pressure_pa: With a thermal model, the pressure at the initial
            temperature (the cold fill pressure); pressure then rises with
            temperature. Without one, the constant pressure the tire is held
            at, so give the pressure it runs at, not the cold fill pressure.
    """

    reference_pressure_pa: float = 80_000.0
    optimal_pressure_pa: float = 80_000.0
    grip_curvature: float = 0.5
    stiffness_sensitivity: float = 0.3
    load_sensitivity_slope: float = 0.05
    fill_pressure_pa: float = 80_000.0

    def __post_init__(self):
        for name in ("reference_pressure_pa", "optimal_pressure_pa"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if not self.fill_pressure_pa >= 0:
            raise ValueError(f"fill_pressure_pa cannot be negative, got {self.fill_pressure_pa!r}")
        if not self.grip_curvature >= 0:
            raise ValueError(f"grip_curvature cannot be negative, got {self.grip_curvature!r}")


# ----------------------------------------------------------------------
# Factors
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TireFactors:
    """
    Multipliers for the tire parameters behind the magic formula.

    All are 1.0 at the reference temperature and pressure.

    Attributes:
        grip: Scales the peak friction coefficient.
        longitudinal_stiffness: Scales the longitudinal stiffness.
        cornering_stiffness: Scales the cornering stiffness.
        load_sensitivity: Scales the load sensitivity exponent.
    """

    grip: FloatOrArray = 1.0
    longitudinal_stiffness: FloatOrArray = 1.0
    cornering_stiffness: FloatOrArray = 1.0
    load_sensitivity: FloatOrArray = 1.0


# No change to the tire
IDENTITY = TireFactors()


def combine_factors(*factors: TireFactors) -> TireFactors:
    """Multiply factors together field by field, then limit the result to the allowed range."""
    grip = longitudinal = cornering = load_sensitivity = 1.0
    for item in factors:
        grip = grip * item.grip
        longitudinal = longitudinal * item.longitudinal_stiffness
        cornering = cornering * item.cornering_stiffness
        load_sensitivity = load_sensitivity * item.load_sensitivity
    return TireFactors(
        grip=_clamp(grip),
        longitudinal_stiffness=_clamp(longitudinal),
        cornering_stiffness=_clamp(cornering),
        load_sensitivity=_clamp(load_sensitivity),
    )


def thermal_factors(temperature_k: ArrayLike, params: ThermalParams) -> TireFactors:
    """
    Grip and stiffness factors at a tire temperature.

    Grip follows a bell curve around the optimal temperature, normalized so it
    is 1.0 at the reference temperature. Stiffness changes linearly with
    temperature. Only ``grip`` and the two stiffnesses are affected. Accepts a
    single temperature or an array of them.
    """
    temperature = _as_input(temperature_k)

    def grip_curve(t):
        z = (t - params.optimal_temperature_k) / params.grip_width_k
        return params.grip_floor + (1.0 - params.grip_floor) * np.exp(-0.5 * z * z)

    grip = grip_curve(temperature) / grip_curve(params.reference_temperature_k)
    stiffness = 1.0 + params.stiffness_temp_coeff * (temperature - params.reference_temperature_k)

    return TireFactors(
        grip=_clamp(_as_input(grip)),
        longitudinal_stiffness=_clamp(_as_input(stiffness)),
        cornering_stiffness=_clamp(_as_input(stiffness)),
    )


def pressure_factors(pressure_pa: ArrayLike, params: PressureParams) -> TireFactors:
    """
    Grip, cornering stiffness and load sensitivity factors at a gauge pressure.

    Grip falls off quadratically away from the optimal pressure, normalized so
    it is 1.0 at the reference pressure. Cornering stiffness and the load
    sensitivity exponent change linearly with pressure relative to the
    reference. Longitudinal stiffness is not affected. Accepts a single
    pressure or an array of them.
    """
    pressure = _as_input(pressure_pa)
    ratio = pressure / params.reference_pressure_pa

    def grip_curve(p):
        x = (p - params.optimal_pressure_pa) / params.optimal_pressure_pa
        return np.maximum(1.0 - params.grip_curvature * x * x, FACTOR_MIN)

    grip = grip_curve(pressure) / grip_curve(params.reference_pressure_pa)
    stiffness = 1.0 + params.stiffness_sensitivity * (ratio - 1.0)
    load_sensitivity = 1.0 + params.load_sensitivity_slope * (ratio - 1.0)

    return TireFactors(
        grip=_clamp(_as_input(grip)),
        cornering_stiffness=_clamp(_as_input(stiffness)),
        load_sensitivity=_clamp(_as_input(load_sensitivity)),
    )


# ----------------------------------------------------------------------
# Physics helpers
# ----------------------------------------------------------------------


def gauge_pressure_from_temperature(
    temperature_k: ArrayLike,
    fill_temperature_k: float,
    fill_pressure_pa: float,
    atmospheric_pa: float = STANDARD_ATMOSPHERE_PA,
) -> FloatOrArray:
    """
    Gauge pressure of the air in a sealed tire at a given temperature.

    Constant volume gas law: the absolute pressure is proportional to the
    absolute temperature. The tire is filled to ``fill_pressure_pa`` (gauge) at
    ``fill_temperature_k``. Pressures are converted to absolute for the
    calculation and back to gauge for the result.
    """
    temperature = _as_input(temperature_k)
    if np.any(temperature <= 0) or fill_temperature_k <= 0:
        raise ValueError("Temperatures must be positive (kelvin).")
    absolute_fill = fill_pressure_pa + atmospheric_pa
    return _as_input(absolute_fill * temperature / fill_temperature_k - atmospheric_pa)


def slip_power_w(
    f_x: ArrayLike,
    f_y: ArrayLike,
    slip_ratio: ArrayLike,
    slip_angle_rad: ArrayLike,
    speed_ms: ArrayLike,
) -> FloatOrArray:
    """
    Power dissipated by the tire sliding on the road, in watts.

    Force times sliding speed on each axis. ``speed_ms`` is the speed of the
    tire over the ground. The lateral sliding speed is ``speed * sin(alpha)``
    and the longitudinal sliding speed is ``kappa * speed * cos(alpha)``, since
    the slip ratio is relative to the forward speed. Magnitudes are used, so
    the result is never negative.
    """
    f_x, f_y, kappa, alpha, speed = (
        np.asarray(item, dtype=float) for item in (f_x, f_y, slip_ratio, slip_angle_rad, speed_ms)
    )
    speed = np.abs(speed)
    power = np.abs(f_y) * speed * np.abs(np.sin(alpha)) + np.abs(f_x) * np.abs(kappa) * speed * np.abs(
        np.cos(alpha)
    )
    return _as_input(power)


def next_temperature_k(
    temperature_k: float,
    dt_s: float,
    power_w: float,
    speed_ms: float,
    ambient_k: float,
    params: ThermalParams,
) -> float:
    """
    Tire temperature after ``dt_s`` seconds of constant heating and cooling.

    The lumped balance is ``C dT/dt = eta * P - h * (T - T_ambient)`` where ``h``
    grows with speed. Over a step with constant inputs this has an exact
    solution, an exponential approach to the steady temperature
    ``T_ambient + eta * P / h``. Using it (instead of a simple Euler step)
    keeps the result stable for any step size.

    Args:
        temperature_k: Current temperature.
        dt_s: Time step, zero or more.
        power_w: Sliding power, zero or more (see ``slip_power_w``).
        speed_ms: Speed over the ground, zero or more.
        ambient_k: Ambient (air and road) temperature.
        params: Thermal settings.
    """
    if temperature_k <= 0 or ambient_k <= 0:
        raise ValueError("Temperatures must be positive (kelvin).")
    if dt_s < 0 or power_w < 0 or speed_ms < 0:
        raise ValueError("dt_s, power_w and speed_ms cannot be negative.")

    heat_in = params.heating_fraction * power_w
    loss_per_k = params.convection_w_per_k + params.convection_speed_coeff * speed_ms

    if loss_per_k <= 0:  # nothing carries heat away, so it only accumulates
        return temperature_k + heat_in * dt_s / params.heat_capacity_j_per_k

    steady = ambient_k + heat_in / loss_per_k
    decay = math.exp(-loss_per_k * dt_s / params.heat_capacity_j_per_k)
    return steady + (temperature_k - steady) * decay


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TireSnapshot:
    """
    The tire's state at one moment.

    Temperature and pressure travel together in one immutable object, so a
    reader always sees a matching pair even while another thread is updating.

    Attributes:
        temperature_k: Temperature in kelvin, or None without a thermal model.
        pressure_pa: Gauge pressure in pascals, or None without pressure parameters.
    """

    temperature_k: Optional[float] = None
    pressure_pa: Optional[float] = None


class TireState:
    """
    The temperature and pressure of one tire, and the factors they imply.

    Give it ``ThermalParams`` to let the temperature evolve, and
    ``PressureParams`` to make pressure affect grip. With both, the pressure is
    derived from the temperature with the gas law. With only ``PressureParams``
    the pressure stays constant and a ``PressureWithoutThermalWarning`` is
    issued, since pressure normally rises as a tire heats up. With only
    ``ThermalParams`` there are no pressure effects.

    The state is held as an immutable ``TireSnapshot`` that ``update`` replaces
    in a single assignment, so ``factors`` and the properties always read a
    consistent pair. Keep to one writer per tire: concurrent calls to
    ``update`` on the same tire have no meaningful result.
    """

    def __init__(
        self,
        thermal: Optional[ThermalParams] = None,
        pressure: Optional[PressureParams] = None,
        *,
        stacklevel: int = 2,
    ):
        """
        Args:
            thermal: Thermal settings, or None for no temperature model.
            pressure: Pressure settings, or None for no pressure effects.
            stacklevel: Which caller the pressure-only warning points at. The
                default is the caller of this constructor; classes wrapping it
                raise this by one per layer.
        """
        if thermal is None and pressure is None:
            raise ValueError("A TireState needs thermal and/or pressure parameters.")
        if thermal is None:
            warnings.warn(
                "Pressure parameters were given without thermal parameters, so the pressure "
                "is held constant at fill_pressure_pa and will not rise as the tire heats up. "
                "Pass the pressure the tire runs at (not the cold fill pressure), or add "
                "ThermalParams to derive pressure from temperature.",
                PressureWithoutThermalWarning,
                stacklevel=stacklevel,
            )

        self._thermal = thermal
        self._pressure = pressure
        self._initial = TireSnapshot(
            temperature_k=None if thermal is None else thermal.initial_temperature_k,
            pressure_pa=None if pressure is None else pressure.fill_pressure_pa,
        )
        self._snapshot = self._initial

    def __repr__(self) -> str:
        return (
            f"TireState(temperature_k={self._snapshot.temperature_k}, "
            f"pressure_pa={self._snapshot.pressure_pa})"
        )

    # ------------------------------------------------------------------
    # Reading the state
    # ------------------------------------------------------------------

    @property
    def is_dynamic(self) -> bool:
        """True if the state changes with ``update`` (a thermal model is present)."""
        return self._thermal is not None

    @property
    def snapshot(self) -> TireSnapshot:
        """The current temperature and pressure, as one consistent pair."""
        return self._snapshot

    @property
    def temperature_k(self) -> Optional[float]:
        """Current temperature in kelvin, or None without a thermal model."""
        return self._snapshot.temperature_k

    @property
    def pressure_pa(self) -> Optional[float]:
        """Current gauge pressure in pascals, or None without pressure parameters."""
        return self._snapshot.pressure_pa

    def factors(self) -> TireFactors:
        """The combined factors for the current temperature and pressure."""
        snapshot = self._snapshot  # read once so temperature and pressure match
        found = []
        if self._thermal is not None:
            found.append(thermal_factors(snapshot.temperature_k, self._thermal))
        if self._pressure is not None:
            found.append(pressure_factors(snapshot.pressure_pa, self._pressure))
        return combine_factors(*found)

    # ------------------------------------------------------------------
    # Changing the state
    # ------------------------------------------------------------------

    def update(
        self,
        dt_s: float,
        power_w: float,
        speed_ms: float,
        ambient_k: float = DEFAULT_AMBIENT_K,
    ) -> None:
        """
        Advance the state by ``dt_s`` seconds.

        Does nothing without a thermal model. With pressure parameters as well,
        the pressure is re-derived from the new temperature.

        Args:
            dt_s: Time step in seconds.
            power_w: Sliding power over the step (see ``slip_power_w``).
            speed_ms: Speed over the ground in metres per second.
            ambient_k: Ambient temperature in kelvin.
        """
        thermal = self._thermal
        if thermal is None:
            return

        temperature = next_temperature_k(
            self._snapshot.temperature_k, dt_s, power_w, speed_ms, ambient_k, thermal
        )
        pressure = None
        if self._pressure is not None:
            pressure = gauge_pressure_from_temperature(
                temperature, thermal.initial_temperature_k, self._pressure.fill_pressure_pa
            )
        self._snapshot = TireSnapshot(temperature_k=temperature, pressure_pa=pressure)

    def reset(self) -> None:
        """Return to the initial temperature and pressure."""
        self._snapshot = self._initial

    def copy(self) -> "TireState":
        """An independent state with the same settings, starting from the current values."""
        return _copy.copy(self)  # settings and snapshots are immutable, so sharing them is safe
