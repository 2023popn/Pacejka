"""Tire force model based on the pacejka "magic formula"."""

import copy as _copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .tire_state import (
    DEFAULT_AMBIENT_K,
    IDENTITY,
    PressureParams,
    ThermalParams,
    TireFactors,
    TireState,
    slip_power_w,
)

# Conversion factor used to turn a stiffness per degree into a stiffness per radian
DEG_PER_RAD = 180.0 / np.pi

# Inputs can be single numbers or anything array-like; results take the shape of the inputs
FloatOrArray = Union[float, np.ndarray]

# Plain Python/numpy floats take the fast scalar path; anything else takes the array path
_SCALAR = (int, float)

# Accepted values for the ``alpha_unit`` argument. Left as None, the unit is detected.
ALPHA_UNITS = ("rad", "deg")

# When ``alpha_unit`` is not given, slip angles are assumed to be radians unless any value
# has a magnitude beyond this limit (pi/2 rad = 90 deg, where a tire is no longer rolling
# forward). The whole input is then treated as degrees.
ALPHA_RADIAN_LIMIT = np.pi / 2

# Default magic formula shape (C) and curvature (E) factors
DEFAULT_C = 1.4
DEFAULT_E_LONGITUDINAL = 0.1
DEFAULT_E_LATERAL = -0.1

# Defaults used when building a tire from macro properties
DEFAULT_LOAD_SENSITIVITY = 0.9
DEFAULT_REFERENCE_LOAD = 150


@dataclass(frozen=True)
class AxisShape:
    """
    Fixed magic formula settings for one axis (longitudinal or lateral).

    These describe the tire itself. They are set once at construction and
    cannot be modified afterwards.

    Attributes:
        c: Shape factor.
        e: Curvature factor.
        b: Stiffness factor. Only set for ``constant_b`` tires; otherwise B
            depends on the load and is computed per request.
    """

    c: float
    e: float
    b: Optional[float] = None


@dataclass(frozen=True)
class MagicFormulaCoefficients:
    """
    The complete set of magic formula coefficients for one axis at one load.

    Created fresh for every request by ``Tire.coefficients_at`` and never
    stored on the tire, so concurrent calls cannot interfere with each other.
    """

    b: FloatOrArray
    c: float
    d: FloatOrArray
    e: float


class Tire:
    """
    Holds tire parameters and computes tire forces for the applied conditions.

    The fixed magic formula settings for each axis live in ``longitudinal`` and
    ``lateral`` (both ``AxisShape``). The load-dependent coefficients (D, and B
    unless ``constant_b`` is set) are computed on demand by ``coefficients_at``
    and are never stored on the tire.

    The force methods accept single numbers or numpy arrays. Array inputs are
    broadcast together by numpy, so for example a column of loads and a row of
    slip angles produce a grid of forces.

    A tire can also track its own temperature and pressure. Pass ``thermal``
    (``ThermalParams``) and/or ``pressure`` (``PressureParams``) and the tire
    keeps a ``TireState``. The force methods then use its current state without
    ever changing it, and ``update`` advances the state through time. Without
    either, the tire behaves exactly as a plain stateless tire. A tire with
    state must only be updated from one thread at a time: use one ``Tire`` per
    physical tire, and ``copy`` for independent runs. All state is in SI units
    (kelvin, gauge pascals, newtons, metres per second).

    Attributes:
        name (str): Tire name (intended to be the compound name).
        mu_peak (float): Peak coefficient of friction.
        cornering_stiffness_per_rad (float): Force per radian of slip angle (alpha)
            at small angles, i.e. the slope of the force vs. angle curve. The
            constructor only accepts radians; use ``from_macro_properties`` to
            supply a per-degree value.
        longitudinal_stiffness (float): Force per unit of longitudinal slip (kappa).
        load_sensitivity_n (float): Non-linearity of peak friction as load increases.
        f_z0 (float): Typical (reference) load, used in the load sensitivity calculations.
        constant_b (bool): If True, B is supplied by the user and never recomputed.
        longitudinal (AxisShape): Fixed C, E (and B if constant) for the longitudinal axis.
        lateral (AxisShape): Fixed C, E (and B if constant) for the lateral axis.
        temperature (Optional[float]): Current temperature in kelvin, or None if the
            tire has no thermal model.
        pressure (Optional[float]): Current gauge pressure in pascals, or None if the
            tire has no pressure parameters.
    """

    def __init__(
        self,
        name: str,
        mu_peak: float,
        cornering_stiffness_per_rad: Optional[float] = None,
        longitudinal_stiffness: Optional[float] = None,
        load_sensitivity_n: Optional[float] = None,
        f_z0: Optional[float] = None,
        b_x: Optional[float] = None,
        c_x: float = DEFAULT_C,
        e_x: float = DEFAULT_E_LONGITUDINAL,
        b_y: Optional[float] = None,
        c_y: float = DEFAULT_C,
        e_y: float = DEFAULT_E_LATERAL,
        constant_b: bool = False,
        thermal: Optional[ThermalParams] = None,
        pressure: Optional[PressureParams] = None,
        _stacklevel: int = 3,  # internal: where a pressure-only warning should point
    ):
        self.name = name
        self.mu_peak = mu_peak
        self.cornering_stiffness_per_rad = cornering_stiffness_per_rad
        self.longitudinal_stiffness = longitudinal_stiffness
        self.load_sensitivity_n = load_sensitivity_n
        self.f_z0 = f_z0
        self.constant_b = constant_b

        # Only the fixed settings are stored. D (and B unless constant_b is set)
        # depend on the load, so they are computed per request instead.
        self.longitudinal = AxisShape(c=c_x, e=e_x, b=b_x)
        self.lateral = AxisShape(c=c_y, e=e_y, b=b_y)

        # Temperature and pressure, if modeled. The one piece of changing state.
        self._state: Optional[TireState] = None
        if thermal is not None or pressure is not None:
            self._state = TireState(thermal, pressure, stacklevel=_stacklevel)

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_constants(
        cls,
        compound_name: str,
        mu_base: float,
        b_x: float,
        c_x: float,
        e_x: float,
        b_y: Optional[float] = None,
        c_y: Optional[float] = None,
        e_y: Optional[float] = None,
        thermal: Optional[ThermalParams] = None,
        pressure: Optional[PressureParams] = None,
    ) -> "Tire":
        """
        Build a tire from fixed B, C, E constants. Lateral values default to the longitudinal ones.

        With ``thermal`` and/or ``pressure``, stiffness effects work by scaling
        the fixed B values (B is stiffness / (C * D)), so a grip change leaves
        the small-slip stiffness alone, as it does for ``from_macro_properties``.
        """
        return cls(
            compound_name,
            mu_base,
            b_x=b_x,
            c_x=c_x,
            e_x=e_x,
            b_y=b_x if b_y is None else b_y,
            c_y=c_x if c_y is None else c_y,
            e_y=e_x if e_y is None else e_y,
            constant_b=True,
            thermal=thermal,
            pressure=pressure,
            _stacklevel=4,
        )

    @classmethod
    def from_macro_properties(
        cls,
        compound_name: str,
        mu_base: float,
        cornering_stiffness_per_deg: Optional[float] = None,
        longitudinal_stiffness: Optional[float] = None,
        load_sensitivity_n: float = DEFAULT_LOAD_SENSITIVITY,
        f_z0: float = DEFAULT_REFERENCE_LOAD,
        cornering_stiffness_per_rad: Optional[float] = None,
        thermal: Optional[ThermalParams] = None,
        pressure: Optional[PressureParams] = None,
    ) -> "Tire":
        """
        Build a tire from measurable stiffnesses; B is derived from them at each load.

        Cornering stiffness can be given per degree or per radian (exactly one of
        the two). Longitudinal stiffness is required. ``thermal`` and ``pressure``
        optionally give the tire a temperature and pressure (see the class docs).
        """
        if cornering_stiffness_per_deg is None and cornering_stiffness_per_rad is None:
            raise ValueError(
                "Provide cornering stiffness per degree or per radian."
            )
        if longitudinal_stiffness is None:
            raise ValueError("longitudinal_stiffness is required.")

        if cornering_stiffness_per_deg is not None and cornering_stiffness_per_rad is not None:
            raise ValueError(
                "Provide cornering stiffness per degree or per radian, not both."
            )

        # The constructor works in radians only, so convert degrees here
        if cornering_stiffness_per_rad is None:
            cornering_stiffness_per_rad = cornering_stiffness_per_deg * DEG_PER_RAD

        return cls(
            compound_name,
            mu_base,
            cornering_stiffness_per_rad=cornering_stiffness_per_rad,
            longitudinal_stiffness=longitudinal_stiffness,
            load_sensitivity_n=load_sensitivity_n,
            f_z0=f_z0,
            thermal=thermal,
            pressure=pressure,
            _stacklevel=4,
        )

    # ------------------------------------------------------------------
    # Public force calculations
    # ------------------------------------------------------------------

    def calculate_forces(
        self,
        f_z: ArrayLike,
        alpha: ArrayLike,
        kappa: ArrayLike,
        friction_circle: bool = True,
        alpha_unit: Optional[str] = None,
    ) -> Tuple[FloatOrArray, FloatOrArray]:
        """
        Calculate tire forces for a vertical load, slip angle and slip ratio.

        Each of ``f_z``, ``alpha`` and ``kappa`` can be a single number or an
        array, and arrays are broadcast together. With ``friction_circle=True`` both
        forces have the broadcast shape of all three inputs. With
        ``friction_circle=False`` each force only has the shape of its own inputs
        (lateral: ``f_z`` and ``alpha``; longitudinal: ``f_z`` and ``kappa``).

        Args:
            f_z: Vertical load on the tire.
            alpha: Slip angle. See ``alpha_unit`` for the unit.
            kappa: Longitudinal slip ratio.
            friction_circle: If True, limit the combined force to the friction circle.
            alpha_unit: ``"rad"`` or ``"deg"`` to state the unit of ``alpha`` explicitly.
                If None, ``alpha`` is assumed to be radians unless any value has a
                magnitude above ALPHA_RADIAN_LIMIT (pi/2), in which case all of
                ``alpha`` is treated as degrees.

        Returns:
            (lateral_force, longitudinal_force)
        """
        factors = self._factors()  # read once, so every step below sees the same state
        longitudinal, lateral = self._coefficients_at(f_z, factors)

        lateral_force = self._magic_formula_core(
            self._slip_angle_to_radians(alpha, alpha_unit), lateral
        )
        longitudinal_force = self._magic_formula_core(kappa, longitudinal)

        if friction_circle:
            longitudinal_force, lateral_force = self._friction_circle(
                longitudinal_force, lateral_force, f_z, factors
            )

        return lateral_force, longitudinal_force

    def calculate_lateral_force(
        self, f_z: ArrayLike, alpha: ArrayLike, alpha_unit: Optional[str] = None
    ) -> FloatOrArray:
        """
        Calculate the pure lateral force (no friction circle) for a load and slip angle.

        ``f_z`` and ``alpha`` can be single numbers or arrays (broadcast together).
        ``alpha_unit`` is ``"rad"``, ``"deg"`` or None to detect the unit, exactly
        as described in ``calculate_forces``.
        """
        _, lateral = self.coefficients_at(f_z)
        return self._magic_formula_core(self._slip_angle_to_radians(alpha, alpha_unit), lateral)

    # ------------------------------------------------------------------
    # Temperature and pressure
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> Optional[float]:
        """Current temperature in kelvin, or None if the tire has no thermal model."""
        return None if self._state is None else self._state.temperature_k

    @property
    def pressure(self) -> Optional[float]:
        """Current gauge pressure in pascals, or None if the tire has no pressure parameters."""
        return None if self._state is None else self._state.pressure_pa

    def update(
        self,
        dt: float,
        f_z: float,
        alpha: float,
        kappa: float,
        speed: float,
        ambient: float = DEFAULT_AMBIENT_K,
        friction_circle: bool = True,
        alpha_unit: Optional[str] = None,
    ) -> None:
        """
        Advance the tire's temperature (and the pressure that follows from it) by ``dt``.

        Takes the same conditions as ``calculate_forces``, which it uses to find
        the power dissipated by sliding, plus the time step and the speed. The
        conditions are held constant over the step. Does nothing if the tire has
        no thermal model. Forces are never affected by calling this; they only
        change through the state this advances.

        Only single numbers are accepted, since a step describes one moment.
        Forces must be in newtons for the heating to be meaningful.

        Args:
            dt: Time step in seconds.
            f_z: Vertical load in newtons.
            alpha: Slip angle (see ``calculate_forces`` for ``alpha_unit``).
            kappa: Longitudinal slip ratio.
            speed: Speed of the tire over the ground in metres per second.
            ambient: Ambient temperature in kelvin.
            friction_circle: Whether the forces are limited to the friction circle.
            alpha_unit: ``"rad"``, ``"deg"`` or None to detect the unit.
        """
        if self._state is None or not self._state.is_dynamic:
            return

        for name, value in (("dt", dt), ("f_z", f_z), ("alpha", alpha), ("kappa", kappa), ("speed", speed)):
            if np.ndim(value) != 0:
                raise ValueError(f"update() takes single values, but {name} is an array.")

        lateral_force, longitudinal_force = self.calculate_forces(
            f_z, alpha, kappa, friction_circle=friction_circle, alpha_unit=alpha_unit
        )
        alpha_rad = self._slip_angle_to_radians(alpha, alpha_unit)
        speed = abs(speed)
        power = slip_power_w(longitudinal_force, lateral_force, kappa, alpha_rad, speed)
        self._state.update(dt, power, speed, ambient)

    def reset(self) -> None:
        """Return the temperature and pressure to their initial values."""
        if self._state is not None:
            self._state.reset()

    def copy(self) -> "Tire":
        """An independent tire with the same settings, starting from the current state."""
        clone = _copy.copy(self)
        if self._state is not None:
            clone._state = self._state.copy()
        return clone

    # ------------------------------------------------------------------
    # Coefficient calculations
    # ------------------------------------------------------------------

    def coefficients_at(
        self, f_z: ArrayLike
    ) -> Tuple[MagicFormulaCoefficients, MagicFormulaCoefficients]:
        """
        Build the full B, C, D, E coefficients for each axis at a vertical load.

        Nothing is stored on the tire, so this has no side effects and is safe to
        call from several threads at once. If the tire has a temperature and
        pressure, their current values are applied.

        Returns:
            (longitudinal, lateral)
        """
        return self._coefficients_at(f_z, self._factors())

    def _coefficients_at(
        self, f_z: ArrayLike, factors: TireFactors
    ) -> Tuple[MagicFormulaCoefficients, MagicFormulaCoefficients]:
        """``coefficients_at`` with explicit factors, so one request uses one state."""
        d = self._calculate_d(f_z, factors)

        if self.constant_b:
            # B is fixed, so a stiffness factor scales B directly. Stiffness is
            # B * C * D, so B is also divided by the grip factor that scales D.
            b_longitudinal = self.longitudinal.b * (
                factors.longitudinal_stiffness / factors.grip
            )
            b_lateral = self.lateral.b * (factors.cornering_stiffness / factors.grip)
        else:
            b_longitudinal, b_lateral = self._calculate_b(f_z, d, factors)

        longitudinal = MagicFormulaCoefficients(
            b=b_longitudinal, c=self.longitudinal.c, d=d, e=self.longitudinal.e
        )
        lateral = MagicFormulaCoefficients(
            b=b_lateral, c=self.lateral.c, d=d, e=self.lateral.e
        )
        return longitudinal, lateral

    def calculate_d(self, f_z: ArrayLike) -> FloatOrArray:
        """
        Return the peak factor D (peak force) at a vertical load.

        Assumes the same coefficient of friction in both directions, so the
        value applies to both axes.

        A tire carrying no load (or lifted off the ground, f_z <= 0) can produce
        no force. This is returned directly, because the load sensitivity term
        raises zero to a negative power and cannot be evaluated there. For an
        array of loads this is applied to each element separately.

        If ``load_sensitivity_n`` is None, load sensitivity is skipped and the
        friction coefficient is treated as constant (D = mu_peak * f_z).
        """
        return self._calculate_d(f_z, self._factors())

    def _calculate_d(self, f_z: ArrayLike, factors: TireFactors) -> FloatOrArray:
        """``calculate_d`` with explicit factors, so one request uses one state."""
        mu_peak, _, _, n = self._effective_parameters(factors)

        if isinstance(f_z, _SCALAR):  # fast path for a single load
            if f_z <= 0:
                return 0.0
            if n is None:
                return mu_peak * f_z
            return mu_peak * pow(self._load_ratio(f_z), n - 1) * f_z

        f_z = np.asarray(f_z, dtype=float)
        loaded = f_z > 0
        # Swap unloaded points for 1.0 so the power below never sees 0 or a negative
        safe_f_z = np.where(loaded, f_z, 1.0)

        if n is None:
            mu_effective = mu_peak
        else:
            # Peak friction falls off as load rises above the reference load
            mu_effective = mu_peak * pow(self._load_ratio(safe_f_z), n - 1)
        return np.where(loaded, mu_effective * safe_f_z, 0.0)[()]

    def calculate_b(
        self, f_z: ArrayLike, d: FloatOrArray
    ) -> Tuple[FloatOrArray, FloatOrArray]:
        """
        Return the stiffness factor B for each axis, from the cornering and
        longitudinal stiffness at a vertical load.

        Args:
            f_z: Vertical load.
            d: Peak factor D at this load.

        Returns:
            (b_longitudinal, b_lateral)
        """
        return self._calculate_b(f_z, d, self._factors())

    def _calculate_b(
        self, f_z: ArrayLike, d: FloatOrArray, factors: TireFactors
    ) -> Tuple[FloatOrArray, FloatOrArray]:
        """``calculate_b`` with explicit factors, so one request uses one state."""
        _, cornering_stiffness, longitudinal_stiffness, n = self._effective_parameters(factors)

        # Unloaded points have d == 0, which already forces B to 0 below
        if isinstance(f_z, _SCALAR):
            safe_f_z = f_z if f_z > 0 else 1.0
        else:
            f_z = np.asarray(f_z, dtype=float)
            safe_f_z = np.where(f_z > 0, f_z, 1.0)
        load_scaling = self._load_ratio(safe_f_z) ** n

        # Lateral: B needs the cornering stiffness per radian
        cornering_stiffness_rad = cornering_stiffness * load_scaling
        b_lateral = self._stiffness_to_b(cornering_stiffness_rad, self.lateral.c, d)

        # Longitudinal
        longitudinal_stiffness = longitudinal_stiffness * load_scaling
        b_longitudinal = self._stiffness_to_b(longitudinal_stiffness, self.longitudinal.c, d)

        return b_longitudinal, b_lateral

    def friction_circle(
        self, fx_raw: ArrayLike, fy_raw: ArrayLike, f_z: ArrayLike
    ) -> Tuple[FloatOrArray, FloatOrArray]:
        """
        Limit the combined force to the friction circle. The radius is the peak
        force at this load, so it includes load sensitivity (same as D).

        Args:
            fx_raw: Unconstrained longitudinal force.
            fy_raw: Unconstrained lateral force.
            f_z: Vertical load.

        Returns:
            (fx_final, fy_final)
        """
        return self._friction_circle(fx_raw, fy_raw, f_z, self._factors())

    def _friction_circle(
        self, fx_raw: ArrayLike, fy_raw: ArrayLike, f_z: ArrayLike, factors: TireFactors
    ) -> Tuple[FloatOrArray, FloatOrArray]:
        """``friction_circle`` with explicit factors, so one request uses one state."""
        if isinstance(fx_raw, _SCALAR) and isinstance(fy_raw, _SCALAR) and isinstance(f_z, _SCALAR):
            f_max = self._calculate_d(f_z, factors)  # fast path for single values
            magnitude = np.sqrt(fx_raw ** 2 + fy_raw ** 2)
            if magnitude > f_max and magnitude > 0:
                scale = f_max / magnitude
                return fx_raw * scale, fy_raw * scale
            return fx_raw, fy_raw

        fx_raw = np.asarray(fx_raw, dtype=float)
        fy_raw = np.asarray(fy_raw, dtype=float)
        f_max = self._calculate_d(f_z, factors)
        magnitude = np.sqrt(fx_raw ** 2 + fy_raw ** 2)

        # Where total demand exceeds physical grip, scale back to the circle boundary
        over = (magnitude > f_max) & (magnitude > 0)
        scale = np.where(over, f_max / np.where(over, magnitude, 1.0), 1.0)
        return (fx_raw * scale)[()], (fy_raw * scale)[()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _factors(self) -> TireFactors:
        """The factors for the tire's current state (no change if it has none)."""
        return IDENTITY if self._state is None else self._state.factors()

    def _effective_parameters(
        self, factors: TireFactors
    ) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
        """
        The tire parameters with the factors applied.

        Returns:
            (mu_peak, cornering_stiffness_per_rad, longitudinal_stiffness, load_sensitivity_n).
            The last three are None where the tire does not define them.
        """
        if factors is IDENTITY:
            return (
                self.mu_peak,
                self.cornering_stiffness_per_rad,
                self.longitudinal_stiffness,
                self.load_sensitivity_n,
            )

        cornering = self.cornering_stiffness_per_rad
        if cornering is not None:
            cornering = cornering * factors.cornering_stiffness
        longitudinal = self.longitudinal_stiffness
        if longitudinal is not None:
            longitudinal = longitudinal * factors.longitudinal_stiffness
        n = self.load_sensitivity_n
        if n is not None:
            # The exponent may not be pushed above 1 (grip rising with load) by a factor
            n = min(n * factors.load_sensitivity, max(n, 1.0))
        return self.mu_peak * factors.grip, cornering, longitudinal, n

    def _load_ratio(self, f_z: FloatOrArray) -> FloatOrArray:
        """Vertical load relative to the reference load."""
        return f_z / self.f_z0

    @staticmethod
    def _slip_angle_to_radians(
        alpha: ArrayLike, alpha_unit: Optional[str] = None
    ) -> FloatOrArray:
        """
        Return the slip angle in radians.

        Args:
            alpha: Slip angle, a single value or an array.
            alpha_unit: ``"rad"`` (no conversion), ``"deg"`` (always convert), or
                None to detect the unit. Detection makes one decision for the
                whole input: if any value is beyond ALPHA_RADIAN_LIMIT, all of it
                is treated as degrees, so a sweep is never half degrees, half radians.
        """
        if alpha_unit is not None and alpha_unit not in ALPHA_UNITS:
            raise ValueError(f"alpha_unit must be one of {ALPHA_UNITS} or None, got {alpha_unit!r}")

        is_scalar = isinstance(alpha, _SCALAR)
        if not is_scalar:
            alpha = np.asarray(alpha, dtype=float)

        if alpha_unit == "rad":
            return alpha
        if alpha_unit == "deg":
            return np.radians(alpha)

        exceeds_limit = abs(alpha) > ALPHA_RADIAN_LIMIT if is_scalar else np.any(
            np.abs(alpha) > ALPHA_RADIAN_LIMIT
        )
        return np.radians(alpha) if exceeds_limit else alpha

    @staticmethod
    def _stiffness_to_b(stiffness: FloatOrArray, c: float, d: FloatOrArray) -> FloatOrArray:
        """Convert a stiffness into the magic formula B factor (B = stiffness / (C * D))."""
        if isinstance(d, _SCALAR):  # fast path for a single load
            return stiffness / (c * d) if d > 0 else 0
        d = np.asarray(d, dtype=float)
        positive = d > 0
        safe_d = np.where(positive, d, 1.0)
        return np.where(positive, stiffness / (c * safe_d), 0.0)[()]

    @staticmethod
    def _magic_formula_core(x_input: ArrayLike, coeffs: MagicFormulaCoefficients) -> FloatOrArray:
        """
        Evaluate the magic formula: D * sin(C * arctan(Bx - E * (Bx - arctan(Bx)))).

        Args:
            x_input: Slip input (slip angle or slip ratio).
            coeffs: The B, C, D and E coefficients for this axis and load.
        """
        if not isinstance(x_input, _SCALAR):
            x_input = np.asarray(x_input, dtype=float)

        bx = coeffs.b * x_input
        argument = coeffs.c * np.arctan(bx - coeffs.e * (bx - np.arctan(bx)))
        return coeffs.d * np.sin(argument)
