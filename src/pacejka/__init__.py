"""pacejka: tire force calculations using the pacejka magic formula."""

from .tire import AxisShape, MagicFormulaCoefficients, Tire
from .tire_state import (
    PressureParams,
    PressureWithoutThermalWarning,
    ThermalParams,
    TireFactors,
    TireSnapshot,
    TireState,
    combine_factors,
    pressure_factors,
    thermal_factors,
)

__version__ = "0.2.1"

__all__ = [
    "AxisShape",
    "MagicFormulaCoefficients",
    "PressureParams",
    "PressureWithoutThermalWarning",
    "ThermalParams",
    "Tire",
    "TireFactors",
    "TireSnapshot",
    "TireState",
    "combine_factors",
    "pressure_factors",
    "thermal_factors",
    "__version__",
]
