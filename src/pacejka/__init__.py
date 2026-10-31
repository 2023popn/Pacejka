"""pacejka: tire force calculations using the pacejka magic formula."""

from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("pacejka")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "unknown"

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
