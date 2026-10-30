"""Shared pytest fixtures, plus saving of sweep plots."""

from pathlib import Path

import pytest

from pacejka import PressureParams, ThermalParams, Tire


def pytest_addoption(parser):
    parser.addoption(
        "--sweep-dir",
        action="store",
        default=None,
        help="Folder where sweep plots are saved (default: <project>/sweep_output).",
    )


def pytest_configure(config):
    config._saved_sweeps = []


def pytest_terminal_summary(terminalreporter, config):
    saved = getattr(config, "_saved_sweeps", [])
    if saved:
        terminalreporter.write_sep("-", "sweep plots saved")
        for path in saved:
            terminalreporter.write_line(str(path))


# ----------------------------------------------------------------------
# Tires used across the tests
# ----------------------------------------------------------------------

# Peak friction 1.6, 25 force-units per degree of slip angle, 2000 per unit slip
# ratio, measured at a reference load of 150 with a load sensitivity of 0.9.
MU = 1.6
CORNERING_PER_DEG = 25.0
LONGITUDINAL = 2000.0
F_Z0 = 150.0
LOAD_SENSITIVITY = 0.9


@pytest.fixture
def r55a():
    """A tire built from macro properties (load sensitive, B derived per load)."""
    return Tire.from_macro_properties(
        compound_name="Test R55A",
        mu_base=MU,
        cornering_stiffness_per_deg=CORNERING_PER_DEG,
        longitudinal_stiffness=LONGITUDINAL,
        load_sensitivity_n=LOAD_SENSITIVITY,
        f_z0=F_Z0,
    )


@pytest.fixture
def constant_tire():
    """A tire built from fixed B, C, E constants (no load sensitivity)."""
    return Tire.from_constants(
        "Constant", mu_base=1.5, b_x=8.0, c_x=1.4, e_x=0.1, b_y=9.0, c_y=1.3, e_y=-0.2
    )


# A small kart tire in SI units (newtons), used by the temperature and pressure tests.
# Its thermal and pressure settings are the (placeholder) defaults.
KART_MU = 1.3
KART_CORNERING_PER_DEG = 400.0
KART_LONGITUDINAL = 9000.0
KART_F_Z0 = 500.0
KART_LOAD_SENSITIVITY = 0.9


def make_kart_tire(thermal=None, pressure=None, **overrides):
    """A macro-property kart tire in newtons, with optional thermal and pressure models."""
    settings = dict(
        compound_name="Test kart",
        mu_base=KART_MU,
        cornering_stiffness_per_deg=KART_CORNERING_PER_DEG,
        longitudinal_stiffness=KART_LONGITUDINAL,
        load_sensitivity_n=KART_LOAD_SENSITIVITY,
        f_z0=KART_F_Z0,
    )
    settings.update(overrides)
    return Tire.from_macro_properties(thermal=thermal, pressure=pressure, **settings)


@pytest.fixture
def kart_base():
    """The kart tire with no temperature or pressure model."""
    return make_kart_tire()


@pytest.fixture
def kart_tire():
    """The kart tire with thermal and pressure models, starting cold."""
    return make_kart_tire(ThermalParams(), PressureParams())


# ----------------------------------------------------------------------
# Sweep plots
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def sweep_dir(pytestconfig):
    option = pytestconfig.getoption("--sweep-dir")
    path = Path(option) if option else pytestconfig.rootpath / "sweep_output"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def plt():
    """matplotlib's pyplot with a non-interactive backend, or skip if not installed."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


@pytest.fixture
def save_sweep(plt, sweep_dir, pytestconfig):
    """Return a function that saves a figure into the sweep folder as <name>.png."""

    def _save(fig, name):
        path = sweep_dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        pytestconfig._saved_sweeps.append(path)
        return path

    return _save
