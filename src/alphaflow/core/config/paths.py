import os
from pathlib import Path

PROJECT_ROOT_MARKER = "pyproject.toml"
SHARED_DRIVE_ENV_VAR = "ALPHAFLOW_SHARED_DRIVE"

DEFAULT_SYSTEM_CONFIG = "configs/system.toml"
DEFAULT_PRODUCTION_CONFIG = "configs/production.toml"
DEFAULT_OPTIMIZER_CONFIG = "configs/optimizer.toml"
DEFAULT_REGISTRY_CONFIG = "configs/registry.toml"
DEFAULT_MODELS_DIR = "configs/models/"
DEFAULT_UNIVERSE_DIR = "configs/universe/"


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk up from start (default: this file) until pyproject.toml is found."""
    current = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / PROJECT_ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(f"no {PROJECT_ROOT_MARKER} found at or above {current}")


def resolve_config_path(path: str | Path) -> Path:
    """Absolute paths pass through; relative ones anchor to the project root, not the CWD.

    This is what lets a notebook in notebooks/alpha_research/ call
    SignalBuilderRunner.from_config() with no arguments.
    """
    p = Path(path)
    return p if p.is_absolute() else find_project_root() / p


def resolve_shared_drive(configured_path: str) -> Path:
    """$ALPHAFLOW_SHARED_DRIVE beats system.toml, so a dev box can redirect all reads and
    writes without editing a committed config file. A relative value anchors to the project
    root, not the CWD - otherwise a notebook and a script would disagree about where the
    shared drive is."""
    path = Path(os.environ.get(SHARED_DRIVE_ENV_VAR) or configured_path).expanduser()
    return path if path.is_absolute() else find_project_root() / path
