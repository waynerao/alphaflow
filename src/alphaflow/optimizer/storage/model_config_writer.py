import re
from datetime import date
from pathlib import Path

import pandas as pd

from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.paths import DEFAULT_MODELS_DIR, resolve_config_path

WEIGHTS_SECTION = "[alpha_weights]"
SECTION_PATTERN = re.compile(r"^\s*\[")
WEIGHT_PRECISION = 6


class ModelConfigWriter:
    """Rewrites only the [alpha_weights] table in configs/models/{model_id}.toml.

    Done as a line-level edit rather than a parse-and-dump so that comments, key ordering and
    formatting elsewhere in the file survive untouched - the model TOML is hand-maintained,
    and the optimizer only owns this one section.
    """

    def write_weights(self, model_id: str, models_dir: str = DEFAULT_MODELS_DIR, weights: pd.Series | None = None) -> str:
        if weights is None or weights.empty:
            raise ValueError(f"no weights to write for model {model_id!r}")
        path = Path(resolve_config_path(models_dir)) / f"{model_id}.toml"
        if not path.is_file():
            raise ConfigValidationError(f"model config not found: {path}")
        original = path.read_text(encoding="utf-8")
        path.write_text(self._replace_section(original, weights, model_id), encoding="utf-8")
        return str(path)

    @classmethod
    def _replace_section(cls, content: str, weights: pd.Series, model_id: str) -> str:
        lines = content.splitlines()
        start = next((i for i, line in enumerate(lines) if line.strip() == WEIGHTS_SECTION), None)
        block = cls._render(weights, model_id)
        if start is None:
            # No section yet - append one, keeping a blank line before it
            tail = "" if content.endswith("\n\n") else ("\n" if content.endswith("\n") else "\n\n")
            return content + tail + "\n".join(block) + "\n"
        end = next((i for i in range(start + 1, len(lines)) if SECTION_PATTERN.match(lines[i])), len(lines))
        return "\n".join(lines[:start] + block + lines[end:]) + "\n"

    @classmethod
    def _render(cls, weights: pd.Series, model_id: str) -> list[str]:
        width = max((len(str(k)) for k in weights.index), default=0)
        body = [f"{str(alpha_id):<{width}} = {round(float(value), WEIGHT_PRECISION)}" for alpha_id, value in weights.items()]
        return [WEIGHTS_SECTION, f"# Written by OptimizerRunner for {model_id} on {date.today():%Y-%m-%d}. Do not hand-edit.", *body, ""]
