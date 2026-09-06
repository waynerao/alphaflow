from datetime import date

import pandas as pd
import pytest

from alphaflow.core.alpha_base.mid_alpha import MidAlpha
from alphaflow.core.config.alpha_config.mid_config import MidAlphaConfig
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.loader import load_alpha_config

# No concrete MidAlpha ships yet (the spec gives no example), so these cover the base
# contract every future mid alpha inherits.


class SampleMidAlpha(MidAlpha):
    def compute(self, market, as_of_date):
        return self.working_data.set_index("RIC")["value"]


@pytest.fixture
def config():
    return load_alpha_config("configs/alphas/mid/_template.toml")


class TestMidAlphaBase:
    def test_template_config_is_a_mid_config_with_an_interval(self, config):
        assert isinstance(config, MidAlphaConfig) and config.interval_minutes == 5

    def test_expected_type_is_enforced(self, config):
        assert SampleMidAlpha(config, None).alpha_type == "mid"

    def test_a_low_config_cannot_be_loaded_as_mid(self):
        low = load_alpha_config("configs/alphas/low/momentum_price_21d.toml")
        with pytest.raises(ConfigValidationError, match="loaded as 'mid'"):
            SampleMidAlpha(low, None)

    def test_identity_normalization_by_default(self, config):
        alpha = SampleMidAlpha(config, None)
        alpha._working_data = pd.DataFrame({"RIC": ["A.HK", "B.HK"], "value": [2.0, 4.0]})
        assert list(alpha.run("HK", date(2024, 1, 1))) == [2.0, 4.0]

    def test_interval_must_be_positive(self):
        with pytest.raises(Exception):
            MidAlphaConfig(name="m", description="d", category="c", formula="f", handler_class="m.C",
                           alpha_type="mid", market_scope=["HK"], required_inputs=["minute_bar"], interval_minutes=0)
