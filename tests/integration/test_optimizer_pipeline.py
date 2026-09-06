import shutil
import tomllib

import pytest

from alphaflow.core.config.loader import load_model_config
from alphaflow.optimizer.runner import OptimizerRunner
from alphaflow.optimizer.storage.result_writer import PORTFOLIO_ROW
from alphaflow.signal_builder.runner import SignalBuilderRunner

MARKET = "HK"
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


@pytest.fixture
def models_dir(tmp_path, scoring_days):
    """A model whose window matches the scored dates, in a writable directory."""
    directory = tmp_path / "models"
    directory.mkdir()
    source = (directory / "hk_momentum.toml")
    shutil.copy("configs/models/hk_momentum.toml", source)
    content = source.read_text()
    content = content.replace('start_date      = "20230101"', f'start_date      = "{min(scoring_days):%Y%m%d}"')
    content = content.replace('end_date        = "20231231"', f'end_date        = "{max(scoring_days):%Y%m%d}"')
    source.write_text(content)
    return directory


@pytest.fixture
def prepared(system_config, registry, raw_data_window, scoring_days, date_strings, models_dir):
    raw_data_window(scoring_days)
    builder = SignalBuilderRunner(system_config, registry)
    for alpha_id in (MOMENTUM, IMBALANCE, RETURN):
        builder.build_multi_day_alpha(alpha_id, MARKET, date_strings(scoring_days), save=True)
    registry.models["hk_momentum"] = load_model_config(models_dir / "hk_momentum.toml")
    registry.models_dir = str(models_dir)
    return registry


class TestOptimizerPipeline:
    """alpha scores on disk -> weights written back to the model TOML + an audit table."""

    def test_weights_are_written_back_into_the_model_toml(self, system_config, optimizer_config, prepared,
                                                          models_dir, research_pit, mocker):
        mocker.patch("alphaflow.optimizer.runner.PITDataManager.for_research", return_value=research_pit())
        OptimizerRunner(system_config, optimizer_config, prepared, str(models_dir)).run("hk_momentum")
        parsed = tomllib.loads((models_dir / "hk_momentum.toml").read_text())
        assert set(parsed["alpha_weights"]) == {MOMENTUM, IMBALANCE}
        assert sum(parsed["alpha_weights"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_the_updated_toml_still_validates_as_a_model_config(self, system_config, optimizer_config, prepared,
                                                                models_dir, research_pit, mocker):
        mocker.patch("alphaflow.optimizer.runner.PITDataManager.for_research", return_value=research_pit())
        OptimizerRunner(system_config, optimizer_config, prepared, str(models_dir)).run("hk_momentum")
        reloaded = load_model_config(models_dir / "hk_momentum.toml")
        assert reloaded.is_optimized and reloaded.rtn_alpha_id == RETURN

    def test_the_audit_table_has_a_row_per_alpha_plus_portfolio(self, system_config, optimizer_config, prepared,
                                                                models_dir, research_pit, mocker):
        mocker.patch("alphaflow.optimizer.runner.PITDataManager.for_research", return_value=research_pit())
        table = OptimizerRunner(system_config, optimizer_config, prepared, str(models_dir)).run("hk_momentum")["hk_momentum"]
        assert list(table["row_id"]) == [MOMENTUM, IMBALANCE, PORTFOLIO_ROW]
        assert any(c.startswith("barra_") for c in table.columns)

    def test_the_result_is_saved_to_the_spec_path(self, system_config, optimizer_config, prepared, models_dir,
                                                  research_pit, mocker, isolated_shared_drive):
        mocker.patch("alphaflow.optimizer.runner.PITDataManager.for_research", return_value=research_pit())
        OptimizerRunner(system_config, optimizer_config, prepared, str(models_dir)).run("hk_momentum")
        assert (isolated_shared_drive / "optimizer_results" / MARKET / "hk_momentum.parquet").is_file()

    def test_rerunning_refuses_to_clobber_without_overwrite(self, system_config, optimizer_config, prepared,
                                                             models_dir, research_pit, mocker):
        mocker.patch("alphaflow.optimizer.runner.PITDataManager.for_research", return_value=research_pit())
        runner = OptimizerRunner(system_config, optimizer_config, prepared, str(models_dir))
        runner.run("hk_momentum")
        with pytest.raises(FileExistsError):
            runner.run("hk_momentum")
        runner.run("hk_momentum", overwrite=True)
