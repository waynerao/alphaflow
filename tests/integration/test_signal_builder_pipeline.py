

from alphaflow.signal_builder.runner import SignalBuilderRunner
from alphaflow.signal_builder.storage.score_writer import ScoreWriter

MARKET = "HK"
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


class TestSignalBuilderPipeline:
    """data_dumper CSVs -> alpha scores on the shared drive."""

    def test_all_three_alpha_types_compute_and_persist(self, system_config, registry, raw_data_window,
                                                       scoring_days, date_strings):
        raw_data_window(scoring_days)
        runner = SignalBuilderRunner(system_config, registry)
        writer = ScoreWriter(system_config.storage)
        for alpha_id in (MOMENTUM, IMBALANCE, RETURN):
            result = runner.build_multi_day_alpha(alpha_id, MARKET, date_strings(scoring_days), save=True)
            assert not result[alpha_id].empty
            assert writer.file_exists(MARKET, alpha_id, scoring_days[0])

    def test_stored_scores_carry_the_spec_columns(self, system_config, registry, raw_data_window,
                                                 scoring_days, date_strings):
        raw_data_window(scoring_days)
        SignalBuilderRunner(system_config, registry).build_multi_day_alpha(
            MOMENTUM, MARKET, date_strings(scoring_days), save=True)
        stored = ScoreWriter(system_config.storage).read(MARKET, MOMENTUM, scoring_days[0])
        assert list(stored.columns) == ["RIC", "raw_signal", "score"] and not stored.empty

    def test_building_all_alphas_at_once_covers_the_active_registry(self, system_config, registry,
                                                                    raw_data_window, scoring_days, date_strings):
        raw_data_window(scoring_days)
        runner = SignalBuilderRunner(system_config, registry)
        result = runner.build_one_day_alpha("all", MARKET, date_strings(scoring_days)[0])
        assert set(result) == {MOMENTUM, IMBALANCE, RETURN}

    def test_a_second_save_is_skipped_and_returns_the_stored_frame(self, system_config, registry,
                                                                   raw_data_window, scoring_days, date_strings):
        raw_data_window(scoring_days)
        runner = SignalBuilderRunner(system_config, registry)
        day = date_strings(scoring_days)[0]
        first = runner.build_one_day_alpha(MOMENTUM, MARKET, day, save=True)[MOMENTUM]
        second = runner.build_one_day_alpha(MOMENTUM, MARKET, day, save=True)[MOMENTUM]
        assert set(first["RIC"]) == set(second["RIC"])
