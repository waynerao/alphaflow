"""AlphaFlow - alpha research, calculation and post-trade reconciliation for APAC equities.

Each package exposes exactly one Runner, called directly from notebooks or from another
package. There is no CLI.

    DataDumperRunner          manual raw-data backfill to the shared drive
    SignalBuilderRunner       computes alpha scores (research and production share this)
    StrategyBacktesterRunner  single-alpha performance against one or more return definitions
    OptimizerRunner           mean-variance weights across the alphas in a model
    AlphaAggregatorRunner     production producer threads + on-demand model consumer
    PostTradeAnalyzerRunner   predicted vs realised, attributed to Barra factors
"""

__version__ = "0.1.0"

__all__ = [
    "DataDumperRunner", "SignalBuilderRunner", "StrategyBacktesterRunner",
    "OptimizerRunner", "AlphaAggregatorRunner", "PostTradeAnalyzerRunner",
]


def __getattr__(name: str):
    # Lazy so that importing alphaflow does not pull in scipy, statsmodels and apscheduler
    if name == "DataDumperRunner":
        from alphaflow.data_dumper.runner import DataDumperRunner
        return DataDumperRunner
    if name == "SignalBuilderRunner":
        from alphaflow.signal_builder.runner import SignalBuilderRunner
        return SignalBuilderRunner
    if name == "StrategyBacktesterRunner":
        from alphaflow.strategy_backtester.runner import StrategyBacktesterRunner
        return StrategyBacktesterRunner
    if name == "OptimizerRunner":
        from alphaflow.optimizer.runner import OptimizerRunner
        return OptimizerRunner
    if name == "AlphaAggregatorRunner":
        from alphaflow.alpha_aggregator.runner import AlphaAggregatorRunner
        return AlphaAggregatorRunner
    if name == "PostTradeAnalyzerRunner":
        from alphaflow.post_trade_analyzer.runner import PostTradeAnalyzerRunner
        return PostTradeAnalyzerRunner
    raise AttributeError(f"module 'alphaflow' has no attribute {name!r}")
