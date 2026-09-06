import pandas as pd

from alphaflow.strategy_backtester.score_loader import DATE_COLUMN, ScoreLoader


class StrategyReturnBuilder:
    """Turns each alpha into a daily return series, so the optimizer can treat it as a
    standalone strategy.

        daily_return(t) = sum_i( score_i(t) * return_i(t) )  over matched stocks

    No dollar-neutral rescaling: if you want that, point the model at a hedged rtn_alpha_id.
    """

    def build(self, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame) -> pd.Series:
        aligned = ScoreLoader.align(alpha_scores, return_scores)
        if aligned.empty:
            return pd.Series(dtype=float, name="daily_return")
        contribution = aligned["score"] * aligned["forward_return"]
        return contribution.groupby(aligned[DATE_COLUMN]).sum().sort_index().rename("daily_return")

    def build_for_model(self, alpha_ids: list[str], market: str, rtn_alpha_id: str, dates: list[str],
                        score_loader: ScoreLoader) -> pd.DataFrame:
        """One column per alpha, indexed by date. Dates where an alpha has no data become
        NaN and are dropped, so the covariance matrix is built on a common window."""
        return_scores = score_loader.load_return_scores(market, rtn_alpha_id, dates)
        if return_scores.empty:
            raise FileNotFoundError(f"no stored return scores for {rtn_alpha_id} in {market} over the requested dates")
        series = {}
        for alpha_id in alpha_ids:
            alpha_scores = score_loader.load_alpha_scores(market, alpha_id, dates)
            if alpha_scores.empty:
                raise FileNotFoundError(f"no stored scores for {alpha_id} in {market} over the requested dates")
            series[alpha_id] = self.build(alpha_scores, return_scores)
        frame = pd.DataFrame(series).sort_index()
        return frame.dropna(how="any")
