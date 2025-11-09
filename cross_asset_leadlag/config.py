from dataclasses import dataclass


@dataclass
class BacktestConfig:
    window: int = 252  # lookback for network
    max_lag: int = 5
    min_abs_corr: float = 0.15
    rebalance: str = "M"  # "M" (monthly) or "W-FRI"
    mom_lookback: int = 50
    ema_span: int | None = 20  # None => no smoothing
    target_ann_vol: float | None = 0.10  # None => no vol targeting
    tc_bps: float = 3.0  # transaction cost bps per side
    cov_win: int = 60  # cov window for vol targeting
    leader_method: str = (
        "out_strength"  # e.g., "out_strength", "abs_out_strength", "pos_only", "sign_aware_pagerank", "pagerank"
    )
    pagerank_alpha: float = 0.9
    sparsify_topk: int | None = None
    regime: str = "price_sma"  # "price_sma", "ma_cross", "ma_slope", "ma_strength"
    fast_ma: int = 50
    slow_ma: int = 200
    slope_win: int = 10
    shrink_lambda: float | None = None
    use_numba: bool = False
