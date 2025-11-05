"""
Cross-Asset Lead–Lag Network package.

Public API:
- Graph construction and utilities: see `cross_asset_leadlag.graph`
- Strategy & backtest: see `cross_asset_leadlag.algo`
- Statistical validation utilities: see `cross_asset_leadlag.validation`
"""

from .algo import (
    leader_scores,
    ema_update,
    momentum_gate_binary,
    momentum_signal_continuous,
    ma_cross_regime_gate,
    ma50_rising_gate,
    ma_strength_continuous,
    sparsify_topk_outgoing,
    combine_leader_momentum_longonly,
    combine_leader_momentum_continuous,
    scale_to_target_vol,
    transaction_cost,
    backtest_network_momentum,
)
from .graph import (
    MAX_LAG,
    corr_at_lag,
    best_lag_corr,
    build_adj,
    build_adj_fast,
    leadlag_graph,
    filter_edges_fdr,
)
from .validation import walkforward_validation, sharpe_ratio

__all__ = [
    # graph
    "MAX_LAG",
    "corr_at_lag",
    "best_lag_corr",
    "build_adj",
    "build_adj_fast",
    "leadlag_graph",
    "filter_edges_fdr",
    # algo
    "leader_scores",
    "ema_update",
    "momentum_gate_binary",
    "momentum_signal_continuous",
    "ma_cross_regime_gate",
    "ma50_rising_gate",
    "ma_strength_continuous",
    "sparsify_topk_outgoing",
    "combine_leader_momentum_longonly",
    "combine_leader_momentum_continuous",
    "scale_to_target_vol",
    "transaction_cost",
    "backtest_network_momentum",
    # validation
    "walkforward_validation",
    "sharpe_ratio",
]
