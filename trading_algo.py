import networkx as nx
import numpy as np
import pandas as pd

from lead_lag_graph import (
    build_adj,
    build_adj_fast,
    filter_edges_fdr,
)

MAX_LAG = 5


def leader_scores(
    A: pd.DataFrame, method: str = "out_strength", pagerank_alpha: float = 0.9
) -> pd.Series:
    if A.empty:
        return pd.Series(dtype=float)

    method = (method or "out_strength").lower()
    if method == "pagerank":
        # Standard PageRank on signed matrix can misbehave; assume weights are nonnegative
        G = nx.from_pandas_adjacency(A.clip(lower=0.0), create_using=nx.DiGraph)
        if G.number_of_edges() == 0:
            s = pd.Series(0.0, index=A.index)
        else:
            # Provide stable defaults and robust fallbacks to avoid convergence failures
            try:
                n_nodes = G.number_of_nodes()
                dangling = {node: 1.0 / n_nodes for node in G}
            except Exception:
                dangling = None
            try:
                pr = nx.pagerank(
                    G,
                    alpha=pagerank_alpha,
                    weight="weight",
                    max_iter=2000,
                    tol=1.0e-6,
                    dangling=dangling,
                )
            except nx.PowerIterationFailedConvergence:
                # Fallback 1: numpy-based PageRank (eigenvector solver)
                try:
                    pr = nx.pagerank_numpy(G, alpha=pagerank_alpha, weight="weight")
                except Exception:
                    # Fallback 2: use normalized positive out-strengths
                    s_out = A.clip(lower=0.0).sum(axis=1)
                    total = float(s_out.sum()) if np.isfinite(s_out.sum()) else 0.0
                    if total <= 0.0:
                        s = pd.Series(0.0, index=A.index)
                    else:
                        s = (s_out / total).reindex(A.index).fillna(0.0)
                    pr = s.to_dict()
            s = pd.Series(pr).reindex(A.index).fillna(0.0)
    elif method == "abs_out_strength":
        s = A.abs().sum(axis=1)
    elif method == "pos_only":
        s = A.clip(lower=0.0).sum(axis=1)
    elif method == "sign_aware_pagerank":
        # Influence via |A|, orientation from sign of row-sum of A
        G = nx.from_pandas_adjacency(A.abs(), create_using=nx.DiGraph)
        if G.number_of_edges() == 0:
            infl = pd.Series(0.0, index=A.index)
        else:
            # Use robust PageRank with fallbacks as above
            try:
                n_nodes = G.number_of_nodes()
                dangling = {node: 1.0 / n_nodes for node in G}
            except Exception:
                dangling = None
            try:
                pr = nx.pagerank(
                    G,
                    alpha=pagerank_alpha,
                    weight="weight",
                    max_iter=2000,
                    tol=1.0e-6,
                    dangling=dangling,
                )
            except nx.PowerIterationFailedConvergence:
                try:
                    pr = nx.pagerank_numpy(G, alpha=pagerank_alpha, weight="weight")
                except Exception:
                    # Fallback: degree-based influence on |A|
                    s_out = A.abs().sum(axis=1)
                    total = float(s_out.sum()) if np.isfinite(s_out.sum()) else 0.0
                    if total <= 0.0:
                        infl = pd.Series(0.0, index=A.index)
                        pr = infl.to_dict()
                    else:
                        infl = (s_out / total).reindex(A.index).fillna(0.0)
                        pr = infl.to_dict()
            infl = pd.Series(pr).reindex(A.index).fillna(0.0)
        sign = A.sum(axis=1).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
        s = infl * sign
    else:
        # Default: signed out-strength (row sum)
        s = A.sum(axis=1)

    # z-score (avoid div-by-zero)
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        z = s - mu
    else:
        z = (s - mu) / sd
    z.name = "leader_z"
    return z


def ema_update(
    prev_smoothed: pd.Series | None, new_z: pd.Series, span: int = 20
) -> pd.Series:
    if prev_smoothed is None:
        return new_z.fillna(0.0)

    # align index
    idx = new_z.index.union(prev_smoothed.index)
    prev = prev_smoothed.reindex(idx).fillna(0.0)
    cur = new_z.reindex(idx).fillna(0.0)

    alpha = 2.0 / (1.0 + span)
    smoothed = alpha * cur + (1 - alpha) * prev
    return smoothed.reindex(new_z.index)


# --- binary gate: 1 if price > SMA(lookback), else 0 ---
def momentum_gate_binary(
    prices: pd.DataFrame, ix: int, lookback: int = 50
) -> pd.Series:
    sma = prices.rolling(lookback).mean()
    gate = (prices.iloc[ix] > sma.iloc[ix]).astype(float)
    return gate.fillna(0.0)


# --- continuous signal: (price - SMA)/SMA ---
def momentum_signal_continuous(
    prices: pd.DataFrame, ix: int, lookback: int = 50
) -> pd.Series:
    sma = prices.rolling(lookback).mean()
    sig = (prices.iloc[ix] - sma.iloc[ix]) / sma.iloc[ix]
    return sig.replace([np.inf, -np.inf], 0.0).fillna(0.0)


# --- regime gates based on moving averages ---
def ma_cross_regime_gate(
    prices: pd.DataFrame, ix: int, fast: int = 50, slow: int = 200
) -> pd.Series:
    s_fast = prices.rolling(fast).mean()
    s_slow = prices.rolling(slow).mean()
    gate = (s_fast.iloc[ix] > s_slow.iloc[ix]).astype(float)
    return gate.fillna(0.0)


def ma50_rising_gate(
    prices: pd.DataFrame, ix: int, lookback: int = 50, slope_win: int = 10
) -> pd.Series:
    sma = prices.rolling(lookback).mean()
    prev_ix = max(0, ix - slope_win)
    s = sma.iloc[ix] - sma.iloc[prev_ix]
    gate = (s > 0).astype(float)
    return gate.fillna(0.0)


def ma_strength_continuous(
    prices: pd.DataFrame, ix: int, fast: int = 50, slow: int = 200
) -> pd.Series:
    s_fast = prices.rolling(fast).mean()
    s_slow = prices.rolling(slow).mean()
    strength = ((s_fast.iloc[ix] - s_slow.iloc[ix]) / s_slow.iloc[ix]).replace(
        [np.inf, -np.inf], 0.0
    )
    return strength.fillna(0.0)


# --- adjacency sparsification: keep top-k outgoing edges per node by |weight| ---
def sparsify_topk_outgoing(A: pd.DataFrame, k: int) -> pd.DataFrame:
    if k is None or k <= 0:
        return A
    A2 = A.copy()
    # for each row (leader), keep top-k absolute weights
    for i in A2.index:
        row = A2.loc[i].abs()
        if (row > 0).sum() > k:
            cutoff = row.nlargest(k).min()
            # zero out entries with abs weight below cutoff
            mask = row < cutoff
            A2.loc[i, mask.index[mask]] = 0.0
    return A2


def combine_leader_momentum_longonly(zlead: pd.Series, mom_gate: pd.Series):
    w_raw = np.maximum(0, zlead) * (mom_gate.reindex(zlead.index).fillna(0.0))
    gross = np.abs(w_raw).sum()
    return (w_raw / gross) if gross > 0 else w_raw * 0.0


def combine_leader_momentum_continuous(zlead: pd.Series, mom_sig: pd.Series):
    # keep leaders only; scale by positive momentum strength
    pos_mom = mom_sig.clip(lower=0.0)
    w_raw = np.maximum(0, zlead) * (pos_mom.reindex(zlead.index).fillna(0.0))
    gross = np.abs(w_raw).sum()
    return (w_raw / gross) if gross > 0 else w_raw * 0.0


def scale_to_target_vol(
    returns_win: pd.DataFrame,
    w: pd.Series,
    target_ann_vol: float = 0.10,
    max_leverage: float = 1.5,
    shrink_lambda: float | None = None,
):
    if returns_win.shape[0] < 20 or w.abs().sum() == 0:
        return w
    cov = returns_win.cov()
    # optional ridge shrinkage to stabilize covariance
    if shrink_lambda is not None and shrink_lambda > 0:
        lam = float(shrink_lambda)
        diag_mean = np.nanmean(np.diag(cov.values)) if cov.size else 0.0
        cov = (1 - lam) * cov + lam * np.eye(cov.shape[0]) * diag_mean
    # align weights to covariance matrix columns to avoid order/index mismatches
    w_aligned = w.reindex(cov.columns).fillna(0.0)
    port_var_daily = float(np.dot(w_aligned.values, cov.values @ w_aligned.values))
    if port_var_daily <= 0:
        return w
    port_vol_ann = np.sqrt(port_var_daily * 252.0)
    if port_vol_ann == 0:
        return w
    scale = target_ann_vol / port_vol_ann
    scale = np.clip(scale, 0.0, max_leverage)
    return w * scale


def transaction_cost(turnover_abs: float, bps_per_side: float = 3.0) -> float:
    return (bps_per_side / 10000.0) * turnover_abs


def backtest_network_momentum(
    prices: pd.DataFrame,  # Close or Adj Close (wide DF)
    returns: pd.DataFrame,  # daily log returns aligned with prices
    window: int = 252,  # lookback for network
    max_lag: int = 5,
    min_abs_corr: float = 0.15,
    rebalance: str = "M",  # "M" (monthly) or "W-FRI"
    mom_lookback: int = 50,
    ema_span: int | None = 20,  # None => no smoothing
    target_ann_vol: float | None = 0.10,  # None => no vol targeting
    tc_bps: float = 3.0,  # transaction cost bps per side
    cov_win: int = 60,  # cov window for vol targeting
    # New options
    leader_method: str = "out_strength",  # e.g., "out_strength", "abs_out_strength", "pos_only", "sign_aware_pagerank", "pagerank"
    pagerank_alpha: float = 0.9,
    sparsify_topk: int | None = None,
    regime: str = "price_sma",  # "price_sma", "ma_cross", "ma_slope", "ma_strength"
    fast_ma: int = 50,
    slow_ma: int = 200,
    slope_win: int = 10,
    shrink_lambda: float | None = None,
    use_numba: bool = False,
):
    if rebalance.upper() == "M":
        rebal_dates = prices.resample("M").last().index
    elif rebalance.upper() == "W-FRI":
        # weekly on Fridays
        rebal_dates = prices[prices.index.weekday == 4].index
    else:
        raise ValueError("rebalance must be 'M' or 'W-FRI'")

    w_prev = pd.Series(0.0, index=prices.columns)
    zlead_prev = None
    port_daily = []
    w_records = []

    for t in rebal_dates:
        if t not in prices.index:
            continue
        ix = prices.index.get_loc(t)
        if ix < max(window, mom_lookback) + 1:
            continue

        # build adj
        sample = returns.iloc[ix - window : ix]
        A = (build_adj_fast if use_numba else build_adj)(
            sample, max_lag=max_lag, min_abs_corr=min_abs_corr
        )

        A = filter_edges_fdr(A, n_eff=sample.shape[0], q=0.2)

        # optional sparsification of outgoing edges per node
        if sparsify_topk is not None and sparsify_topk > 0:
            A = sparsify_topk_outgoing(A, sparsify_topk)

        # leader scoring (configurable)
        zlead = leader_scores(A, method=leader_method, pagerank_alpha=pagerank_alpha)
        if ema_span is not None:
            zlead = ema_update(prev_smoothed=zlead_prev, new_z=zlead, span=ema_span)
            zlead_prev = zlead.copy()

        # regime gating / signals
        if regime == "price_sma":
            mom_gate = momentum_gate_binary(prices, ix, lookback=mom_lookback)
            w = combine_leader_momentum_longonly(zlead, mom_gate)
        elif regime == "ma_cross":
            mom_gate = ma_cross_regime_gate(prices, ix, fast=fast_ma, slow=slow_ma)
            w = combine_leader_momentum_longonly(zlead, mom_gate)
        elif regime == "ma_slope":
            mom_gate = ma50_rising_gate(
                prices, ix, lookback=fast_ma, slope_win=slope_win
            )
            w = combine_leader_momentum_longonly(zlead, mom_gate)
        elif regime == "ma_strength":
            mom_sig = ma_strength_continuous(prices, ix, fast=fast_ma, slow=slow_ma)
            w = combine_leader_momentum_continuous(zlead, mom_sig)
        else:
            # fallback to original binary momentum gate
            mom_gate = momentum_gate_binary(prices, ix, lookback=mom_lookback)
            w = combine_leader_momentum_longonly(zlead, mom_gate)

        if target_ann_vol is not None:
            cov_window = returns.iloc[ix - cov_win : ix]
            w = scale_to_target_vol(
                cov_window,
                w,
                target_ann_vol=target_ann_vol,
                shrink_lambda=shrink_lambda,
            )

        w_records.append(pd.Series(w, name=prices.index[ix]))

        next_pos = prices.index.searchsorted(t, side="right")
        if next_pos >= len(prices.index):
            break
        if t == rebal_dates[-1]:
            end_pos = len(prices.index)
        else:
            nxt = rebal_dates[rebal_dates.get_loc(t) + 1]
            end_pos = prices.index.searchsorted(nxt, side="left")

        turnover_abs = (w - w_prev).abs().sum()
        cost = transaction_cost(turnover_abs, bps_per_side=tc_bps)
        w_prev = w.copy()

        hold_slice = returns.iloc[next_pos:end_pos]
        if hold_slice.empty:
            continue

        first_day = True
        for day, rvec in hold_slice.iterrows():
            r_p = float(
                np.dot(w.reindex(rvec.index, fill_value=0.0).values, rvec.values)
            )
            if first_day:
                r_p -= cost
                first_day = False
            port_daily.append((day, r_p))

    if not port_daily:
        return pd.Series({d: r for d, r in port_daily}).sort_index()

    port_rets = pd.Series({d: r for d, r in port_daily}).sort_index()

    # ---- Metrics ----
    ann_mult = 252.0
    cagr = (
        (np.exp(port_rets.sum()) ** (ann_mult / len(port_rets)) - 1)
        if len(port_rets) > 0
        else 0.0
    )
    ann_vol = port_rets.std() * np.sqrt(ann_mult)
    sharpe = (port_rets.mean() * ann_mult) / ann_vol if ann_vol > 0 else np.nan
    cum = port_rets.cumsum().apply(np.exp)
    peak = cum.cummax()
    dd = cum / peak - 1.0
    max_dd = dd.min() if not dd.empty else 0.0
    avg_turnover = np.nan

    metrics = {
        "CAGR": cagr,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "NumTrades": len(w_records),
        "Start": str(port_rets.index[0]),
        "End": str(port_rets.index[-1]),
    }
    # with open("metrics.json", "w") as f:
    #     json.dump(metrics, f, indent=4)

    w_hist = pd.DataFrame(w_records)
    return port_rets, metrics, w_hist
