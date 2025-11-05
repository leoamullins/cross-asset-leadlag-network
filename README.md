# Cross-Asset Lead–Lag Network

A quantitative research framework for discovering directional lead–lag relationships across financial asset returns using network theory and correlation structure analysis. The toolkit constructs signed, directed graphs from lagged cross-asset correlations and implements a systematic trading strategy that exploits leadership dynamics in multi-asset portfolios.

## Project Structure

The repository implements a modular Python package (`cross_asset_leadlag/`) with clear separation of concerns:

- **`cross_asset_leadlag.graph`** — Lagged correlation estimation, adjacency matrix construction, network visualisation, and FDR-based edge filtering
- **`cross_asset_leadlag.algo`** — Strategy implementation including leader scoring (PageRank, out-strength variants), regime filters, volatility targeting with covariance shrinkage, and vectorised backtesting
- **`cross_asset_leadlag.validation`** — Walk-forward validation, Sharpe ratio computation, and out-of-sample performance metrics
- **`example.ipynb`** — End-to-end demonstration with data acquisition, network construction, and strategy backtesting

## Performance Summary

The strategy demonstrates robust risk-adjusted returns over a 16-year sample spanning multiple market regimes:

| Metric           | Value   |
|------------------|---------|
| **CAGR**         | 5.07%   |
| **Ann. Vol**     | 9.31%   |
| **Sharpe Ratio** | 0.53    |
| **Max Drawdown** | -14.7%  |
| **Rebalances**   | 844     |
| **Period**       | Apr 2008 – Dec 2024 |

### Network Visualisation

![Lead–Lag Network (last 252d)](images/leadlag_network.png)
*Directed graph showing lead–lag relationships. Node size reflects leadership strength (z-scored out-strength); edge direction indicates temporal precedence; edge thickness represents correlation magnitude.*

### Cumulative Returns

![Network Momentum: Cumulative Growth of $1](images/cum_returns.png)
*Strategy equity curve: portfolio log-returns compounded over the backtest period, after transaction costs (3bps per side).*

## Research Motivation

Traditional momentum and correlation-based strategies typically assume contemporaneous relationships or rely on historical price patterns alone. This approach overlooks directional, lagged influence structures between assets—where certain instruments systematically lead others in price discovery.

This research investigates whether:
1. Lagged cross-correlations reveal stable leadership hierarchies across asset classes
2. Network-based leader scoring (PageRank, out-strength) improves systematic portfolio construction
3. Combining leadership signals with momentum regime filters enhances risk-adjusted returns

## Methodology

### 1. Lead–Lag Network Construction

For each pair of assets *(i, j)* in the universe:
- Compute Pearson correlations at lags *τ ∈ [-L, +L]* (default *L = 5* days)
- Select lag *τ\** with maximum absolute correlation
- If *|ρ(τ\*)| ≥ θ* (default *θ = 0.15*), create directed edge:
  - *τ\* > 0*: asset *i* leads *j* → edge *i → j* with weight *ρ(τ\*)*
  - *τ\* < 0*: asset *j* leads *i* → edge *j → i* with weight *ρ(τ\*)*
- Apply Benjamini–Hochberg FDR correction (*q = 0.2*) to control false discovery rate
- Optional: sparsify by retaining top-*k* outgoing edges per node

### 2. Leader Scoring

Multiple centrality measures available:
- **Out-strength**: Σⱼ *Aᵢⱼ* (signed row sum)
- **Absolute out-strength**: Σⱼ *|Aᵢⱼ|* (ignores correlation sign)
- **Positive-only strength**: Σⱼ max(*Aᵢⱼ*, 0)
- **PageRank**: Standard random-walk centrality on positive edges
- **Sign-aware PageRank**: Influence via *|A|*, modulated by sign of out-strength

All scores are z-normalised and optionally smoothed via exponential moving average.

### 3. Portfolio Construction

**Signal generation** (rebalanced monthly/weekly):
1. Rank assets by leader score *zᵢ*
2. Apply momentum regime filter:
   - `price_sma`: *Pᵢ > SMA₅₀(Pᵢ)*
   - `ma_cross`: *SMA₅₀(Pᵢ) > SMA₂₀₀(Pᵢ)*
   - `ma_slope`: *SMA₅₀(Pᵢ)* rising over 10-day window
   - `ma_strength`: continuous signal *[SMA₅₀ - SMA₂₀₀] / SMA₂₀₀*
3. Combine: *wᵢ ∝ max(zᵢ, 0) × gate*ᵢ (long-only allocation)

**Risk management**:
- Volatility targeting: scale to *σ_target* = 10% annualised using rolling 60-day covariance
- Optional Ledoit–Wolf shrinkage (*λ = 0.1*) for covariance stabilisation
- Max leverage: 1.5×
- Transaction costs: 3bps per side

### 4. Performance Attribution

- Walk-forward validation with out-of-sample testing
- Sharpe ratio, CAGR, maximum drawdown
- Turnover analysis and cost sensitivity

## Key Features

- **Numba-accelerated computation**: Parallel adjacency matrix construction for large universes
- **Statistical robustness**: FDR-controlled edge filtering, robust PageRank with multiple fallbacks
- **Flexible regime filters**: Binary gates (SMA, MA cross) or continuous momentum signals
- **Production-ready risk management**: Volatility targeting with shrinkage, leverage constraints, realistic transaction costs
- **Modular architecture**: Clean separation of graph construction, scoring, and backtesting logic

## Installation & Usage

### Requirements

- Python ≥3.10
- Core: `numpy`, `pandas`, `matplotlib`, `networkx`, `scipy`
- Data: `yfinance` (market data retrieval)
- Performance: `numba` (optional; enables JIT-compiled adjacency construction)

Install dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

**Option 1: Interactive Notebook**

Open [example.ipynb](example.ipynb) to run the complete pipeline:
1. Fetch daily ETF prices (equities, bonds, commodities, currencies)
2. Construct 252-day rolling lead–lag network
3. Visualise graph with leadership rankings
4. Backtest network momentum strategy
5. Analyse performance metrics

**Option 2: Python Script**

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from cross_asset_leadlag.graph import build_adj, leadlag_graph
from cross_asset_leadlag.algo import backtest_network_momentum

# 1. Data acquisition
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM',      # Equities
           'TLT', 'IEF', 'SHY', 'LQD', 'HYG',      # Fixed income
           'GLD', 'SLV', 'DBC',                     # Commodities
           'UUP', 'FXE']                            # Currencies

prices = yf.download(tickers, start='2007-01-01', end='2025-01-01')['Close'].dropna()
returns = np.log(prices / prices.shift(1)).dropna()

# 2. Network construction
A = build_adj(returns.tail(252), max_lag=5, min_abs_corr=0.15)
H, leader_scores = leadlag_graph(
    A,
    title='Lead–Lag Network (last 252d)',
    max_edges=80,
    node_score='out_strength',
    layout='circular',
    seed=42,
    visualise=True
)

print("Top 5 Leaders:\n", leader_scores.sort_values(ascending=False).head())
plt.savefig("leadlag_network.png", bbox_inches="tight", dpi=150)

# 3. Strategy backtest
portfolio_returns, metrics, weights = backtest_network_momentum(
    prices, returns,
    window=252,
    leader_method="abs_out_strength",
    max_lag=5,
    min_abs_corr=0.15,
    sparsify_topk=3,
    regime="ma_cross",
    fast_ma=50,
    slow_ma=200,
    ema_span=15,
    target_ann_vol=0.10,
    cov_win=60,
    shrink_lambda=0.1,
    rebalance="W-FRI",
    tc_bps=3.0,
    use_numba=True
)

# 4. Performance analysis
print("\nPerformance Metrics:")
for k, v in metrics.items():
    print(f"{k:15s}: {v}")

# Plot equity curve
cumulative = portfolio_returns.cumsum().apply(np.exp)
fig, ax = plt.subplots(figsize=(10, 6))
cumulative.plot(ax=ax, linewidth=2)
ax.set_title('Network Momentum Strategy: Equity Curve', fontsize=14)
ax.set_ylabel('Growth of £1', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.grid(alpha=0.3)
plt.savefig("cum_returns.png", bbox_inches="tight", dpi=150)
plt.show()
```

## Technical Implementation Details

### Graph Construction Algorithm

```python
def build_adj(returns: pd.DataFrame, max_lag: int = 5, min_abs_corr: float = 0.15):
    """
    Constructs signed, directed adjacency matrix from lagged correlations.

    For each pair (i, j):
      1. Compute ρ(τ) for τ ∈ [-max_lag, +max_lag]
      2. Select τ* = argmax |ρ(τ)|
      3. If |ρ(τ*)| ≥ min_abs_corr:
           - τ* > 0: A[i,j] = ρ(τ*)  (i leads j)
           - τ* < 0: A[j,i] = ρ(τ*)  (j leads i)

    Returns: N×N DataFrame with assets as index/columns
    """
```

**Numba acceleration**: `build_adj_fast()` provides parallel implementation with ~10× speedup on large universes.

### Statistical Validation

- **FDR correction**: Benjamini–Hochberg procedure controls false discovery rate in multiple testing
- **Walk-forward analysis**: Out-of-sample validation with non-overlapping test periods
- **Robust covariance estimation**: Ledoit–Wolf shrinkage prevents overfitting in small samples

### Risk Management Implementation

**Volatility targeting** solves:

*w_scaled = w × min(σ_target / √(w'Σw × 252), leverage_max)*

where *Σ* is the 60-day rolling covariance matrix (optionally shrunk).

## Research Extensions & Limitations

### Current Limitations

1. **Correlation stability**: Lagged relationships may exhibit regime-dependence; edges can flip during structural breaks
2. **Linear assumption**: Pearson correlation captures only linear relationships; nonlinear lead–lag structures (copulas, mutual information) not addressed
3. **Transaction cost model**: Simplified proportional costs (3bps per side); no market impact, spread modelling, or capacity analysis
4. **Universe construction**: Fixed ETF universe; no asset selection, delisting bias, or survivorship adjustment
5. **Parameter sensitivity**: Performance materially affected by `min_abs_corr`, `max_lag`, `sparsify_topk`

### Potential Enhancements

- **Adaptive parameters**: Rolling cross-validation for optimal threshold selection
- **Regime detection**: Hidden Markov models for state-dependent networks
- **Nonlinear extensions**: Granger causality, transfer entropy, conditional mutual information
- **Portfolio constraints**: Sector limits, turnover penalties, short-sale constraints
- **Execution modelling**: VWAP slippage, temporary/permanent impact (Almgren–Chriss framework)
- **Alternative centrality**: Eigenvector centrality, betweenness, Katz centrality

## API Reference

### `cross_asset_leadlag.graph`

**Core Functions:**

- **`build_adj(returns, max_lag=5, min_abs_corr=0.15)`** → `pd.DataFrame`
  - Constructs signed, directed adjacency matrix from lagged correlations
  - Returns *N×N* matrix where *A[i,j]* = correlation weight if *i* leads *j*

- **`build_adj_fast(returns, max_lag=5, min_abs_corr=0.15)`** → `pd.DataFrame`
  - Numba-accelerated version (10× faster); automatic fallback if Numba unavailable

- **`filter_edges_fdr(A, n_eff, q=0.1)`** → `pd.DataFrame`
  - Benjamini–Hochberg FDR correction; zeros out non-significant edges

- **`leadlag_graph(A, title, max_edges=80, node_score='out_strength', layout='circular', visualise=True)`** → `(nx.DiGraph, pd.Series)`
  - Visualises network and computes leadership z-scores
  - Node scoring methods: `out_strength`, `abs_out_strength`, `pos_only`, `pagerank`, `sign_aware_pagerank`

### `cross_asset_leadlag.algo`

**Strategy Functions:**

- **`leader_scores(A, method='out_strength', pagerank_alpha=0.9)`** → `pd.Series`
  - Computes z-scored leadership centrality measures

- **`backtest_network_momentum(prices, returns, window=252, ...)`** → `(pd.Series, dict, pd.DataFrame)`
  - Full backtesting engine with rolling network construction
  - Key parameters:
    - `leader_method`: `'out_strength'`, `'abs_out_strength'`, `'pagerank'`, `'sign_aware_pagerank'`
    - `regime`: `'price_sma'`, `'ma_cross'`, `'ma_slope'`, `'ma_strength'`
    - `target_ann_vol`: Volatility target (e.g., 0.10 for 10%)
    - `shrink_lambda`: Covariance shrinkage intensity (0.0 = none, 1.0 = full)
    - `sparsify_topk`: Retain top-*k* edges per node
    - `use_numba`: Enable JIT acceleration
  - Returns: `(daily_returns, metrics_dict, weight_history)`

**Regime Filters:**

- `momentum_gate_binary(prices, ix, lookback)`: Price > SMA filter
- `ma_cross_regime_gate(prices, ix, fast, slow)`: Golden/death cross
- `ma50_rising_gate(prices, ix, slope_win)`: SMA slope filter
- `ma_strength_continuous(prices, ix, fast, slow)`: Continuous momentum signal

### `cross_asset_leadlag.validation`

- **`walkforward_validation(prices, returns, window=252, test_len=63, ...)`** → `pd.Series`
  - Out-of-sample walk-forward testing with purging and embargo

- **`sharpe_ratio(returns, ann_factor=252, risk_free=0.0)`** → `float`
  - Annualised Sharpe ratio calculation

## Usage Notes

**Performance optimisation:**
- Enable `use_numba=True` for universes >20 assets
- Reduce `max_lag` or `window` during prototyping
- Cache yfinance data locally to avoid API throttling

**Stability considerations:**
- Increase `min_abs_corr` (e.g., 0.20) for sparser, more stable networks
- Apply `sparsify_topk=3` to retain only strongest edges per node
- Use `ema_span=15` to smooth noisy leadership signals

**Regime sensitivity:**
- `ma_cross` works well in trending markets but lags in reversals
- `price_sma` provides faster regime detection but higher turnover
- `ma_strength` offers continuous exposure scaling (no binary gates)

## Project Background

This research project was developed as part of a quantitative finance portfolio demonstrating:
- **Statistical signal processing**: Time-series correlation analysis with lag optimisation
- **Network science applications**: Graph theory applied to financial markets
- **Systematic strategy development**: End-to-end pipeline from research to backtesting
- **Software engineering**: Modular Python package with performance optimisation (Numba JIT)
- **Risk management**: Volatility targeting, covariance estimation, transaction cost modelling

The implementation showcases practical skills relevant to quantitative researcher and systematic trading roles, including backtesting best practices, statistical validation, and production-quality code organisation.

## References & Further Reading

**Lead–lag relationships:**
- Brunetti, C., & Lildholdt, P. (2007). *Time-varying correlations and the spatial structure of stock market*.
- Curme, C., et al. (2015). *Emergence of statistically validated financial intraday lead-lag relationships*.

**Network methods in finance:**
- Mantegna, R. N. (1999). *Hierarchical structure in financial markets*.
- Onnela, J.-P., et al. (2003). *Dynamics of market correlations: Taxonomy and portfolio analysis*.

**Portfolio construction & risk management:**
- Ledoit, O., & Wolf, M. (2004). *Honey, I shrunk the sample covariance matrix*.
- Grinold, R. C., & Kahn, R. N. (1999). *Active Portfolio Management*.

## Licence & Disclaimer

**Licence**: MIT — See [LICENSE](LICENSE) for details.

**Disclaimer**: This repository is for research and educational purposes only. Nothing herein constitutes investment advice, and past performance does not guarantee future results. All backtests are subject to survivorship bias, overfitting risk, and simplifying assumptions that may not hold in live trading.
