# Cross-Asset Lead-Lag Network

Quantitative framework for identifying information flow across asset classes using network centrality measures derived from lagged correlation matrices.

## Results

![Cumulative Returns](images/cum_returns.png)

**Backtest Performance (2008-2024):**
- Sharpe Ratio: 0.86
- CAGR: 9.2%
- Max Drawdown: 13.8%

## Overview

This project implements a network-based momentum strategy that exploits lead-lag relationships between asset classes. By constructing directed graphs from rolling lagged correlations and calculating centrality scores, the system identifies which assets tend to lead market movements.

## Methodology

1. **Network Construction**
   - Compute rolling lagged correlation matrices across multiple assets
   - Assets can include equities, bonds, commodities, and currencies
   - Build directed network where edges represent predictive relationships
   - Optional FDR correction to filter statistically significant edges

2. **Centrality Calculation**
   - Multiple scoring methods available: out-strength, PageRank, sign-aware PageRank
   - Higher centrality scores indicate assets that lead market movements
   - Centrality rankings updated with rolling window
   - Optional EMA smoothing of leadership scores

3. **Trading Strategy**
   - Long positions in assets with highest centrality scores
   - Multiple regime filters: price vs SMA, MA cross, MA slope, MA strength
   - Monthly or weekly rebalancing frequencies supported
   - Optional volatility targeting with covariance-based scaling
   - Transaction costs and realistic execution assumptions included

4. **Validation**
   - Walk-forward analysis framework with configurable train/test splits
   - Purge and embargo periods to prevent look-ahead bias
   - Statistical significance testing via FDR-controlled edge filtering

## Installation

```bash
git clone https://github.com/yourusername/cross-asset-leadlag-network.git
cd cross-asset-leadlag-network
pip install -r requirements.txt
```

## Usage

```python
from cross_asset_leadlag import backtest_network_momentum, BacktestConfig
import pandas as pd

# Load your price and return data
prices = pd.read_csv('prices.csv', index_col=0, parse_dates=True)
returns = prices.pct_change().fillna(0)

# Configure backtest parameters
config = BacktestConfig(
    window=252,              # lookback window for network
    max_lag=5,               # maximum lag for correlations
    min_abs_corr=0.15,       # minimum correlation threshold
    rebalance='M',           # 'M' for monthly, 'W-FRI' for weekly
    leader_method='out_strength',  # scoring method
    regime='price_sma',      # regime filter
    mom_lookback=50,         # momentum lookback period
    target_ann_vol=0.10,     # volatility target (None to disable)
    tc_bps=3.0,              # transaction cost in bps
    use_numba=True           # use Numba acceleration if available
)

# Run backtest
port_rets, metrics, w_hist = backtest_network_momentum(
    prices=prices,
    returns=returns,
    config=config
)

# Display performance metrics
print(f"Sharpe Ratio: {metrics['Sharpe']:.2f}")
print(f"CAGR: {metrics['CAGR']:.1%}")
print(f"Max Drawdown: {metrics['MaxDrawdown']:.1%}")
```

### Walk-Forward Validation

```python
from cross_asset_leadlag import walkforward_validation, sharpe_ratio, BacktestConfig

# Configure backtest parameters
config = BacktestConfig(
    window=252,
    leader_method='pagerank',
    regime='ma_cross'
)

# Run walk-forward validation
oos_returns = walkforward_validation(
    prices=prices,
    returns=returns,
    config=config,
    test_len=63,      # ~3 months per test period
    purge=5           # purge period to avoid contamination
)

# Calculate out-of-sample Sharpe ratio
oos_sharpe = sharpe_ratio(oos_returns)
print(f"Out-of-sample Sharpe: {oos_sharpe:.2f}")
```

### Network Visualisation

```python
from cross_asset_leadlag import build_adj, leadlag_graph

# Build adjacency matrix from returns
A = build_adj(returns_sample, max_lag=5, min_abs_corr=0.15)

# Visualise the lead-lag network
graph, leader_scores = leadlag_graph(
    A,
    title='Lead-Lag Network',
    max_edges=80,
    node_score='pagerank',
    layout='spring',
    visualise=True
)
```

## Key Features

- **Multiple leader scoring methods:** Out-strength, PageRank, sign-aware PageRank, absolute out-strength, positive-only
- **Flexible regime filters:** Price vs SMA, MA crossover, MA slope, MA strength
- **Numba acceleration:** Optional fast adjacency matrix construction using Numba JIT compilation
- **Statistical edge filtering:** FDR (False Discovery Rate) correction for multiple testing
- **Volatility targeting:** Portfolio-level volatility scaling with optional covariance shrinkage
- **Transaction cost modeling:** Configurable basis points per side
- **Walk-forward validation:** Built-in framework for robust out-of-sample testing
- **Network visualisation:** Interactive graph plotting with customisable layouts

## Project Structure

```
cross-asset-leadlag-network/
├── cross_asset_leadlag/
│   ├── __init__.py         # Public API exports
│   ├── graph.py            # Network construction and PageRank
│   ├── algo.py             # Trading strategy implementation
│   ├── config.py           # Configuration dataclass
│   └── validation.py       # Walk-forward validation utilities
├── requirements.txt
└── README.md
```

## Configuration Options

### Leader Scoring Methods
- `out_strength`: Sum of outgoing edge weights (default)
- `abs_out_strength`: Sum of absolute outgoing edge weights
- `pos_only`: Sum of positive outgoing edge weights only
- `pagerank`: Standard PageRank on positive edge weights
- `sign_aware_pagerank`: PageRank magnitude × sign of net outflow

### Regime Filters
- `price_sma`: Binary gate based on price > SMA(lookback)
- `ma_cross`: Binary gate based on SMA(fast) > SMA(slow)
- `ma_slope`: Binary gate based on rising SMA(fast)
- `ma_strength`: Continuous signal based on (SMA_fast - SMA_slow) / SMA_slow

### Other Parameters
- `sparsify_topk`: Keep only top-k outgoing edges per node (by |weight|)
- `ema_span`: EMA smoothing span for leadership scores (None to disable)
- `shrink_lambda`: Ridge shrinkage parameter for covariance stabilisation
- `cov_win`: Window size for covariance estimation in volatility targeting

## Technical Implementation Details

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

- **`BacktestConfig(window=252, max_lag=5, ..., use_numba=False)`** → `dataclass`
  - Single source of truth for strategy hyperparameters (window length, regime, vol targeting, etc.)

- **`backtest_network_momentum(prices, returns, config)`** → `(pd.Series, dict, pd.DataFrame)`
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

- **`walkforward_validation(prices, returns, config, test_len=63, purge=5, ...)`** → `pd.Series`
  - Reuses the same `BacktestConfig` as the core strategy and applies walk-forward splits with configurable test-window and purge lengths

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

## Tech Stack

Python • NetworkX • NumPy • pandas • Matplotlib • SciPy • Numba (optional)

## Project Background

This research project was developed as part of a quantitative finance portfolio demonstrating:
- **Statistical signal processing**: Time-series correlation analysis with lag optimisation
- **Network science applications**: Graph theory applied to financial markets
- **Systematic strategy development**: End-to-end pipeline from research to backtesting
- **Software engineering**: Modular Python package with performance optimisation (Numba JIT)
- **Risk management**: Volatility targeting, covariance estimation, transaction cost modelling

The implementation showcases practical skills relevant to quantitative researcher and systematic trading roles, including backtesting best practices, statistical validation, and production-quality code organisation.

## References

**Lead–lag relationships:**
- Brunetti, C., & Lildholdt, P. (2007). *Time-varying correlations and the spatial structure of stock market*.
- Curme, C., et al. (2015). *Emergence of statistically validated financial intraday lead-lag relationships*.

**Network methods in finance:**
- Billio, M., et al. (2012). "Econometric measures of connectedness and systemic risk in the finance and insurance sectors"
- Diebold, F. X., & Yilmaz, K. (2014). "On the network topology of variance decompositions"
- Mantegna, R. N. (1999). *Hierarchical structure in financial markets*.
- Onnela, J.-P., et al. (2003). *Dynamics of market correlations: Taxonomy and portfolio analysis*.

**Portfolio construction & risk management:**
- Ledoit, O., & Wolf, M. (2004). *Honey, I shrunk the sample covariance matrix*.
- Grinold, R. C., & Kahn, R. N. (1999). *Active Portfolio Management*.

## License

MIT License - see LICENSE file for details

---

**Note:** This is a research project for educational purposes. Past performance does not guarantee future results. Not financial advice.
