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
from cross_asset_leadlag import backtest_network_momentum
import pandas as pd

# Load your price and return data
prices = pd.read_csv('prices.csv', index_col=0, parse_dates=True)
returns = prices.pct_change().fillna(0)

# Run backtest with custom parameters
port_rets, metrics, w_hist = backtest_network_momentum(
    prices=prices,
    returns=returns,
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

# Display performance metrics
print(f"Sharpe Ratio: {metrics['Sharpe']:.2f}")
print(f"CAGR: {metrics['CAGR']:.1%}")
print(f"Max Drawdown: {metrics['MaxDrawdown']:.1%}")
```

### Walk-Forward Validation

```python
from cross_asset_leadlag import walkforward_validation, sharpe_ratio

# Run walk-forward validation
oos_returns = walkforward_validation(
    prices=prices,
    returns=returns,
    window=252,
    test_len=63,      # ~3 months per test period
    purge=5,          # purge period to avoid contamination
    embargo=5,        # embargo period after test
    # pass any backtest_network_momentum parameters
    leader_method='pagerank',
    regime='ma_cross'
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

## Tech Stack

Python • NetworkX • NumPy • pandas • Matplotlib • SciPy • Numba (optional)

## Future Improvements

- [ ] Extend to intraday data for higher frequency signals
- [ ] Incorporate fundamental factors alongside price-based networks
- [ ] Implement adaptive rebalancing based on market volatility
- [ ] Add options overlay for tail risk hedging
- [ ] Multi-factor regime classification using machine learning

## References

- Billio, M., et al. (2012). "Econometric measures of connectedness and systemic risk in the finance and insurance sectors"
- Diebold, F. X., & Yilmaz, K. (2014). "On the network topology of variance decompositions"

## License

MIT License - see LICENSE file for details

---

**Note:** This is a research project for educational purposes. Past performance does not guarantee future results. Not financial advice.
