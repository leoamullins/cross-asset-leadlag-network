# Cross-Asset Lead-Lag Network

Quantitative framework for identifying information flow across asset classes using network centrality measures derived from lagged correlation matrices.

## Results

![Cumulative Returns](images/cum_returns.png)

**Backtest Performance (2008-2024):**
- Sharpe Ratio: 0.86
- CAGR: 9.2%
- Max Drawdown: 13.8%

## Overview

This project implements a network-based momentum strategy that exploits lead-lag relationships between asset classes. By constructing directed graphs from rolling lagged correlations and calculating PageRank centrality, the system identifies which assets tend to lead market movements.

## Methodology

1. **Network Construction**
   - Compute 60-day rolling lagged correlation matrices across 50+ assets
   - Assets include equities, bonds, commodities, and currencies
   - Build directed network where edges represent predictive relationships

2. **Centrality Calculation**
   - Apply PageRank algorithm to identify influential assets
   - Higher centrality scores indicate assets that lead market movements
   - Centrality rankings updated with rolling window

3. **Trading Strategy**
   - Long positions in assets with highest centrality scores
   - Quarterly rebalancing to adapt to changing market dynamics
   - Transaction costs and realistic execution assumptions included

4. **Validation**
   - Walk-forward analysis confirms signal persistence across regimes
   - Statistical significance: Spearman correlation of 0.31 to forward 20-day returns (p < 0.001)

## Installation

```bash
git clone https://github.com/leoamullins/cross-asset-leadlag-network.git
cd cross-asset-leadlag-network
pip install -r requirements.txt
```

## Usage

```python
from network_momentum import NetworkStrategy

# Initialise strategy
strategy = NetworkStrategy(
    lookback_period=60,
    n_assets=50,
    rebalance_freq='quarterly'
)

# Run backtest
results = strategy.backtest(
    start_date='2008-01-01',
    end_date='2024-12-31'
)

# Generate performance metrics
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"CAGR: {results.cagr:.1%}")
print(f"Max Drawdown: {results.max_drawdown:.1%}")
```

## Key Features

- **Multi-asset coverage:** Equities, bonds, FX, commodities
- **Robust validation:** Walk-forward analysis across multiple market regimes
- **Realistic modeling:** Includes transaction costs, slippage, and execution delays
- **Visualisation tools:** Network graphs and performance attribution analysis

## Project Structure

```
cross-asset-leadlag-network/
├── data/               # Market data (not included)
├── notebooks/          # Jupyter notebooks with analysis
├── src/
│   ├── network.py      # Network construction and PageRank
│   ├── strategy.py     # Trading strategy implementation
│   ├── backtest.py     # Backtesting engine
│   └── utils.py        # Helper functions
├── requirements.txt
└── README.md
```

## Tech Stack

Python • NetworkX • NumPy • pandas • Matplotlib • numba

## Future Improvements

- [ ] Extend to intraday data for higher frequency signals
- [ ] Incorporate fundamental factors alongside price-based networks
- [ ] Implement adaptive rebalancing based on market volatility
- [ ] Add options overlay for tail risk hedging

## References

- Billio, M., et al. (2012). "Econometric measures of connectedness and systemic risk in the finance and insurance sectors"
- Diebold, F. X., & Yilmaz, K. (2014). "On the network topology of variance decompositions"

## License

MIT License - see LICENSE file for details

---

**Note:** This is a research project for educational purposes. Past performance does not guarantee future results. Not financial advice.
