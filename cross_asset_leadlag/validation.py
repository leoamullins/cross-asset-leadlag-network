import numpy as np
import pandas as pd
from .algo import backtest_network_momentum
from .config import BacktestConfig


def walkforward_validation(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
    test_len: int = 63,
    purge: int = 5,
):
    window: int = config.window
    n = len(prices)
    test_starts = np.arange(window, n - test_len, test_len)
    all_rets = []
    for start in test_starts:
        train_slice = slice(start - window, start - purge)
        test_slice = slice(start, start + test_len)
        p_train, r_train = prices.iloc[train_slice], returns.iloc[train_slice]
        p_test, r_test = prices.iloc[test_slice], returns.iloc[test_slice]
        if len(p_test) == 0:
            continue

        port_rets, _, _ = backtest_network_momentum(
            prices=pd.concat([p_train, p_test]), returns=pd.concat([r_train, r_test]), config=config
        )

        oos_rets = port_rets.loc[p_test.index[0] : p_test.index[-1]]
        all_rets.append(oos_rets)
    return pd.concat(all_rets).sort_index()


def sharpe_ratio(returns: pd.Series, ann_factor: int = 252, risk_free: float = 0.0) -> float:
    """Compute annualised Sharpe ratio."""
    excess = returns - risk_free / ann_factor
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return (mu / sigma) * np.sqrt(ann_factor)
