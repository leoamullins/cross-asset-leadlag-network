import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

WINDOW = 252
REBAL_FREQ = "M"
MAX_LAG = 5


# calc corr
def corr_at_lag(x: pd.Series, y: pd.Series, lag: int) -> float:
    """
    Calculates the Pearson correlation coefficient between two pandas Series at a given lag. The function aligns the
    input series based on the specified lag, removes missing values, and computes the correlation.

    Args:
        x (pd.Series): The first time series.
        y (pd.Series): The second time series.
        lag (int): The lag value used to offset one of the series. Positive lag shifts `y` forward, negative lag shifts
            `x` forward, and zero lag computes the correlation without shifting.

    Returns:
        float: The Pearson correlation coefficient between the two input series at the specified lag. If there are fewer
        than 5 aligned non-missing points, returns NaN.
    """
    if lag > 0:
        x_, y_ = x.iloc[:-lag], y.iloc[lag:]
    elif lag < 0:
        x_, y_ = x.iloc[:lag], y.iloc[-lag:]
    else:
        x_, y_ = x, y

    # align and drop NaN
    m = x_.notna() & y_.notna()
    if m.sum() < 5:
        return np.nan
    return np.corrcoef(x_[m], y_[m])[0, 1]


def best_lag_corr(x: pd.Series, y: pd.Series, max_lag: int = MAX_LAG):
    """
    Calculates the best lag with the highest correlation between two time series up to a specified maximum lag. It iterates
    through the possible lags within the range [-max_lag, max_lag], computes the correlation for each lag, and identifies
    the lag with the highest absolute correlation. The result includes both the lag and the corresponding correlation value.

    Args:
        x (pd.Series): The first time series.
        y (pd.Series): The second time series.
        max_lag (int, optional): The maximum absolute lag to consider. Defaults to MAX_LAG.

    Returns:
        dict: A dictionary containing the best lag and its corresponding correlation with keys:
            - "lag" (int): The lag with the highest absolute correlation.
            - "corr" (float): The correlation value at the best lag.
    """
    best = {"lag": 0, "corr": np.nan}
    best_abs = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        r = corr_at_lag(x, y, lag)
        if np.isnan(r):
            continue
        if abs(r) > best_abs:
            best_abs = abs(r)
            best["lag"] = lag
            best["corr"] = r
    return best


def build_adj(
    returns: pd.DataFrame, max_lag: int = MAX_LAG, min_abs_corr: float = 0.15
):
    """
    Constructs an adjacency matrix representing the correlation structure among time series
    data with specified maximum lag and minimum absolute correlation threshold. The matrix
    reveals leading and following relationships between time series based on their
    calculated lags and correlations.

    Args:
        returns (pd.DataFrame): A DataFrame where each column represents a time series,
            and rows correspond to time steps.
        max_lag (int): The maximum lag to consider when calculating correlations between
            the time series. Defaults to MAX_LAG.
        min_abs_corr (float): The minimum absolute correlation value required to consider
            inclusion in the adjacency matrix. Defaults to 0.15.

    Returns:
        pd.DataFrame: An adjacency matrix where entries represent the correlation magnitude
            and direction between time series, with non-zero entries indicating meaningful
            relationships based on the provided threshold.
    """
    cols = list(returns.columns)
    N = len(cols)
    A = pd.DataFrame(0.0, index=cols, columns=cols)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            res = best_lag_corr(returns.iloc[:, i], returns.iloc[:, j], max_lag=max_lag)
            lag, r = res["lag"], res["corr"]

            # If lag > 0: i_t correlates with j_{t+lag}, so i leads j
            # If lag < 0: i_t correlates with j_{t+lag}, so j leads i
            if (r is not None) and (abs(r) >= min_abs_corr):
                if lag > 0:
                    # i leads j
                    A.iat[i, j] = r
                elif lag < 0:
                    # j leads i, so add edge from j to i
                    A.iat[j, i] = r
    return A


def leadlag_graph(
    A: pd.DataFrame,
    title="lead-lag network",
    max_edges: int = 80,
    node_score: str = "out_strength",
    layout: str = "circular",
    seed: int = 42,
    visualise: bool = True,
):
    """
    Generates a directed graph from a given adjacency matrix and visualizes it as a
    lead-lag network by filtering the most significant edges and applying a scoring
    metric to determine node importance.

    Args:
        A (pd.DataFrame): Input adjacency matrix representing the graph.
        title (str): Title for the visualized graph. Default is "lead-lag network".
        max_edges (int): Maximum number of edges to display, sorted by weight.
            Default is 80.
        node_score (str): Method for scoring nodes to determine importance. Options
            include "pagerank" or "out_strength". Default is "out_strength".
        layout (str): Layout algorithm for visualizing the graph. Options include
            "spring", "kamada_kawai", "circular". Default is "circular".
        seed (int): Random seed for layout reproducibility (used in spring layout
            algorithm). Default is 42.
        visualise (bool): Whether to display the graph visualization. Default is True.

    Returns:
        tuple: A tuple containing the filtered directed graph (nx.DiGraph) and a
        pandas Series representing the z-scored node importance scores (leaders and
        followers).
    """
    G = nx.from_pandas_adjacency(A, create_using=nx.DiGraph)

    edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    if not edges:
        print("No edges to display (A is empty).")
        return
    edges_sorted = sorted(edges, key=lambda x: abs(x[2]), reverse=True)[:max_edges]

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, w in edges_sorted:
        H.add_edge(u, v, weight=w)

    # node leadership
    if node_score == "pagerank":
        pr = nx.pagerank(H, alpha=0.9, weight="weight")
        score = pd.Series(pr)
    else:
        # out-strength (sum of outgoing weights)
        out_strength = {n: 0.0 for n in H.nodes()}
        for u, v, d in H.edges(data=True):
            out_strength[u] += d["weight"]
        score = pd.Series(out_strength)

    # normalise
    s = (score - score.mean()) / (score.std() if score.std() > 0 else 1.0)
    node_sizes = (s.clip(lower=0) + 0.2) * 1200  # leaders bigger
    node_colors = s  # positive=leaders, negative=followers

    # layout
    if layout == "spring":
        pos = nx.spring_layout(H, seed=seed, k=None, weight="weight")
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H, weight="weight")
    elif layout == "circular":
        pos = nx.circular_layout(H)
    else:
        pos = nx.spring_layout(H, seed=seed)

    # edge style
    weights = np.array([abs(H[u][v]["weight"]) for u, v in H.edges()])
    if weights.size:
        # scale edge widths
        ew = 1.0 + 4.0 * (weights - weights.min()) / (
            np.ptp(weights) if np.ptp(weights) > 0 else 1.0
        )
    else:
        ew = 1.5

    if visualise:
        plt.figure(figsize=(10, 7), dpi=80)
        nodes = nx.draw_networkx_nodes(
            H,
            pos,
            node_size=[node_sizes.get(n, 400) for n in H.nodes()],
            node_color=[node_colors.get(n, 0.0) for n in H.nodes()],
            cmap="coolwarm",
        )
        edges = nx.draw_networkx_edges(
            H, pos, arrowstyle="->", arrowsize=12, width=ew, edge_color="#555"
        )
        labels = nx.draw_networkx_labels(H, pos, font_size=9)

        cbar = plt.colorbar(nodes, shrink=0.8, pad=0.02)
        cbar.set_label("Leader score (z-scored)", rotation=270, labelpad=12)

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return H, s


# --- Optional: Numba-accelerated adjacency build ---
try:
    from numba import njit, prange  # type: ignore

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # fallback no-op decorator
        def _wrap(f):
            return f

        return _wrap

    def prange(*args, **kwargs):  # fallback to range
        return range(*args, **kwargs)


@njit(fastmath=True)
def _corr_numba(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    cnt = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for k in range(n):
        xv = x[k]
        yv = y[k]
        if not (np.isnan(xv) or np.isnan(yv)):
            cnt += 1
            sx += xv
            sy += yv
            sxx += xv * xv
            syy += yv * yv
            sxy += xv * yv
    if cnt < 5:
        return np.nan
    cov = sxy - (sx * sy) / cnt
    vx = sxx - (sx * sx) / cnt
    vy = syy - (sy * sy) / cnt
    if vx <= 0.0 or vy <= 0.0:
        return np.nan
    return cov / np.sqrt(vx * vy)


@njit(fastmath=True)
def _corr_at_lag_numba(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    n = x.shape[0]
    if lag > 0:
        return _corr_numba(x[: n - lag], y[lag:])
    elif lag < 0:
        lag2 = -lag
        return _corr_numba(x[lag2:], y[: n - lag2])
    else:
        return _corr_numba(x, y)


@njit(fastmath=True)
def _best_lag_corr_numba(x: np.ndarray, y: np.ndarray, max_lag: int):
    best_lag = 0
    best_corr = np.nan
    best_abs = -1.0
    for lag in range(-max_lag, max_lag + 1):
        r = _corr_at_lag_numba(x, y, lag)
        if np.isnan(r):
            continue
        ar = abs(r)
        if ar > best_abs:
            best_abs = ar
            best_lag = lag
            best_corr = r
    return best_lag, best_corr


@njit(parallel=True, fastmath=True)
def _build_adj_numba(
    returns_np: np.ndarray, max_lag: int, min_abs_corr: float
) -> np.ndarray:
    T, N = returns_np.shape
    A = np.zeros((N, N), dtype=np.float64)
    for i in prange(N):
        for j in range(N):
            if i == j:
                continue
            lag, r = _best_lag_corr_numba(returns_np[:, i], returns_np[:, j], max_lag)
            if not np.isnan(r) and abs(r) >= min_abs_corr:
                if lag > 0:
                    A[i, j] = r
                elif lag < 0:
                    A[j, i] = r
                # lag == 0 -> skip
    return A


def build_adj_fast(
    returns: pd.DataFrame, max_lag: int = MAX_LAG, min_abs_corr: float = 0.15
) -> pd.DataFrame:
    """Numba-accelerated adjacency builder. Falls back to Python version if Numba unavailable."""
    cols = list(returns.columns)
    returns_np = returns.values.astype(np.float64, copy=False)
    if _NUMBA_AVAILABLE:
        A_np = _build_adj_numba(returns_np, int(max_lag), float(min_abs_corr))
        return pd.DataFrame(A_np, index=cols, columns=cols)
    # Fallback to original implementation
    return build_adj(returns, max_lag=max_lag, min_abs_corr=min_abs_corr)
