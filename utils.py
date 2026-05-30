"""
utils.py — Feature engineering, sequence creation, scaling, and ensemble helpers.

All pipeline scripts (train_model.py, app.py) import from here.
FEATURES list is the single source of truth — never duplicate it.

v5 changes (28 features — +4 macro):
  * KSE100_Ret    — KSE-100 index daily log return (market-wide signal)
  * PKRUSD_Ret    — PKR/USD daily log return (currency / macro risk)
  * Oil_Ret       — Brent crude daily log return (energy sector + inflation)
  * MacroMomentum — 20-day rolling composite of the three macro signals

v4 changes (24 features):
  * OBV_Norm    — On Balance Volume rolling z-score (volume intelligence)
  * VWAP_Dev    — Price deviation from 20d VWAP (institutional price anchor)
  * RegimeScore — Composite bull/bear/vol regime score (−1 to +1)
  * RelVol      — Log relative volume vs 20d average (volume spike detection)

v3 changes:
  * Log returns replace simple % returns (better ML stability + compounding math)
  * Added: CumLogRet7, CumLogRet30, RollVol20 (momentum + regime features)
  * compute_direction_label uses 5-day log-return with 1.5% threshold
  * Financial helpers: compute_cagr, compute_max_drawdown, compute_sharpe
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ── Feature list (single source of truth — 24 features) ──────────────────────
# Log returns (not simple %) — additive, stationary, better for ML
FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "LogReturn",                                # log(close/prev_close)
    "EMA20", "EMA50", "SMA20", "SMA50",
    "RSI", "Volume_Change",
    "MACD", "MACD_Signal", "BB_Width", "ATR", "Momentum",
    "CumLogRet7",                               # 7-day cumulative log return
    "CumLogRet30",                              # 30-day cumulative log return
    "RollVol20",                                # 20-day rolling volatility
    # v4 additions — volume intelligence + market regime
    "OBV_Norm",                                 # OBV rolling z-score (scale-invariant)
    "VWAP_Dev",                                 # % deviation from 20d VWAP
    "RegimeScore",                              # bull/bear/vol regime composite (−1→+1)
    "RelVol",                                   # log(volume / 20d avg volume)
    # v5 macro features — market-wide context
    "KSE100_Ret",                               # KSE-100 index daily log return
    "PKRUSD_Ret",                               # PKR/USD daily log return (currency risk)
    "Oil_Ret",                                  # Brent crude oil daily log return
    "MacroMomentum",                            # 20-day composite macro momentum
]

SEQUENCE_LENGTH  = 60     # 60-day rolling window
FORWARD_DAYS     = 5      # predict 5-day-ahead trend
TREND_THRESHOLD  = 0.015  # 1.5 % log-return threshold — filters micro-noise


# ── Technical indicators ──────────────────────────────────────────────────────

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range normalised by Close (dimensionless)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean() / close.replace(0, np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 20 technical / return features.
    Input:  Open, High, Low, Close, Volume
    Output: clean DataFrame with no NaN or Inf in FEATURES columns.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close  = df["Close"]
    volume = df["Volume"]

    # ── Moving averages ───────────────────────────────────────────────────────
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()

    # ── Log return (single-day) ───────────────────────────────────────────────
    # log(Ct / Ct-1) — additive, stationary, naturally models compounding
    df["LogReturn"] = np.log(close / close.shift(1).replace(0, np.nan))

    # ── Volume change ─────────────────────────────────────────────────────────
    df["Volume_Change"] = np.log(volume / volume.shift(1).replace(0, np.nan))

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["RSI"] = calculate_rsi(close)

    # ── MACD (normalised by price so it's scale-invariant) ───────────────────
    ema12            = close.ewm(span=12, adjust=False).mean()
    ema26            = close.ewm(span=26, adjust=False).mean()
    df["MACD"]       = (ema12 - ema26) / close.replace(0, np.nan)
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ── Bollinger Band width ──────────────────────────────────────────────────
    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_Width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)

    # ── ATR (normalised) ──────────────────────────────────────────────────────
    df["ATR"] = calculate_atr(df)

    # ── 5-day price momentum ──────────────────────────────────────────────────
    df["Momentum"] = close.pct_change(5)

    # ── Cumulative log returns (momentum memory) ──────────────────────────────
    log_ret          = df["LogReturn"]
    df["CumLogRet7"]  = log_ret.rolling(7).sum()    # short-term momentum
    df["CumLogRet30"] = log_ret.rolling(30).sum()   # medium-term momentum

    # ── Rolling realised volatility (vol-regime feature) ─────────────────────
    df["RollVol20"] = log_ret.rolling(20).std()

    # ── v4: OBV normalised (rolling z-score, scale-invariant) ─────────────────
    # On Balance Volume accumulates volume in the direction of price movement
    obv_direction  = np.sign(close.diff()).fillna(0)
    obv_raw        = (obv_direction * volume).cumsum()
    obv_mu         = obv_raw.rolling(60, min_periods=20).mean()
    obv_sigma      = obv_raw.rolling(60, min_periods=20).std().replace(0, np.nan)
    df["OBV_Norm"] = ((obv_raw - obv_mu) / obv_sigma).clip(-4, 4)

    # ── v4: VWAP deviation (price vs rolling 20d VWAP) ────────────────────────
    typical_price  = (df["High"] + df["Low"] + close) / 3.0
    vol_cumsum20   = volume.rolling(20).sum()
    vwap20         = (typical_price * volume).rolling(20).sum() / vol_cumsum20.replace(0, np.nan)
    df["VWAP_Dev"] = ((close - vwap20) / vwap20.replace(0, np.nan)).clip(-0.5, 0.5)

    # ── v4: Market regime score (composite −1 to +1) ───────────────────────────
    # Three signals: EMA trend (+0.5), momentum direction (+0.3), inverse vol (+0.2)
    ema_bull   = (df["EMA20"] > df["EMA50"]).astype(float) * 2 - 1   # +1 / −1
    mom_sign   = np.sign(df["CumLogRet30"].fillna(0))                 # +1 / 0 / −1
    rv_mu      = df["RollVol20"].rolling(60, min_periods=20).mean()
    rv_sigma   = df["RollVol20"].rolling(60, min_periods=20).std().replace(0, np.nan)
    vol_z      = ((df["RollVol20"] - rv_mu) / rv_sigma).fillna(0).clip(-3, 3)
    vol_regime = (-vol_z / 3.0)                                       # high vol → −1
    df["RegimeScore"] = (ema_bull * 0.5 + mom_sign * 0.3 + vol_regime * 0.2).clip(-1, 1)

    # ── v4: Relative Volume log-ratio ─────────────────────────────────────────
    vol_ma20    = volume.rolling(20).mean()
    rel         = (volume / vol_ma20.replace(0, np.nan)).clip(0.01, 100)
    df["RelVol"] = np.log(rel)

    # ── v5: Macro features — filled later by inject_macro_features() ──────────
    # Initialise to 0 here; train_model.py replaces them after merging macro data
    for _macro_col in ["KSE100_Ret", "PKRUSD_Ret", "Oil_Ret", "MacroMomentum"]:
        if _macro_col not in df.columns:
            df[_macro_col] = 0.0

    # ── Preserve date index as a column before resetting ─────────────────────
    if hasattr(df.index, "date") or str(df.index.dtype).startswith("datetime"):
        df["Date"] = pd.to_datetime(df.index)

    # ── Scrub infinities, drop incomplete rows ────────────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df


# ── Direction label (5-day log-return trend) ──────────────────────────────────

def compute_direction_label(df: pd.DataFrame,
                             forward: int = FORWARD_DAYS,
                             threshold: float = TREND_THRESHOLD) -> np.ndarray:
    """
    1 if the 5-day forward log-return > +threshold (1.5 %), else 0.

    Using log-returns instead of simple %:
      * Additive over time — matches compound growth math exactly
      * Better ML stability (approximately normally distributed)
      * 1.5 % threshold kills micro-noise while still giving ~45 % bullish labels
    Last `forward` rows are 0 (no future data available).
    """
    close  = df["Close"].values.astype(np.float64)
    n      = len(close)
    labels = np.zeros(n, dtype=np.float32)

    for i in range(n - forward):
        log_ret   = np.log(close[i + forward] / (close[i] + 1e-12))
        labels[i] = 1.0 if log_ret > threshold else 0.0

    return labels


# ── Sequence creation ─────────────────────────────────────────────────────────

def create_sequences(data: np.ndarray, labels: np.ndarray,
                     seq_len: int = SEQUENCE_LENGTH,
                     step: int = 1):
    """
    Rolling window → GRU sequences.
    `step` controls stride between consecutive windows.
      step=1 → densest; step=3 → ~3× fewer samples, ~3× faster training.
    Returns X (n, seq_len, n_features) and y (n,) — both float32.
    """
    X, y = [], []
    for i in range(0, len(data) - seq_len, step):
        X.append(data[i : i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Macro data loader ─────────────────────────────────────────────────────────

_MACRO_CACHE: dict = {}   # module-level cache so we only download once

def load_macro_data(start: str = "2013-01-01") -> pd.DataFrame:
    """
    Downloads and returns a daily DataFrame with macro features:
      KSE100_Ret    — KSE-100 index log return
      PKRUSD_Ret    — PKR/USD exchange rate log return
      Oil_Ret       — Brent crude (BZ=F) log return
      MacroMomentum — 20-day rolling mean of the three signals combined

    Falls back to zeros if any ticker fails (network error, delisted, etc.)
    so feature engineering never crashes.
    """
    global _MACRO_CACHE
    if _MACRO_CACHE:
        return _MACRO_CACHE["df"]

    import yfinance as yf   # lazy import — utils may be used without yf

    MACRO_TICKERS = {
        "KSE100_Ret":  "^KSE",        # KSE-100 index
        "PKRUSD_Ret":  "PKR=X",       # PKR per USD
        "Oil_Ret":     "BZ=F",        # Brent crude futures
    }

    series: dict[str, pd.Series] = {}
    for col, ticker in MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                raise ValueError("empty")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close_col = raw["Close"].squeeze()
            log_ret   = np.log(close_col / close_col.shift(1))
            # Winsorise at ±10 % to remove currency crises / data glitches
            log_ret   = log_ret.clip(-0.10, 0.10)
            series[col] = log_ret
            print(f"  Macro [{ticker}] … OK ({len(log_ret):,} rows)")
        except Exception as exc:
            print(f"  Macro [{ticker}] … SKIP ({exc})")
            series[col] = pd.Series(dtype=float)

    # Align all series on a common daily date index
    all_idx = pd.date_range(start=start, end=pd.Timestamp.today(), freq="B")
    macro = pd.DataFrame(index=all_idx)
    for col, s in series.items():
        if len(s):
            macro[col] = s.reindex(macro.index).ffill().fillna(0.0)
        else:
            macro[col] = 0.0

    # MacroMomentum: 20-day rolling mean of all three signals blended equally
    macro["MacroMomentum"] = (
        macro[["KSE100_Ret", "PKRUSD_Ret", "Oil_Ret"]]
        .rolling(20, min_periods=5)
        .mean()
        .mean(axis=1)
        .fillna(0.0)
    )

    _MACRO_CACHE["df"] = macro
    return macro


def inject_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-loaded macro data into a per-stock feature DataFrame.
    Requires df to have a 'Date' column (set by engineer_features).
    If dates don't align (e.g., US stocks on PSX holidays) uses ffill.
    """
    macro = load_macro_data()
    df = df.copy()

    if "Date" not in df.columns:
        # If no Date column, macro features stay 0
        for col in ["KSE100_Ret", "PKRUSD_Ret", "Oil_Ret", "MacroMomentum"]:
            df[col] = 0.0
        return df

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    macro_reset = macro.reset_index().rename(columns={"index": "Date"})
    macro_reset["Date"] = pd.to_datetime(macro_reset["Date"]).dt.normalize()

    df = df.merge(
        macro_reset[["Date", "KSE100_Ret", "PKRUSD_Ret", "Oil_Ret", "MacroMomentum"]],
        on="Date", how="left", suffixes=("_old", "")
    )
    # Drop old stub columns if merge produced duplicates
    for col in ["KSE100_Ret", "PKRUSD_Ret", "Oil_Ret", "MacroMomentum"]:
        old_col = col + "_old"
        if old_col in df.columns:
            df.drop(columns=[old_col], inplace=True)

    # Forward-fill then zero-fill any remaining NaN (weekends, holidays)
    for col in ["KSE100_Ret", "PKRUSD_Ret", "Oil_Ret", "MacroMomentum"]:
        df[col] = df[col].ffill().fillna(0.0)

    return df


# ── Scaler helpers ────────────────────────────────────────────────────────────

def fit_and_save_scaler(df: pd.DataFrame, scaler_path: str) -> MinMaxScaler:
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[FEATURES].astype(np.float32))
    joblib.dump(scaler, scaler_path)
    return scaler


def load_scaler(scaler_path: str) -> MinMaxScaler:
    return joblib.load(scaler_path)


def scale_features(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(df[FEATURES].astype(np.float32)).astype(np.float32)


# ── Financial analysis helpers ────────────────────────────────────────────────

def compute_cagr(df: pd.DataFrame, years: float | None = None) -> float:
    """
    Compound Annual Growth Rate from Close price history.
    CAGR = (end / start)^(1/years) - 1
    """
    closes = df["Close"].dropna().values
    if len(closes) < 2:
        return 0.0
    start  = float(closes[0])
    end    = float(closes[-1])
    if years is None:
        years = max(len(closes) / 252, 0.01)
    if start <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def compute_max_drawdown(df: pd.DataFrame) -> float:
    """
    Maximum peak-to-trough drawdown from Close price history.
    Returns a negative percentage, e.g. -0.42 means -42 % drawdown.
    """
    closes = df["Close"].dropna().values.astype(np.float64)
    if len(closes) < 2:
        return 0.0
    peak   = np.maximum.accumulate(closes)
    dd     = (closes - peak) / np.where(peak > 0, peak, 1e-9)
    return float(dd.min())


def compute_cumulative_return(df: pd.DataFrame) -> float:
    """Total return from first to last Close price."""
    closes = df["Close"].dropna().values
    if len(closes) < 2 or closes[0] <= 0:
        return 0.0
    return float((closes[-1] - closes[0]) / closes[0])


def compute_annual_volatility(df: pd.DataFrame) -> float:
    """Annualised realised volatility from log returns."""
    if "LogReturn" in df.columns:
        lr = df["LogReturn"].dropna()
    else:
        lr = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return float(lr.std() * np.sqrt(252)) if len(lr) > 1 else 0.0


def build_compound_curve(
    start: float,
    daily_log_ret: float,
    annual_vol: float,
    years: int,
    n_steps: int = 252,
) -> np.ndarray:
    """
    Deterministic compound growth curve used for the portfolio forecast chart.
    Returns an array of `years * n_steps` price levels.
    daily_log_ret — expected daily log return (= CAGR-equivalent divided by 252)
    """
    steps = years * n_steps
    return start * np.exp(daily_log_ret * np.arange(steps + 1))


# ── Long-term price forecast ──────────────────────────────────────────────────

def clamp_growth_rate(rate: float, lo: float = -0.20, hi: float = 0.35) -> float:
    return max(lo, min(hi, rate))


# ── Dynamic Sequential Compounding Engine ─────────────────────────────────────

_PERIODS = {"yearly": 1, "monthly": 12, "weekly": 52}


def _period_log_return(
    i: int,
    periods_per_year: int,
    base_annual: float,
    confidence: float,
    trend_strength: float,
    ann_vol_frac: float,
    sentiment_score: float,
    market_regime: float,
    historical_cagr: float,
) -> float:
    """
    Compute the expected log return for a single compounding period.

    Components that change each period (making growth truly dynamic):
      1. Trend persistence  — AI confidence edge decays as the signal ages
      2. Mean reversion     — rate gravitates back toward historical CAGR
      3. Sentiment decay    — news impact fades within ~1 year
      4. Regime boost/drag  — current bull/bear context
      5. Volatility drag    — Ito correction (-½σ²) ensures geometric accuracy
    """
    # 1. Base blend: historical anchor (40%) + AI signal (60%)
    ai_signal   = trend_strength * 0.10 + (confidence - 0.5) * 0.22
    base_annual = 0.40 * historical_cagr + 0.60 * ai_signal

    # 2. Trend persistence: AI edge decays with half-life = 1 year
    half_life      = max(periods_per_year, 1)
    trend_decay    = float(np.exp(-i / half_life))
    trend_adj      = (confidence - 0.5) * 0.12 * trend_decay

    # 3. Mean reversion: rate pulls toward long-run historical CAGR over time
    #    (reaches 60% weight by year 3)
    mr_weight  = min(0.60, i / (3.0 * periods_per_year))
    mean_rev   = mr_weight * (historical_cagr - base_annual)

    # 4. Sentiment adjustment: news impact fades within ~1 year
    sent_decay = float(np.exp(-i / max(periods_per_year, 1)))
    sent_adj   = (sentiment_score - 0.5) * 0.08 * sent_decay

    # 5. Market regime boost/drag (from recent 30-day cumulative log return)
    #    +5% regime → +1% annual boost; decays over 2 years
    regime_decay = float(np.exp(-i / (2.0 * periods_per_year)))
    regime_adj   = market_regime * 0.20 * regime_decay

    # Combine into annual rate then clamp
    annual_rate = clamp_growth_rate(
        base_annual + trend_adj + mean_rev + sent_adj + regime_adj
    )

    # Scale to per-period, then subtract Ito variance drag
    vol_per_period = ann_vol_frac * float(np.sqrt(1.0 / periods_per_year))
    vol_drag       = 0.5 * vol_per_period ** 2
    return annual_rate / periods_per_year - vol_drag


def dynamic_compound_forecast(
    current_price: float,
    years: int,
    historical_cagr: float,
    confidence: float,
    trend_strength: float,
    ann_vol: float,                 # annualised vol as a fraction (e.g. 0.30)
    sentiment_score: float = 0.5,   # 0 = very bearish … 1 = very bullish
    market_regime: float  = 0.0,   # normalised, e.g. CumLogRet30 clipped to ±0.15
    frequency: str        = "monthly",
    n_simulations: int    = 300,
    seed: int             = 42,
) -> dict:
    """
    Dynamic Sequential Compounding Engine.

    Each period's growth rate is recalculated from scratch — it does NOT
    use a single fixed rate applied over the entire horizon.  Prices compound
    iteratively: P[t+1] = P[t] * exp(r[t]).

    Args:
        frequency: "yearly" | "monthly" | "weekly"

    Returns dict with keys:
        path          – deterministic expected price at every period (list)
        upper         – P90 uncertainty band (list, same length)
        lower         – P10 uncertainty band (list, same length)
        period_labels – x-axis labels (list of str)
        f1y, f3y, f5y – expected prices at 1/3/5 years
        annual_rates  – effective annualised rate at each period (list)
        frequency     – the compounding frequency used
    """
    ppy    = _PERIODS.get(frequency, 12)   # periods per year
    total  = years * ppy                   # total periods
    vol_f  = ann_vol                       # already a fraction

    rng = np.random.default_rng(seed)

    # ── Deterministic expected path ──────────────────────────────────────────
    path         = [current_price]
    annual_rates = []

    for i in range(1, total + 1):
        r = _period_log_return(
            i, ppy, 0.0,   # base_annual is computed inside; pass 0 — ignored
            confidence, trend_strength, vol_f,
            sentiment_score, market_regime, historical_cagr,
        )
        annual_rates.append(r * ppy)           # annualised for display
        path.append(path[-1] * float(np.exp(r)))

    # ── Monte Carlo uncertainty bands ────────────────────────────────────────
    sim_ends = np.empty((n_simulations, total + 1), dtype=np.float64)
    for s in range(n_simulations):
        prices = [current_price]
        for i in range(1, total + 1):
            r      = _period_log_return(
                i, ppy, 0.0,
                confidence, trend_strength, vol_f,
                sentiment_score, market_regime, historical_cagr,
            )
            noise  = rng.normal(0.0, vol_f / float(np.sqrt(ppy)))
            prices.append(prices[-1] * float(np.exp(r + noise)))
        sim_ends[s] = prices

    upper = np.percentile(sim_ends, 85, axis=0).tolist()
    lower = np.percentile(sim_ends, 15, axis=0).tolist()

    # ── Extract 1Y / 3Y / 5Y prices from the deterministic path ──────────────
    def _at_year(yr: int) -> float:
        idx = min(yr * ppy, len(path) - 1)
        return path[idx]

    f1y = _at_year(1)
    f3y = _at_year(3) if years >= 3 else path[-1]
    f5y = _at_year(5) if years >= 5 else path[-1]

    # ── Period labels for x-axis ──────────────────────────────────────────────
    if frequency == "yearly":
        labels = [f"Y{i}" if i > 0 else "Now" for i in range(total + 1)]
    elif frequency == "monthly":
        labels = []
        for i in range(total + 1):
            if i == 0:
                labels.append("Now")
            elif i % 12 == 0:
                labels.append(f"Y{i // 12}")
            else:
                labels.append("")
    else:  # weekly
        labels = []
        for i in range(total + 1):
            if i == 0:
                labels.append("Now")
            elif i % 52 == 0:
                labels.append(f"Y{i // 52}")
            else:
                labels.append("")

    return {
        "path":         path,
        "upper":        upper,
        "lower":        lower,
        "period_labels": labels,
        "f1y":          f1y,
        "f3y":          f3y,
        "f5y":          f5y,
        "annual_rates": annual_rates,
        "frequency":    frequency,
        "ppy":          ppy,
    }


def forecast_price(
    current_price: float,
    trend_strength: float,
    ensemble_confidence: float,
    years: int,
    historical_cagr: float = 0.0,
) -> float:
    """
    Legacy single-value wrapper — used as a fallback only.
    Prefer dynamic_compound_forecast for all new display logic.
    """
    ai_component = (trend_strength * 0.10) + (ensemble_confidence - 0.5) * 0.22
    blended_rate = clamp_growth_rate(0.40 * historical_cagr + 0.60 * ai_component)
    return current_price * ((1 + blended_rate) ** years)


# ── Ensemble inference helper ─────────────────────────────────────────────────

def ensemble_predict(df: pd.DataFrame, dl_model, xgb_model, scaler,
                     dl_weight: float = 0.55, xgb_weight: float = 0.45) -> dict:
    """Returns {'confidence': float, 'dl': float, 'xgb': float}."""
    result = {"confidence": 0.5, "dl": 0.5, "xgb": 0.5}
    try:
        scaled = scale_features(df, scaler)

        if dl_model is not None and len(scaled) >= SEQUENCE_LENGTH:
            seq = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(FEATURES))
            result["dl"] = float(dl_model.predict(seq, verbose=0)[0][0])

        if xgb_model is not None:
            xrow = scaled[-1].reshape(1, -1)
            result["xgb"] = float(xgb_model.predict_proba(xrow)[0][1])

        dl_w  = dl_weight  if dl_model  is not None else 0
        xgb_w = xgb_weight if xgb_model is not None else 0
        total = dl_w + xgb_w
        if total > 0:
            result["confidence"] = (
                result["dl"] * dl_w + result["xgb"] * xgb_w
            ) / total
    except Exception:
        pass
    return result


# ── Advanced volume indicators ────────────────────────────────────────────────

def compute_advanced_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add institutional-grade volume analysis columns.
    Returns a copy with: OBV, VWAP (rolling 20), RelVol, VolSpike, ADLine.
    These are DISPLAY-only — they are not in the FEATURES list.
    """
    df     = df.copy()
    close  = df["Close"]
    volume = df["Volume"]
    high   = df["High"]
    low    = df["Low"]

    # On Balance Volume
    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * volume).cumsum()

    # Rolling 20-day VWAP
    tp         = (high + low + close) / 3.0
    vol_roll20 = volume.rolling(20).sum()
    df["VWAP"] = (tp * volume).rolling(20).sum() / vol_roll20.replace(0, np.nan)

    # Relative Volume vs 20-day average
    vol_ma20    = volume.rolling(20).mean()
    df["RelVol"] = volume / vol_ma20.replace(0, np.nan)

    # Volume Spike — z-score
    vol_std20     = volume.rolling(20).std()
    df["VolSpike"] = (volume - vol_ma20) / vol_std20.replace(0, np.nan)

    # Accumulation / Distribution Line
    clv          = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    df["ADLine"] = (clv * volume).cumsum()

    return df


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Dual z-score anomaly detection on log returns and volume, optionally
    augmented by Isolation Forest.
    Adds: 'AnomalyScore' (float) and 'Anomaly' (bool) columns.
    """
    df  = df.copy()
    lr  = df["LogReturn"] if "LogReturn" in df.columns else np.log(df["Close"] / df["Close"].shift(1))
    vol = df["Volume"]

    z_lr  = (lr  - lr.rolling(20).mean())  / lr.rolling(20).std().replace(0, np.nan)
    z_vol = (vol - vol.rolling(20).mean()) / vol.rolling(20).std().replace(0, np.nan)

    df["AnomalyScore"] = z_lr.abs().fillna(0) + z_vol.abs().fillna(0) * 0.5
    df["Anomaly"]      = (
        (z_lr.abs() > z_threshold) | (z_vol.abs() > z_threshold * 1.5)
    ).fillna(False)

    try:
        from sklearn.ensemble import IsolationForest  # type: ignore
        feats = ["LogReturn", "Volume_Change"] if "Volume_Change" in df.columns else ["LogReturn"]
        X     = df[feats].fillna(0).values
        iso   = IsolationForest(contamination=0.03, random_state=42, n_jobs=1)
        df["Anomaly"] = df["Anomaly"] | (iso.fit_predict(X) == -1)
    except Exception:
        pass

    return df


# ── Realistic walk-forward backtest engine ────────────────────────────────────

def run_backtest(
    df:              pd.DataFrame,
    gru_model,
    xgb_model,
    scaler,
    initial_capital: float = 100_000.0,
    txn_cost:        float = 0.002,    # 0.2 % commission per trade (PSX typical)
    slippage:        float = 0.001,    # 0.1 % market impact
    conf_threshold:  float = 0.55,     # minimum confidence to enter/exit
    window_days:     int   = 400,      # how many days to backtest over
) -> dict:
    """
    Realistic backtest with transaction costs, slippage, and ensemble signals.
    Uses batched XGBoost + batched GRU inference for speed.

    Returns a dict with equity curve, Sharpe, max drawdown, win rate, trade log.
    Includes a buy-and-hold benchmark for comparison.
    """
    if scaler is None or (gru_model is None and xgb_model is None):
        return {}

    subset = df.tail(window_days + SEQUENCE_LENGTH).copy().reset_index(drop=True)
    if len(subset) < SEQUENCE_LENGTH + 20:
        return {}

    scaled = scale_features(subset, scaler)
    closes = subset["Close"].values.astype(np.float64)
    dates  = (subset["Date"].dt.strftime("%Y-%m-%d").values
              if "Date" in subset.columns else np.arange(len(subset)).astype(str))

    # ── Batch XGBoost predictions ─────────────────────────────────────────────
    xgb_probs = np.full(len(scaled), 0.5)
    if xgb_model is not None:
        try:
            xgb_probs = xgb_model.predict_proba(scaled)[:, 1].astype(np.float64)
        except Exception:
            pass

    # ── Batch GRU predictions ─────────────────────────────────────────────────
    gru_probs = np.full(len(scaled), 0.5)
    if gru_model is not None:
        try:
            n_seq = len(scaled) - SEQUENCE_LENGTH
            seqs  = np.array(
                [scaled[i: i + SEQUENCE_LENGTH] for i in range(n_seq)],
                dtype=np.float32,
            )
            preds = gru_model.predict(seqs, verbose=0, batch_size=64).flatten()
            gru_probs[SEQUENCE_LENGTH:] = preds.astype(np.float64)
        except Exception:
            pass

    # ── Combine ───────────────────────────────────────────────────────────────
    if gru_model is not None and xgb_model is not None:
        conf_arr = gru_probs * 0.55 + xgb_probs * 0.45
    elif gru_model is not None:
        conf_arr = gru_probs
    else:
        conf_arr = xgb_probs

    # ── Simulate trading ──────────────────────────────────────────────────────
    capital      = float(initial_capital)
    shares       = 0.0
    in_trade     = False
    entry_px     = 0.0
    wins = losses = total_trades = 0
    equity_vals  = []
    equity_dates = []
    trades_log   = []

    for i in range(SEQUENCE_LENGTH, len(closes) - 1):
        conf  = float(conf_arr[i])
        price = float(closes[i])

        # Buy signal
        if conf >= conf_threshold and not in_trade and capital > price:
            buy_px   = price * (1.0 + slippage)
            cost     = capital * txn_cost
            shares   = (capital - cost) / buy_px
            capital  = 0.0
            in_trade = True
            entry_px = buy_px
            trades_log.append({
                "date": str(dates[i]), "type": "BUY",
                "price": round(buy_px, 2), "conf": round(conf, 3),
            })

        # Sell signal
        elif conf <= (1.0 - conf_threshold) and in_trade:
            sell_px  = price * (1.0 - slippage)
            capital  = shares * sell_px * (1.0 - txn_cost)
            pnl_pct  = (sell_px - entry_px) / entry_px * 100.0
            shares   = 0.0
            in_trade = False
            total_trades += 1
            wins  += 1 if pnl_pct > 0 else 0
            losses += 0 if pnl_pct > 0 else 1
            trades_log.append({
                "date": str(dates[i]), "type": "SELL",
                "price": round(sell_px, 2), "conf": round(conf, 3),
                "pnl_pct": round(pnl_pct, 2),
            })

        mtm = capital + shares * price
        equity_vals.append(mtm)
        equity_dates.append(str(dates[i]))

    # Close open position
    if in_trade and shares > 0:
        final_px = float(closes[-1]) * (1.0 - slippage)
        capital  = shares * final_px * (1.0 - txn_cost)

    if not equity_vals:
        return {}

    # Buy-and-hold benchmark
    bh_start = float(closes[SEQUENCE_LENGTH])
    bh_vals  = (
        initial_capital / bh_start * closes[SEQUENCE_LENGTH: SEQUENCE_LENGTH + len(equity_vals)]
    ).tolist()

    # Risk metrics
    eq_arr = np.array(equity_vals, dtype=np.float64)
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / np.where(peak > 0, peak, 1e-9)
    max_dd = float(dd.min()) * 100.0

    rets   = pd.Series(equity_vals).pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    total_ret = (equity_vals[-1] / initial_capital - 1.0) * 100.0
    win_rate  = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        "equity_vals":   [float(v) for v in equity_vals],
        "equity_dates":  equity_dates,
        "bh_vals":       [float(v) for v in bh_vals],
        "initial":       initial_capital,
        "final":         float(equity_vals[-1]),
        "total_return":  total_ret,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "total_trades":  total_trades,
        "win_rate":      win_rate,
        "trades":        trades_log[-30:],
    }


# ── AI Screener helper ────────────────────────────────────────────────────────

def quick_screen(df: pd.DataFrame, gru_model, xgb_model, scaler) -> dict:
    """
    Compact screening dict for one stock.
    Returns: price, rsi, ma_bull, momentum, ann_vol, confidence, signal, score.
    """
    if df is None or df.empty:
        return {}

    rsi   = float(df["RSI"].iloc[-1])        if "RSI"      in df.columns else 50.0
    ema20 = float(df["EMA20"].iloc[-1])       if "EMA20"    in df.columns else 0.0
    ema50 = float(df["EMA50"].iloc[-1])       if "EMA50"    in df.columns else 0.0
    mom   = float(df["Momentum"].iloc[-1]) * 100.0 if "Momentum" in df.columns else 0.0
    rvol  = (float(df["RollVol20"].iloc[-1]) * 100.0 * float(np.sqrt(252))
             if "RollVol20" in df.columns else 0.0)
    close = float(df["Close"].iloc[-1])

    conf = 0.5
    if scaler is not None and (gru_model is not None or xgb_model is not None):
        ens  = ensemble_predict(df, gru_model, xgb_model, scaler)
        conf = ens["confidence"]

    score = 0
    if   conf >= 0.65: score += 2
    elif conf >= 0.55: score += 1
    elif conf <= 0.35: score -= 2
    elif conf <= 0.45: score -= 1

    if   rsi <= 30: score += 2
    elif rsi >= 70: score -= 2

    score += (1 if ema20 > ema50 else -1)

    if   mom >  2.0: score += 1
    elif mom < -2.0: score -= 1

    if   score >= 4:  sig = "STRONG BUY"
    elif score >= 2:  sig = "BUY"
    elif score >= -1: sig = "HOLD"
    elif score >= -3: sig = "SELL"
    else:             sig = "STRONG SELL"

    return {
        "price":      close,
        "rsi":        rsi,
        "ma_bull":    ema20 > ema50,
        "momentum":   mom,
        "ann_vol":    rvol,
        "confidence": conf,
        "signal":     sig,
        "score":      score,
    }
