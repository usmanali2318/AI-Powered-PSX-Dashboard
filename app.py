"""
app.py — PSX AI Dashboard v3
Ensemble: GRU + XGBoost  |  FinBERT Sentiment  |  5-day log-return prediction
Compounding: CAGR · Max Drawdown · Compound Growth Curve · Portfolio Simulation
Run: streamlit run app.py --server.port 5000
"""

import os
import warnings
import urllib.parse
from datetime import datetime

import feedparser
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from utils import (
    FEATURES, SEQUENCE_LENGTH,
    engineer_features, load_scaler, scale_features,
    forecast_price, calculate_rsi, ensemble_predict,
    compute_cagr, compute_max_drawdown, compute_cumulative_return,
    compute_annual_volatility, build_compound_curve,
    dynamic_compound_forecast,
    compute_advanced_volume, detect_anomalies,
    run_backtest, quick_screen,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSX AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background: #0a0d17;
    color: #e8ecf4;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Sidebar: fully scrollable, no empty top gap ── */
[data-testid="stSidebar"] {
    background: #0e1220 !important;
    border-right: 1px solid #1e2235;
    min-width: 260px !important;
    max-width: 300px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
/* Kill the blank header gap Streamlit injects above sidebar content */
[data-testid="stSidebar"] > div:first-child {
    padding: 0.6rem 1rem 0.6rem 1rem !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: auto !important;
    max-height: 100vh !important;
    margin-top: 0 !important;
}
/* Let the sidebar be exactly as tall as its content — no phantom space */
section[data-testid="stSidebar"] > div {
    overflow-y: auto !important;
    height: auto !important;
    max-height: 100vh !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* Tighten Streamlit's default vertical spacing in sidebar */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
    padding-bottom: 0 !important;
}
/* Zero out the last element's margin so nothing hangs below footer */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:last-child {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
/* Hide Streamlit's own sidebar header/logo area that creates blank space */
[data-testid="stSidebarHeader"],
[data-testid="stSidebarNavItems"],
[data-testid="stDecoration"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* remove the global overflow:visible that was blocking scroll */
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] input  { overflow: visible; }

.metric-card {
    background: #12172a;
    border: 1px solid #1e2640;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    height: 100%;
    min-height: 88px;
}
.metric-label {
    font-size: .67rem; color: #6b7494;
    text-transform: uppercase; letter-spacing: 1.2px;
    margin-bottom: 6px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.metric-value { font-size: 1.3rem; font-weight: 700; line-height: 1.2; }
.metric-sub   { font-size: .7rem; color: #6b7494; margin-top: 4px; }

.badge { display:inline-block; padding:5px 16px; border-radius:20px;
         font-weight:700; font-size:.9rem; letter-spacing:.5px; }
.badge-strong-buy  { background:#00c853; color:#000; }
.badge-buy         { background:#00e676; color:#000; }
.badge-hold        { background:#ffd740; color:#000; }
.badge-sell        { background:#ff6e40; color:#fff; }
.badge-strong-sell { background:#ff1744; color:#fff; }

.section-header {
    font-size:1rem; font-weight:600; color:#00d4aa;
    border-left:3px solid #00d4aa; padding-left:10px;
    margin:28px 0 14px 0;
}
.chart-label {
    font-size:.72rem; font-weight:600; color:#6b7494;
    text-transform:uppercase; letter-spacing:1.1px;
    margin-bottom:4px; padding-left:2px;
}
.news-card {
    background:#12172a; border:1px solid #1e2640; border-radius:10px;
    padding:12px 16px; margin-bottom:8px;
}
.news-title { font-size:.84rem; font-weight:500; line-height:1.4; }
.news-pos   { color:#00c853; font-size:.71rem; font-weight:600; margin-top:4px; }
.news-neg   { color:#ff1744; font-size:.71rem; font-weight:600; margin-top:4px; }
.news-neu   { color:#6b7494; font-size:.71rem; font-weight:600; margin-top:4px; }

.forecast-row {
    background:#12172a; border:1px solid #1e2640; border-radius:10px;
    padding:14px 18px; margin-bottom:8px;
    display:flex; justify-content:space-between; align-items:center;
}
.forecast-label { font-size:.78rem; color:#6b7494; }
.forecast-val   { font-size:1.05rem; font-weight:700; }
.forecast-pnl   { font-size:.82rem; font-weight:600; }

.status-ok   { background:#0d2e1c; border:1px solid #00c853; border-radius:8px;
               padding:10px 14px; color:#00c853; font-size:.82rem; }
.status-warn { background:#2e1f0d; border:1px solid #ffd740; border-radius:8px;
               padding:10px 14px; color:#ffd740; font-size:.82rem; }

hr { border-color:#1e2235 !important; }
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label { color:#9ba3bf !important; font-size:.78rem !important; }

/* ── Sidebar always open — hide the native collapse button ── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Stock universe ────────────────────────────────────────────────────────────
STOCK_LIST = {
    "FFC — Fauji Fertilizer":       "FFC.KA",
    "ENGRO — Engro Corp":           "ENGRO.KA",
    "EFERT — Engro Fertilizers":    "EFERT.KA",
    "MEBL — Meezan Bank":           "MEBL.KA",
    "HBL — Habib Bank":             "HBL.KA",
    "UBL — United Bank":            "UBL.KA",
    "MCB — MCB Bank":               "MCB.KA",
    "OGDC — Oil & Gas Dev Corp":    "OGDC.KA",
    "PPL — Pakistan Petroleum":     "PPL.KA",
    "PSO — Pakistan State Oil":     "PSO.KA",
    "SYS — Systems Ltd":            "SYS.KA",
    "TRG — TRG Pakistan":           "TRG.KA",
    "NETSOL — NetSol Technologies": "NETSOL.KA",
    "SAZEW — Pak Suzuki":           "SAZEW.KA",
    "ATLH — Atlas Honda":           "ATLH.KA",
    "HCAR — Honda Atlas Cars":      "HCAR.KA",
    "INDU — Indus Motor":           "INDU.KA",
    "LUCK — Lucky Cement":          "LUCK.KA",
    "DGKC — D.G. Khan Cement":      "DGKC.KA",
    "HUBC — Hub Power":             "HUBC.KA",
    "KEL — K-Electric":             "KEL.KA",
    "PTC — Pakistan Telecom":       "PTC.KA",
    "── International ──":          None,
    "AAPL — Apple Inc.":            "AAPL",
    "MSFT — Microsoft":             "MSFT",
    "GOOGL — Alphabet":             "GOOGL",
    "NVDA — NVIDIA":                "NVDA",
    "TSLA — Tesla":                 "TSLA",
}

# ── Sector mappings (for screener grouping) ───────────────────────────────────
SECTORS: dict[str, list[str]] = {
    "Fertilizer":    ["FFC.KA", "ENGRO.KA", "EFERT.KA"],
    "Banking":       ["MEBL.KA", "HBL.KA", "UBL.KA", "MCB.KA"],
    "Oil & Gas":     ["OGDC.KA", "PPL.KA", "PSO.KA"],
    "Technology":    ["SYS.KA", "TRG.KA", "NETSOL.KA"],
    "Auto":          ["SAZEW.KA", "ATLH.KA", "HCAR.KA", "INDU.KA"],
    "Cement":        ["LUCK.KA", "DGKC.KA"],
    "Power":         ["HUBC.KA", "KEL.KA"],
    "Telecom":       ["PTC.KA"],
    "International": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
}

# Short ticker → display label map for screener table
TICKER_LABELS: dict[str, str] = {v: k.split("—")[0].strip()
                                   for k, v in STOCK_LIST.items() if v}

GRU_PATH    = "models/gru_model.keras"
XGB_PATH    = "models/xgb_model.pkl"
SCALER_PATH = "data/scaler.pkl"

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start="2015-01-01", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            return None
        df = df[list(needed)].dropna()
        return engineer_features(df)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_models():
    gru, xgb, scaler = None, None, None
    if os.path.exists(GRU_PATH):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            gru = load_model(GRU_PATH)
        except Exception:
            pass
    if os.path.exists(XGB_PATH):
        try:
            xgb = joblib.load(XGB_PATH)
        except Exception:
            pass
    if os.path.exists(SCALER_PATH):
        try:
            scaler = load_scaler(SCALER_PATH)
        except Exception:
            pass
    return gru, xgb, scaler


# ── Signal engine ─────────────────────────────────────────────────────────────

def compute_signal(confidence: float, rsi: float, df: pd.DataFrame) -> str:
    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    macd  = float(df["MACD"].iloc[-1])
    score = 0

    if   confidence >= 0.65: score += 2
    elif confidence >= 0.55: score += 1
    elif confidence <= 0.35: score -= 2
    elif confidence <= 0.45: score -= 1

    if   rsi <= 30: score += 2
    elif rsi <= 45: score += 1
    elif rsi >= 70: score -= 2
    elif rsi >= 55: score -= 1

    score += (1 if ema20 > ema50 else -1)
    score += (1 if macd > 0 else -1)

    if   score >= 5:  return "STRONG BUY"
    elif score >= 2:  return "BUY"
    elif score >= -1: return "HOLD"
    elif score >= -4: return "SELL"
    else:             return "STRONG SELL"


def signal_badge(signal: str) -> str:
    cls = {
        "STRONG BUY":  "badge-strong-buy",
        "BUY":         "badge-buy",
        "HOLD":        "badge-hold",
        "SELL":        "badge-sell",
        "STRONG SELL": "badge-strong-sell",
    }.get(signal, "badge-hold")
    return f'<span class="badge {cls}">{signal}</span>'


# ── Risk & backtest ───────────────────────────────────────────────────────────

def compute_risk(df: pd.DataFrame) -> tuple[float, str]:
    vol = compute_annual_volatility(df)
    cat = "Low" if vol < 20 else ("Medium" if vol < 40 else "High")
    return vol, cat


def backtest_accuracy(df: pd.DataFrame, gru_model, xgb_model, scaler) -> float:
    try:
        subset = df.tail(200 + SEQUENCE_LENGTH).copy()
        scaled = scale_features(subset, scaler)
        closes = subset["Close"].values

        X_seq = [scaled[i : i + SEQUENCE_LENGTH]
                 for i in range(len(scaled) - SEQUENCE_LENGTH - 1)]
        if not X_seq:
            return 0.5

        X_arr     = np.array(X_seq, dtype=np.float32)
        true_dirs = (closes[SEQUENCE_LENGTH + 1 : SEQUENCE_LENGTH + 1 + len(X_seq)] >
                     closes[SEQUENCE_LENGTH     : SEQUENCE_LENGTH     + len(X_seq)]).astype(int)

        gru_prob = (gru_model.predict(X_arr, verbose=0).flatten()
                    if gru_model else np.full(len(X_seq), 0.5))
        xgb_prob = (xgb_model.predict_proba(
                        scaled[SEQUENCE_LENGTH : SEQUENCE_LENGTH + len(X_seq)])[:, 1]
                    if xgb_model else np.full(len(X_seq), 0.5))

        pred = ((gru_prob * 0.55 + xgb_prob * 0.45) >= 0.5).astype(int)
        n = min(len(pred), len(true_dirs))
        return float((pred[:n] == true_dirs[:n]).mean())
    except Exception:
        return 0.5


# ── Charts ────────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0d17",
    plot_bgcolor="#0a0d17",
    font=dict(color="#9ba3bf", size=12),
)


_DATE_AXIS = dict(
    type="date",
    tickformat="%b '%y",
    tickangle=-30,
    gridcolor="#1a1f30",
    tickfont=dict(size=10, color="#6b7494"),
    showgrid=True,
    rangeslider_visible=False,
)
_Y_AXIS = dict(gridcolor="#1a1f30", tickfont=dict(size=10, color="#6b7494"))
_LEGEND = dict(orientation="h", yanchor="top", y=-0.14,
               xanchor="left", x=0, font=dict(size=10, color="#9ba3bf"))


def _dates(df: pd.DataFrame):
    """Return the date series for use as x-axis values."""
    if "Date" in df.columns:
        return df["Date"]
    return list(range(len(df)))   # integer fallback


# ── Graph 1: Price Action ─────────────────────────────────────────────────────

def make_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + Volume — clean price action, no indicators."""
    dates = _dates(df)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=dates, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00c853", decreasing_fillcolor="#ff1744",
        line_width=1,
    ), row=1, col=1)

    vol_colors = [
        "#00c853" if c >= o else "#ff1744"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=dates, y=df["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.55,
    ), row=2, col=1)

    fig.update_layout(
        **CHART_LAYOUT, height=440,
        title=dict(text=f"<b>{ticker}</b> — Price Action",
                   font=dict(color="#e8ecf4", size=13), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=_LEGEND,
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(**_DATE_AXIS)
    fig.update_yaxes(**_Y_AXIS)
    fig.update_yaxes(title_text="Price", row=1, col=1,
                     title_font=dict(size=10, color="#6b7494"))
    fig.update_yaxes(title_text="Volume", row=2, col=1,
                     title_font=dict(size=10, color="#6b7494"))
    return fig


# ── Graph 2: Moving Averages / Trend ─────────────────────────────────────────

def make_ma_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Close price overlaid with MA50, MA200 and EMA20 for trend analysis."""
    dates  = _dates(df)
    ma50   = df["Close"].rolling(50).mean()
    ma200  = df["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=df["Close"], name="Close",
        line=dict(color="rgba(255,255,255,0.30)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=ma50, name="MA 50",
        line=dict(color="#ffab40", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=ma200, name="MA 200",
        line=dict(color="#40c4ff", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=df["EMA20"], name="EMA 20",
        line=dict(color="#ce93d8", width=1.6, dash="dot"),
    ))

    fig.update_layout(
        **CHART_LAYOUT, height=320,
        title=dict(text=f"<b>{ticker}</b> — Moving Averages & Trend",
                   font=dict(color="#e8ecf4", size=13), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=_LEGEND,
        yaxis=dict(**_Y_AXIS, title="Price",
                   title_font=dict(size=10, color="#6b7494")),
        xaxis=_DATE_AXIS,
    )
    return fig


# ── Graph 3: RSI ─────────────────────────────────────────────────────────────

def make_rsi_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """RSI (14) with overbought / oversold zones."""
    dates = _dates(df)
    rsi   = df["RSI"]

    fig = go.Figure()

    # Shaded zones
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,23,68,0.07)",
                  line_width=0, annotation_text="Overbought",
                  annotation_position="top left",
                  annotation_font=dict(size=9, color="#ff1744"))
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,200,83,0.07)",
                  line_width=0, annotation_text="Oversold",
                  annotation_position="bottom left",
                  annotation_font=dict(size=9, color="#00c853"))

    # Reference lines
    for level, color, dash in [(70, "#ff1744", "dash"),
                                 (50, "#3d4566",  "dot"),
                                 (30, "#00c853",  "dash")]:
        fig.add_hline(y=level, line_color=color, line_dash=dash, line_width=1)

    # RSI line — colour by zone
    rsi_vals = rsi.values
    ob = rsi_vals >= 70
    os_ = rsi_vals <= 30
    fig.add_trace(go.Scatter(
        x=dates, y=rsi, name="RSI (14)",
        line=dict(color="#ce93d8", width=2),
        fill="tozeroy", fillcolor="rgba(206,147,216,0.06)",
    ))

    fig.update_layout(
        **CHART_LAYOUT, height=240,
        title=dict(text=f"<b>{ticker}</b> — RSI (14)",
                   font=dict(color="#e8ecf4", size=13), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=44, b=10),
        yaxis=dict(**_Y_AXIS, range=[0, 100], title="RSI",
                   title_font=dict(size=10, color="#6b7494"),
                   tickvals=[0, 30, 50, 70, 100]),
        xaxis=_DATE_AXIS,
        legend=_LEGEND,
    )
    return fig


# ── Graph 4: MACD ─────────────────────────────────────────────────────────────

def make_macd_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """MACD line, Signal line, and Histogram — price-scaled for readability."""
    dates  = _dates(df)
    scale  = float(df["Close"].mean()) if df["Close"].mean() != 0 else 1.0
    macd   = df["MACD"] * scale
    signal = df["MACD_Signal"] * scale
    hist   = macd - signal
    hcol   = ["#00c853" if v >= 0 else "#ff1744" for v in hist]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=hist, name="Histogram",
        marker_color=hcol, opacity=0.70,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=macd, name="MACD",
        line=dict(color="#00d4aa", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=signal, name="Signal",
        line=dict(color="#ff9100", width=1.8),
    ))
    fig.add_hline(y=0, line_color="#3d4566", line_width=1)

    fig.update_layout(
        **CHART_LAYOUT, height=260,
        title=dict(text=f"<b>{ticker}</b> — MACD Analysis",
                   font=dict(color="#e8ecf4", size=13), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=44, b=10),
        yaxis=dict(**_Y_AXIS, title="MACD (price-scaled)",
                   title_font=dict(size=10, color="#6b7494")),
        xaxis=_DATE_AXIS,
        legend=_LEGEND,
        barmode="overlay",
    )
    return fig


def make_compound_forecast_chart(fc: dict) -> go.Figure:
    """
    Dynamic sequential compounding forecast chart.
    Accepts the dict returned by dynamic_compound_forecast().
    Each period on the path has its own growth rate — no flat extrapolation.
    """
    path   = fc["path"]
    upper  = fc["upper"]
    lower  = fc["lower"]
    ppy    = fc["ppy"]
    f1y    = fc["f1y"]
    f3y    = fc["f3y"]
    f5y    = fc["f5y"]
    total  = len(path) - 1
    years  = total // ppy

    x = list(range(len(path)))

    # Tick marks at year boundaries
    tick_vals = [y * ppy for y in range(years + 1)]
    tick_text = ["Now"] + [f"{y}Y" for y in range(1, years + 1)]

    # AI target x-positions (clamped to path length)
    targets = [(1, f1y), (3, f3y), (5, f5y)]
    tgt_x, tgt_y, tgt_lbl = [], [], []
    for yr, price in targets:
        idx = yr * ppy
        if idx <= total:
            tgt_x.append(idx)
            tgt_y.append(price)
            tgt_lbl.append(f"{price:,.0f}")

    fig = go.Figure()

    # Uncertainty band (filled area)
    fig.add_trace(go.Scatter(
        x=x, y=upper,
        line=dict(color="rgba(0,212,170,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lower,
        fill="tonexty", fillcolor="rgba(0,212,170,0.09)",
        line=dict(color="rgba(0,212,170,0)", width=0),
        name="P15–P85 range", hoverinfo="skip",
    ))

    # Dynamic path — colour segments by per-period rate (green=accelerating, amber=slowing)
    rates = fc.get("annual_rates", [])
    if rates:
        # Split path into acceleration / deceleration segments for colour coding
        mid_rate = float(np.median(rates)) if rates else 0.0
        for j in range(len(path) - 1):
            rate  = rates[j] if j < len(rates) else mid_rate
            color = "#00c853" if rate >= mid_rate else "#ffd740"
            fig.add_trace(go.Scatter(
                x=[j, j + 1], y=[path[j], path[j + 1]],
                mode="lines",
                line=dict(color=color, width=2.2),
                showlegend=False, hoverinfo="skip",
            ))
        # Invisible overlay for tooltip on the full path
        fig.add_trace(go.Scatter(
            x=x, y=path, mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=8),
            name="Dynamic forecast",
            hovertemplate="Period %{x}<br>Price: %{y:,.0f}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=path, name="Dynamic forecast",
            line=dict(color="#00d4aa", width=2.5),
        ))

    # AI target diamonds
    if tgt_x:
        fig.add_trace(go.Scatter(
            x=tgt_x, y=tgt_y, mode="markers+text",
            name="AI targets",
            marker=dict(color="#ffd740", size=11, symbol="diamond",
                        line=dict(color="#fff", width=1)),
            text=tgt_lbl,
            textposition="top center",
            textfont=dict(color="#ffd740", size=10),
        ))

    freq_label = {"yearly": "Yearly", "monthly": "Monthly", "weekly": "Weekly"}.get(
        fc.get("frequency", "monthly"), "Monthly"
    )
    fig.update_layout(
        **CHART_LAYOUT, height=340,
        title=dict(
            text=f"Dynamic Sequential Compounding Forecast ({freq_label})",
            font=dict(color="#e8ecf4", size=13), x=0, xanchor="left",
        ),
        yaxis_title="Price",
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, gridcolor="#1a1f30"),
        yaxis=dict(gridcolor="#1a1f30"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.14,
                    xanchor="left", x=0, font=dict(size=10)),
    )
    return fig


def make_comparison_chart(df1, t1, df2, t2) -> go.Figure:
    """Normalised price comparison — aligns by date where possible."""
    if "Date" in df1.columns and "Date" in df2.columns:
        d1 = df1.set_index("Date")["Close"]
        d2 = df2.set_index("Date")["Close"]
        aligned = pd.concat([d1, d2], axis=1, join="inner")
        aligned.columns = [t1, t2]
        n1 = aligned[t1] / aligned[t1].iloc[0] * 100
        n2 = aligned[t2] / aligned[t2].iloc[0] * 100
        x1, x2 = aligned.index, aligned.index
    else:
        n  = min(len(df1), len(df2))
        n1 = df1["Close"].tail(n) / float(df1["Close"].tail(n).iloc[0]) * 100
        n2 = df2["Close"].tail(n) / float(df2["Close"].tail(n).iloc[0]) * 100
        x1, x2 = list(range(len(n1))), list(range(len(n2)))

    fig = go.Figure([
        go.Scatter(x=x1, y=n1, name=t1, line=dict(color="#00d4aa", width=2)),
        go.Scatter(x=x2, y=n2, name=t2, line=dict(color="#ff9100", width=2)),
    ])
    fig.update_layout(
        **CHART_LAYOUT, height=300,
        title=dict(text="Normalised Price Comparison (base = 100)",
                   font=dict(color="#e8ecf4", size=13), x=0, xanchor="left"),
        yaxis=dict(**_Y_AXIS, title="Indexed Price",
                   title_font=dict(size=10, color="#6b7494")),
        xaxis=_DATE_AXIS,
        margin=dict(l=10, r=10, t=44, b=10),
        legend=_LEGEND,
    )
    return fig


# ── Graph 5: Volume Analysis ──────────────────────────────────────────────────

# ── Volume chart helpers (pre-compute once, pass dfv) ─────────────────────────

def _base_layout(title_text: str, height: int = 280) -> dict:
    return dict(
        **CHART_LAYOUT, height=height,
        title=dict(text=title_text, font=dict(color="#e8ecf4", size=13),
                   x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=_LEGEND,
        showlegend=False,
        xaxis=dict(**_DATE_AXIS),
        yaxis=dict(**_Y_AXIS),
    )


# ── Volume Chart 1: OBV ───────────────────────────────────────────────────────

def make_obv_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """On Balance Volume — standalone independent chart."""
    dfv   = compute_advanced_volume(df)
    dates = _dates(dfv)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=dfv["OBV"], name="OBV",
        line=dict(color="#00d4aa", width=1.8),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
        hovertemplate="%{x}<br>OBV: %{y:,.0f}<extra></extra>",
    ))
    # VWAP overlay
    fig.add_trace(go.Scatter(
        x=dates, y=dfv["VWAP"], name="VWAP (20d)",
        line=dict(color="#ffd740", width=1.2, dash="dot"),
        hovertemplate="%{x}<br>VWAP: %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(f"<b>{ticker}</b> — On Balance Volume (OBV) + VWAP", height=280))
    fig.update_layout(showlegend=True)
    return fig


# ── Volume Chart 2: Relative Volume ──────────────────────────────────────────

def make_relvol_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Relative Volume vs 20-day average — standalone chart."""
    dfv   = compute_advanced_volume(df)
    dates = _dates(dfv)
    rv    = dfv["RelVol"].fillna(1.0)
    colors = ["#00c853" if v >= 1.5 else ("#ffd740" if v >= 1.0 else "#3d4566") for v in rv]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=rv, name="Relative Volume",
        marker_color=colors, opacity=0.85,
        hovertemplate="%{x}<br>Rel. Vol: %{y:.2f}×<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_color="#6b7494", line_dash="dot", line_width=1,
                  annotation_text="Avg baseline",
                  annotation_font=dict(size=9, color="#6b7494"))
    fig.add_hline(y=2.0, line_color="#ff9100", line_dash="dash", line_width=1,
                  annotation_text="2× spike",
                  annotation_font=dict(size=9, color="#ff9100"))
    fig.update_layout(**_base_layout(f"<b>{ticker}</b> — Relative Volume (vs 20d Average)", height=260))
    fig.update_yaxes(title_text="Volume Ratio", title_font=dict(size=10, color="#6b7494"))
    return fig


# ── Volume Chart 3: Volume Spike (z-score) ────────────────────────────────────

def make_volspike_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Volume Spike Detection using z-score — standalone chart."""
    dfv    = compute_advanced_volume(df)
    dates  = _dates(dfv)
    vs     = dfv["VolSpike"].fillna(0.0)
    colors = ["#00c853" if v >= 2 else "#ffd740" if v >= 0 else "#ff1744" for v in vs]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=vs, name="Vol Spike σ",
        marker_color=colors, opacity=0.80,
        hovertemplate="%{x}<br>Z-Score: %{y:.2f}σ<extra></extra>",
    ))
    fig.add_hline(y=0,   line_color="#3d4566", line_width=1)
    fig.add_hline(y=2.0, line_color="#ffd740", line_dash="dash", line_width=1,
                  annotation_text="+2σ",
                  annotation_font=dict(size=9, color="#ffd740"))
    fig.add_hline(y=3.0, line_color="#ff1744", line_dash="dash", line_width=1,
                  annotation_text="+3σ alert",
                  annotation_font=dict(size=9, color="#ff1744"))
    fig.update_layout(**_base_layout(f"<b>{ticker}</b> — Volume Spike Detection (Z-Score)", height=260))
    fig.update_yaxes(title_text="Std Deviations (σ)", title_font=dict(size=10, color="#6b7494"))
    return fig


# ── Anomaly Chart 1: Price with event markers ─────────────────────────────────

def make_price_anomaly_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Price chart with anomaly events highlighted — standalone independent chart."""
    dfa        = detect_anomalies(df)
    dates      = _dates(dfa)
    anom_idx   = dfa[dfa["Anomaly"] == True]
    anom_dates = _dates(anom_idx)
    n_anoms    = int(dfa["Anomaly"].sum())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=dfa["Close"], name="Close",
        line=dict(color="rgba(200,210,235,0.45)", width=1.5),
        hovertemplate="%{x}<br>Close: %{y:,.2f}<extra></extra>",
    ))
    if len(anom_idx) > 0:
        fig.add_trace(go.Scatter(
            x=anom_dates, y=anom_idx["Close"],
            mode="markers", name=f"Anomaly ({n_anoms})",
            marker=dict(color="#ff1744", size=9, symbol="circle",
                        line=dict(color="#ffffff", width=1.2)),
            hovertemplate="%{x}<br>Anomaly: %{y:,.2f}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(
        f"<b>{ticker}</b> — Price Anomaly Events  ({n_anoms} detected)", height=300,
    ))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text="Price", title_font=dict(size=10, color="#6b7494"))
    return fig


# ── Anomaly Chart 2: Anomaly Score ───────────────────────────────────────────

def make_anomaly_score_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Composite anomaly score (z-score) — standalone independent chart."""
    dfa   = detect_anomalies(df)
    dates = _dates(dfa)
    score = dfa["AnomalyScore"].fillna(0)
    colors = ["#ff1744" if v > 3 else "#ffd740" if v > 2 else "#3d4566" for v in score]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=score, name="Anomaly Score",
        marker_color=colors, opacity=0.80,
        hovertemplate="%{x}<br>Score: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=2.0, line_color="#ffd740", line_dash="dot", line_width=1,
                  annotation_text="Warning",
                  annotation_font=dict(size=9, color="#ffd740"))
    fig.add_hline(y=3.0, line_color="#ff1744", line_dash="dash", line_width=1,
                  annotation_text="Alert threshold",
                  annotation_font=dict(size=9, color="#ff1744"))
    fig.update_layout(**_base_layout(f"<b>{ticker}</b> — Anomaly Strength Score (Composite Z-Score)", height=260))
    fig.update_yaxes(title_text="Anomaly Score", title_font=dict(size=10, color="#6b7494"))
    return fig


# ── Comparison: Normalised dual-stock performance ─────────────────────────────

def make_normalised_compare_chart(
    df1: pd.DataFrame, t1: str,
    df2: pd.DataFrame, t2: str,
    lookback_days: int = 504,
) -> go.Figure:
    """Both stocks rebased to 100 on the same calendar start date.

    Uses a calendar-date cutoff (today minus N years) instead of row-count
    tail(), so both series always cover the same date window regardless of
    differences in data density or yfinance coverage per ticker.
    """
    # Convert trading-day count → approximate calendar days
    calendar_days = int(lookback_days / 252 * 365.25)
    cutoff = (pd.Timestamp.now() - pd.Timedelta(days=calendar_days)).normalize()

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Ensure a tz-naive Date column exists
        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)
        elif hasattr(d.index, "date") or str(d.index.dtype).startswith("datetime"):
            d["Date"] = pd.to_datetime(d.index).dt.tz_localize(None)
            d = d.reset_index(drop=True)
        # Filter to the selected calendar window
        if "Date" in d.columns:
            d = d[d["Date"] >= cutoff]
        return d.reset_index(drop=True)

    d1 = _prep(df1)
    d2 = _prep(df2)

    # If BOTH are empty there is truly nothing to show
    if d1.empty and d2.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=380,
                          title=dict(text="No data available for selected period",
                                     font=dict(color="#e8ecf4", size=13)))
        return fig

    # Build whichever series we have data for
    fig = go.Figure()
    coverage_note = ""

    if not d1.empty:
        base1 = float(d1["Close"].iloc[0])
        norm1 = d1["Close"] / base1 * 100.0
        fig.add_trace(go.Scatter(
            x=_dates(d1), y=norm1, name=t1,
            line=dict(color="#00d4aa", width=2.0),
            hovertemplate="%{x}<br>" + t1 + ": %{y:.1f}<extra></extra>",
        ))
    else:
        coverage_note = f"{t1} has no data for this period (Yahoo Finance)"

    if not d2.empty:
        base2 = float(d2["Close"].iloc[0])
        norm2 = d2["Close"] / base2 * 100.0
        fig.add_trace(go.Scatter(
            x=_dates(d2), y=norm2, name=t2,
            line=dict(color="#40c4ff", width=2.0, dash="dash"),
            hovertemplate="%{x}<br>" + t2 + ": %{y:.1f}<extra></extra>",
        ))
    else:
        coverage_note = f"{t2} has no data for this period on Yahoo Finance — showing {t1} only"

    fig.add_hline(y=100, line_color="#3d4566", line_dash="dot", line_width=1,
                  annotation_text="Baseline 100",
                  annotation_font=dict(size=9, color="#6b7494"))

    # Build title / subtitle
    if not d1.empty and not d2.empty:
        final1 = float(norm1.iloc[-1])
        final2 = float(norm2.iloc[-1])
        winner = t1 if final1 >= final2 else t2
        delta  = abs(final1 - final2)
        subtitle = f"{winner} outperformed by {delta:.1f} pts over period"
        # Warn if second stock's data ends much earlier than first
        if "Date" in d1.columns and "Date" in d2.columns:
            gap_days = (d1["Date"].iloc[-1] - d2["Date"].iloc[-1]).days
            if gap_days > 30:
                last2 = d2["Date"].iloc[-1].strftime("%b %Y")
                subtitle += f"  ·  ⚠ {t2} data ends {last2} (Yahoo Finance)"
    else:
        subtitle = coverage_note

    title_text = (
        f"<b>Normalised Performance</b>  —  {t1} vs {t2}<br>"
        f"<span style='font-size:11px;color:#6b7494'>{subtitle}</span>"
    )

    fig.update_layout(
        **CHART_LAYOUT, height=380,
        title=dict(text=title_text, font=dict(color="#e8ecf4", size=13),
                   x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=56, b=10),
        legend=_LEGEND,
        xaxis=dict(**_DATE_AXIS),
        yaxis=dict(**_Y_AXIS, title="Indexed (base = 100)",
                   title_font=dict(size=10, color="#6b7494")),
    )
    return fig


# ── Backtest: Equity Curve ────────────────────────────────────────────────────

def make_equity_chart(bt: dict, ticker: str) -> go.Figure:
    """Strategy equity curve vs buy-and-hold benchmark."""
    dates  = bt.get("equity_dates", [])
    eq     = bt.get("equity_vals",  [])
    bh     = bt.get("bh_vals",      [])
    init   = bt.get("initial", 100_000)

    fig = go.Figure()

    # Buy-and-hold baseline
    if bh:
        fig.add_trace(go.Scatter(
            x=dates, y=bh, name="Buy & Hold",
            line=dict(color="#40c4ff", width=1.8, dash="dot"),
        ))

    # Strategy equity
    fig.add_trace(go.Scatter(
        x=dates, y=eq, name="AI Strategy",
        line=dict(color="#00d4aa", width=2.2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
    ))

    # Initial capital reference
    fig.add_hline(y=init, line_color="#3d4566", line_dash="dash",
                  line_width=1, annotation_text="Initial capital",
                  annotation_font=dict(size=9, color="#6b7494"))

    fig.update_layout(
        **CHART_LAYOUT, height=360,
        title=dict(text=f"<b>{ticker}</b> — Backtest Equity Curve",
                   font=dict(color="#e8ecf4", size=13), x=0, xanchor="left"),
        yaxis=dict(**_Y_AXIS, title="Portfolio Value",
                   title_font=dict(size=10, color="#6b7494")),
        xaxis=dict(gridcolor="#1a1f30", tickfont=dict(size=10, color="#6b7494")),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=_LEGEND,
    )
    return fig


# ── Risk: Rolling Drawdown ────────────────────────────────────────────────────

def make_drawdown_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Rolling peak-to-trough drawdown over the full history."""
    closes   = df["Close"].values.astype(np.float64)
    peak     = np.maximum.accumulate(closes)
    drawdown = (closes - peak) / np.where(peak > 0, peak, 1e-9) * 100.0
    dates    = _dates(df)

    # Rolling 1-year max drawdown
    dd_series = pd.Series(drawdown)
    roll_max_dd = dd_series.rolling(252, min_periods=1).min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=drawdown,
        name="Drawdown", fill="tozeroy",
        fillcolor="rgba(255,23,68,0.12)",
        line=dict(color="#ff1744", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=roll_max_dd,
        name="1Y Rolling Max DD",
        line=dict(color="#ff9100", width=1.5, dash="dot"),
    ))
    fig.add_hline(y=-20, line_color="#ffd740", line_dash="dash", line_width=1,
                  annotation_text="-20%", annotation_font=dict(size=9, color="#ffd740"))
    fig.add_hline(y=-40, line_color="#ff1744", line_dash="dash", line_width=1,
                  annotation_text="-40%", annotation_font=dict(size=9, color="#ff1744"))

    fig.update_layout(
        **CHART_LAYOUT, height=280,
        title=dict(text=f"<b>{ticker}</b> — Rolling Drawdown",
                   font=dict(color="#e8ecf4", size=13), x=0, xanchor="left"),
        yaxis=dict(**_Y_AXIS, title="Drawdown %",
                   title_font=dict(size=10, color="#6b7494")),
        xaxis=_DATE_AXIS,
        margin=dict(l=10, r=10, t=44, b=10),
        legend=_LEGEND,
    )
    return fig


# ── Feature Importance ────────────────────────────────────────────────────────

def make_feature_importance_chart(xgb_model) -> go.Figure | None:
    """Horizontal bar chart of XGBoost feature importances."""
    if xgb_model is None:
        return None
    try:
        importances = xgb_model.feature_importances_
        pairs = sorted(zip(FEATURES, importances), key=lambda x: x[1])
        names = [p[0] for p in pairs]
        vals  = [p[1] for p in pairs]
        colors = ["#00d4aa" if v >= np.percentile(vals, 75) else
                  "#40c4ff" if v >= np.percentile(vals, 50) else "#3d4566"
                  for v in vals]

        fig = go.Figure(go.Bar(
            x=vals, y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=9, color="#9ba3bf"),
        ))
        fig.update_layout(
            **CHART_LAYOUT, height=540,
            title=dict(text="XGBoost Feature Importance (Gain)",
                       font=dict(color="#e8ecf4", size=13), x=0, xanchor="left"),
            xaxis=dict(**_Y_AXIS, title="Importance Score",
                       title_font=dict(size=10, color="#6b7494")),
            yaxis=dict(gridcolor="#1a1f30", tickfont=dict(size=10, color="#9ba3bf")),
            margin=dict(l=130, r=60, t=44, b=30),
            showlegend=False,
        )
        return fig
    except Exception:
        return None


def clamp_for_chart(rate: float) -> float:
    return max(-0.30, min(0.40, rate))


# ── FinBERT news sentiment ────────────────────────────────────────────────────

# Keyword-based fallback — runs in microseconds, no model download needed
_POS_WORDS = {
    "growth", "profit", "gain", "surge", "rally", "beat", "record", "strong",
    "bullish", "rise", "up", "high", "buy", "outperform", "upgrade", "revenue",
    "earnings", "dividend", "expand", "acquisition", "recovery", "positive",
    "increase", "boost", "momentum", "optimism", "robust", "exceeds",
}
_NEG_WORDS = {
    "loss", "fall", "drop", "crash", "sell", "bearish", "down", "low", "weak",
    "miss", "decline", "risk", "concern", "warning", "downgrade", "cut", "debt",
    "inflation", "recession", "layoff", "negative", "slump", "disappoints",
    "worries", "pressure", "deficit", "defaults", "shortage", "fear",
}

def _keyword_sentiment(headline: str) -> tuple[str, float]:
    """Instant rule-based financial sentiment — no model required."""
    words = set(headline.lower().split())
    pos   = len(words & _POS_WORDS)
    neg   = len(words & _NEG_WORDS)
    if pos > neg:
        return "POSITIVE", 0.60 + min(pos * 0.06, 0.35)
    elif neg > pos:
        return "NEGATIVE", 0.60 + min(neg * 0.06, 0.35)
    return "NEUTRAL", 0.55


@st.cache_resource(show_spinner=False)
def _load_nlp_pipeline():
    """
    Load the NLP pipeline ONCE per session and keep it in memory.
    Uses @st.cache_resource so it survives across Streamlit reruns.
    Returns None if the model cannot be loaded within the timeout.
    """
    import concurrent.futures
    def _build():
        try:
            from transformers import pipeline  # type: ignore
            return pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                truncation=True, max_length=128,
            )
        except Exception:
            try:
                from transformers import pipeline  # type: ignore
                return pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True, max_length=128,
                )
            except Exception:
                return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_build)
        try:
            return fut.result(timeout=20)   # give the model 20 s to load
        except concurrent.futures.TimeoutError:
            return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_headlines(ticker: str) -> list[str]:
    """Fetch and deduplicate news headlines — fast, no ML."""
    try:
        query = urllib.parse.quote(f"{ticker} stock market Pakistan finance")
        url   = (f"https://news.google.com/rss/search"
                 f"?q={query}&hl=en-US&gl=US&ceid=US:en")
        feed  = feedparser.parse(url)
        return list(dict.fromkeys(
            e.title for e in feed.entries[:15] if hasattr(e, "title")
        ))[:10]
    except Exception:
        return []


def run_sentiment(headlines: list[str]) -> list[dict]:
    """
    Score headlines using NLP pipeline with 10-second per-batch timeout.
    Falls back to keyword scoring instantly if the model is unavailable.
    """
    import concurrent.futures

    results = []
    nlp = _load_nlp_pipeline()

    if nlp is None:
        # Model not ready → keyword fallback (instant)
        for h in headlines:
            label, score = _keyword_sentiment(h)
            results.append({"headline": h, "label": label,
                            "score": score, "method": "keyword"})
        return results

    def _infer(texts):
        raw = nlp(texts)
        out = []
        for h, r in zip(texts, raw):
            label = r["label"].upper()
            if label == "LABEL_1" or label == "POSITIVE":
                label = "POSITIVE"
            elif label == "LABEL_0" or label == "NEGATIVE":
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            out.append({"headline": h, "label": label,
                        "score": r["score"], "method": "finbert"})
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_infer, headlines)
        try:
            results = fut.result(timeout=10)
        except (concurrent.futures.TimeoutError, Exception):
            # Inference timed out → fall back to keywords
            for h in headlines:
                label, score = _keyword_sentiment(h)
                results.append({"headline": h, "label": label,
                                "score": score, "method": "keyword"})
    return results


# ── Helper: metric card HTML ──────────────────────────────────────────────────

def mc(label: str, value: str, sub: str = "", color: str = "#6b7494") -> str:
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub" style="color:{color}">{sub}</div>'
            f'</div>')


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<h3 style='margin-top:0.2rem;margin-bottom:0'>📈 PSX AI Dashboard</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#6b7494;font-size:.78rem;margin-top:-4px'>"
        "GRU · XGBoost · FinBERT · Log Returns</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    valid_labels   = [l for l, v in STOCK_LIST.items() if v is not None]
    default_idx    = valid_labels.index("FFC — Fauji Fertilizer") if "FFC — Fauji Fertilizer" in valid_labels else 0
    selected_label = st.selectbox("Select Stock", valid_labels, index=default_idx)
    ticker         = STOCK_LIST[selected_label]

    st.markdown("---")
    investment = st.number_input(
        "Portfolio Investment (PKR / USD)",
        min_value=1_000, max_value=100_000_000,
        value=100_000, step=10_000, format="%d",
    )

    st.markdown("---")
    gru_ok = os.path.exists(GRU_PATH)
    xgb_ok = os.path.exists(XGB_PATH)
    scl_ok = os.path.exists(SCALER_PATH)

    if gru_ok and scl_ok:
        status_cls = "status-ok"
        status_txt = "Ensemble ready" if xgb_ok else "GRU model ready"
        status_sub = "GRU + XGBoost" if xgb_ok else "GRU only"
        status_ico = "✅"
    else:
        status_cls = "status-warn"
        status_txt = "No trained model"
        status_sub = "Run: python train_model.py"
        status_ico = "⚠️"

    st.markdown(
        f'<div class="{status_cls}">{status_ico} <b>{status_txt}</b><br>'
        f'<span style="font-size:.74rem;opacity:.8">{status_sub}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        "<p style='color:#2e3450;font-size:.7rem;text-align:center'>"
        "yFinance · TensorFlow · XGBoost<br>FinBERT · Log Returns · Not financial advice</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

stock_name = selected_label.split("—")[1].strip() if "—" in selected_label else selected_label
stock_code = selected_label.split("—")[0].strip()

col_title, col_date = st.columns([3, 1])
with col_title:
    st.markdown(
        f'<div style="font-size:1.85rem;font-weight:800;letter-spacing:-.3px">{stock_name}</div>'
        f'<div style="font-size:.84rem;color:#6b7494;margin-top:2px">'
        f'{stock_code} &nbsp;·&nbsp; {ticker} &nbsp;·&nbsp; '
        f'AI Quant Platform &nbsp;·&nbsp; GRU · XGBoost · FinBERT</div>',
        unsafe_allow_html=True,
    )
with col_date:
    st.markdown(
        f'<div style="text-align:right;color:#6b7494;font-size:.82rem;padding-top:12px">'
        f'{datetime.now().strftime("%B %d, %Y")}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

with st.spinner(f"Fetching {ticker} data…"):
    df = fetch_data(ticker)

if df is None or df.empty:
    st.error(f"Could not load data for **{ticker}**. Try another stock.")
    st.stop()

gru_model, xgb_model, scaler = load_models()
models_ready = (gru_model is not None or xgb_model is not None) and scaler is not None

# ── Core computed metrics (shared across all tabs) ────────────────────────────
current_price = float(df["Close"].iloc[-1])
prev_price    = float(df["Close"].iloc[-2])
price_change  = current_price - prev_price
price_pct     = price_change / prev_price * 100
rsi_val       = float(df["RSI"].iloc[-1])
ema20v        = float(df["EMA20"].iloc[-1])
ema50v        = float(df["EMA50"].iloc[-1])
macd_v        = float(df["MACD"].iloc[-1])
mom_v         = float(df["Momentum"].iloc[-1]) * 100
bb_v          = float(df["BB_Width"].iloc[-1]) * 100
atr_v         = float(df["ATR"].iloc[-1]) * 100

# Financial analytics
historical_cagr = compute_cagr(df)
max_dd          = compute_max_drawdown(df)
cum_ret         = compute_cumulative_return(df)
ann_vol         = compute_annual_volatility(df)
risk_cat        = "Low" if ann_vol < 20 else ("Medium" if ann_vol < 40 else "High")

if models_ready:
    ens        = ensemble_predict(df, gru_model, xgb_model, scaler)
    confidence = ens["confidence"]
    gru_conf   = ens["dl"]
    xgb_conf   = ens["xgb"]
    signal     = compute_signal(confidence, rsi_val, df)
    bt_acc     = backtest_accuracy(df, gru_model, xgb_model, scaler)
else:
    confidence = gru_conf = xgb_conf = 0.5
    signal     = "HOLD"
    bt_acc     = 0.5

# ── Shared derived values ─────────────────────────────────────────────────────
arrow         = "▲" if price_change >= 0 else "▼"
price_color   = "#00c853" if price_change >= 0 else "#ff1744"
ma_spread     = (ema20v - ema50v) / ema50v if ema50v != 0 else 0
trend_strength = float(np.clip(ma_spread * 5 + float(df["LogReturn"].mean()) * 50, -1, 1))
market_regime = float(np.clip(
    df["CumLogRet30"].iloc[-1] if "CumLogRet30" in df.columns else 0.0,
    -0.15, 0.15,
))
_quick_headlines = fetch_headlines(ticker)
if _quick_headlines:
    _kw_results  = [_keyword_sentiment(h) for h in _quick_headlines]
    _pos_count   = sum(1 for lbl, _ in _kw_results if lbl == "POSITIVE")
    sentiment_score = 0.5 + (_pos_count / len(_kw_results) - 0.5) * 0.6
else:
    sentiment_score = 0.5

yrs = len(df) / 252


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_ov, tab_ch, tab_cmp, tab_sc, tab_bt, tab_fc, tab_nw = st.tabs([
    "📊 Overview",
    "📈 Charts",
    "⚖️ Compare",
    "🔍 Screener",
    "⚡ Backtest",
    "🔮 Forecast",
    "📰 News",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════

with tab_ov:
    # Row 1: Price + AI KPIs
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    with r1c1:
        st.markdown(mc(
            "Current Price", f"{current_price:,.2f}",
            f"{arrow} {abs(price_pct):.2f}% today", price_color,
        ), unsafe_allow_html=True)
    with r1c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">AI Signal (5-day)</div>'
            f'<div style="margin-top:10px">{signal_badge(signal)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with r1c3:
        cc = "#00c853" if confidence >= 0.55 else ("#ff1744" if confidence <= 0.45 else "#ffd740")
        st.markdown(mc(
            "Ensemble Confidence", f"{confidence*100:.1f}%",
            "GRU + XGBoost weighted", cc,
        ), unsafe_allow_html=True)
    with r1c4:
        rc = "#00c853" if risk_cat == "Low" else ("#ffd740" if risk_cat == "Medium" else "#ff1744")
        st.markdown(mc(
            "Risk Level", f"{risk_cat} Risk",
            f"Ann. vol {ann_vol:.1f}%", rc,
        ), unsafe_allow_html=True)
    with r1c5:
        bc = "#00c853" if bt_acc >= 0.56 else ("#ffd740" if bt_acc >= 0.50 else "#ff1744")
        st.markdown(mc(
            "Directional Accuracy", f"{bt_acc*100:.1f}%",
            "Last 200 trading days", bc,
        ), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Row 2: Financial performance
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        cagr_c = "#00c853" if historical_cagr >= 0.05 else ("#ffd740" if historical_cagr >= 0 else "#ff1744")
        st.markdown(mc(
            "CAGR (since 2015)", f"{historical_cagr*100:.1f}%",
            "Compound annual growth rate", cagr_c,
        ), unsafe_allow_html=True)
    with r2c2:
        dd_c = "#00c853" if max_dd >= -0.20 else ("#ffd740" if max_dd >= -0.40 else "#ff1744")
        st.markdown(mc(
            "Max Drawdown", f"{max_dd*100:.1f}%",
            "Peak-to-trough worst decline", dd_c,
        ), unsafe_allow_html=True)
    with r2c3:
        cr_c = "#00c853" if cum_ret >= 0 else "#ff1744"
        st.markdown(mc(
            "Cumulative Return", f"{cum_ret*100:+.0f}%",
            f"Total return over {yrs:.1f} years", cr_c,
        ), unsafe_allow_html=True)
    with r2c4:
        rc2  = "#ff1744" if rsi_val >= 70 else ("#00c853" if rsi_val <= 30 else "#ffd740")
        rtxt = "Overbought" if rsi_val >= 70 else ("Oversold" if rsi_val <= 30 else "Neutral")
        st.markdown(mc("RSI (14)", f"{rsi_val:.1f}", rtxt, rc2), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Row 3: Technical indicators
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1:
        tc = "#00c853" if ema20v > ema50v else "#ff1744"
        st.markdown(mc("MA Trend",
                       "Bullish" if ema20v > ema50v else "Bearish",
                       "EMA20 vs EMA50", tc), unsafe_allow_html=True)
    with r3c2:
        mc2 = "#00c853" if mom_v >= 0 else "#ff1744"
        st.markdown(mc("5-Day Momentum", f"{mom_v:+.2f}%",
                       "Price change past 5 days", mc2), unsafe_allow_html=True)
    with r3c3:
        gc = "#00c853" if gru_conf >= 0.55 else ("#ff1744" if gru_conf <= 0.45 else "#ffd740")
        st.markdown(mc("GRU Confidence", f"{gru_conf*100:.1f}%",
                       "Deep learning · 60-day seq", gc), unsafe_allow_html=True)
    with r3c4:
        xc = "#00c853" if xgb_conf >= 0.55 else ("#ff1744" if xgb_conf <= 0.45 else "#ffd740")
        st.markdown(mc("XGBoost Confidence", f"{xgb_conf*100:.1f}%",
                       "Gradient boosting", xc), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Row 4: Extra analytics
    r4c1, r4c2, r4c3, r4c4 = st.columns(4)
    with r4c1:
        bb_c = "#ff1744" if bb_v > 20 else ("#00c853" if bb_v < 8 else "#ffd740")
        st.markdown(mc("BB Width", f"{bb_v:.1f}%",
                       "Bollinger Band spread", bb_c), unsafe_allow_html=True)
    with r4c2:
        at_c = "#ff1744" if atr_v > 3 else ("#00c853" if atr_v < 1 else "#ffd740")
        st.markdown(mc("ATR (norm)", f"{atr_v:.2f}%",
                       "Daily range / price", at_c), unsafe_allow_html=True)
    with r4c3:
        reg_label = ("Bull Market" if market_regime > 0.02
                     else "Bear Market" if market_regime < -0.02 else "Neutral")
        reg_clr   = ("#00c853" if market_regime > 0.02
                     else "#ff1744" if market_regime < -0.02 else "#ffd740")
        st.markdown(mc("Market Regime", reg_label,
                       f"30d log-ret: {market_regime*100:+.1f}%", reg_clr),
                    unsafe_allow_html=True)
    with r4c4:
        sent_label = ("Bullish" if sentiment_score > 0.55
                      else "Bearish" if sentiment_score < 0.45 else "Neutral")
        sc = ("#00c853" if sentiment_score > 0.55
              else "#ff1744" if sentiment_score < 0.45 else "#ffd740")
        st.markdown(mc("News Sentiment", sent_label,
                       f"Score: {sentiment_score:.2f}", sc), unsafe_allow_html=True)

    # AI Signal breakdown
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 AI Signal Analysis</div>', unsafe_allow_html=True)
    sa1, sa2, sa3 = st.columns(3)
    with sa1:
        verdict_txt = (
            f"Bullish edge: {(confidence-0.5)*200:.0f}% above baseline"
            if confidence > 0.5 else
            f"Bearish edge: {(0.5-confidence)*200:.0f}% below baseline"
        )
        st.markdown(mc("Ensemble Verdict", signal,
                       verdict_txt, price_color), unsafe_allow_html=True)
    with sa2:
        rsi_desc = ("Deeply oversold — historically strong reversal zone" if rsi_val < 25
                    else "Oversold — BUY pressure building" if rsi_val < 35
                    else "Overbought — SELL pressure rising" if rsi_val > 65
                    else "Extremely overbought — reversal risk" if rsi_val > 75
                    else "Neutral momentum territory")
        rsi_c2 = "#ff1744" if rsi_val > 65 else "#00c853" if rsi_val < 35 else "#ffd740"
        st.markdown(mc("RSI Analysis", f"RSI {rsi_val:.0f}", rsi_desc, rsi_c2),
                    unsafe_allow_html=True)
    with sa3:
        ma_desc = (f"EMA20 {((ema20v/ema50v)-1)*100:+.1f}% vs EMA50 — "
                   + ("Golden cross zone" if ema20v > ema50v else "Death cross zone"))
        ma_c2 = "#00c853" if ema20v > ema50v else "#ff1744"
        st.markdown(mc("Trend Structure", "Bullish" if ema20v > ema50v else "Bearish",
                       ma_desc, ma_c2), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — CHARTS
# ══════════════════════════════════════════════════════════

with tab_ch:
    lookback_opts = {
        "6 Months": 126, "1 Year": 252, "2 Years": 504,
        "5 Years": 1260, "All": len(df),
    }
    lookback_label = st.select_slider(
        "Chart lookback period",
        options=list(lookback_opts.keys()),
        value="2 Years",
    )
    lookback_days = lookback_opts[lookback_label]
    display_df    = df.tail(lookback_days).copy().reset_index(drop=True)

    # ── Price & Trend ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Price Action</div>', unsafe_allow_html=True)
    st.plotly_chart(make_price_chart(display_df, ticker), use_container_width=True)

    st.markdown('<div class="section-header">📉 Moving Averages & Trend</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_ma_chart(display_df, ticker), use_container_width=True)

    # ── Momentum ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ RSI (14) — Momentum Oscillator</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_rsi_chart(display_df, ticker), use_container_width=True)

    st.markdown('<div class="section-header">〰️ MACD — Trend Strength & Crossover</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_macd_chart(display_df, ticker), use_container_width=True)

    # ── Volume Analysis — 3 independent charts ────────────────────────────────
    st.markdown('<div class="section-header">📦 On Balance Volume (OBV) + VWAP</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_obv_chart(display_df, ticker), use_container_width=True)

    st.markdown('<div class="section-header">📊 Relative Volume — vs 20d Average</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_relvol_chart(display_df, ticker), use_container_width=True)

    st.markdown('<div class="section-header">🔺 Volume Spike Detection — Z-Score</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_volspike_chart(display_df, ticker), use_container_width=True)

    # ── Anomaly Detection — 2 independent charts ──────────────────────────────
    st.markdown('<div class="section-header">🚨 Price Anomaly Events</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_price_anomaly_chart(display_df, ticker), use_container_width=True)

    st.markdown('<div class="section-header">📉 Anomaly Strength Score — Composite Z-Score</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_anomaly_score_chart(display_df, ticker), use_container_width=True)


# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# TAB 3 — COMPARE STOCKS
# ══════════════════════════════════════════════════════════

with tab_cmp:
    st.markdown('<div class="section-header">⚖️ Compare Stocks</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6b7494;font-size:.82rem'>"
        "Side-by-side comparative analytics: metrics, normalised performance, and forecasts.</p>",
        unsafe_allow_html=True,
    )

    cmp_opts   = [l for l, v in STOCK_LIST.items() if v and v != ticker]
    cmp_label  = st.selectbox("Compare with", cmp_opts, key="cmp_select_tab")
    cmp_ticker_tab = STOCK_LIST.get(cmp_label, "")

    if not cmp_ticker_tab:
        st.info("Select a comparison stock above.")
    else:
        with st.spinner(f"Loading {cmp_ticker_tab}…"):
            df_cmp = fetch_data(cmp_ticker_tab)

        if df_cmp is None or df_cmp.empty:
            st.warning(f"Could not load data for {cmp_ticker_tab}.")
        else:
            cmp_name = cmp_label.split("—")[1].strip() if "—" in cmp_label else cmp_label

            # ── Compute comparison metrics ──────────────────────────────────
            cmp_price   = float(df_cmp["Close"].iloc[-1])
            cmp_prev    = float(df_cmp["Close"].iloc[-2])
            cmp_chg_pct = (cmp_price - cmp_prev) / cmp_prev * 100
            cmp_rsi     = float(df_cmp["RSI"].iloc[-1]) if "RSI" in df_cmp.columns else 50.0
            cmp_ema20   = float(df_cmp["EMA20"].iloc[-1]) if "EMA20" in df_cmp.columns else 0.0
            cmp_ema50   = float(df_cmp["EMA50"].iloc[-1]) if "EMA50" in df_cmp.columns else 0.0
            cmp_sma50   = (df_cmp["Close"].rolling(50).mean().iloc[-1]
                           if len(df_cmp) >= 50 else cmp_price)
            cmp_sma200  = (df_cmp["Close"].rolling(200).mean().iloc[-1]
                           if len(df_cmp) >= 200 else cmp_price)
            my_sma50    = (df["Close"].rolling(50).mean().iloc[-1]
                           if len(df) >= 50 else current_price)
            my_sma200   = (df["Close"].rolling(200).mean().iloc[-1]
                           if len(df) >= 200 else current_price)
            cmp_cagr    = compute_cagr(df_cmp)
            cmp_maxdd   = compute_max_drawdown(df_cmp)
            cmp_cumret  = compute_cumulative_return(df_cmp)
            cmp_vol     = compute_annual_volatility(df_cmp)
            cmp_risk    = "Low" if cmp_vol < 20 else ("Medium" if cmp_vol < 40 else "High")

            if models_ready:
                cmp_ens  = ensemble_predict(df_cmp, gru_model, xgb_model, scaler)
                cmp_conf = cmp_ens["confidence"]
                cmp_sig  = compute_signal(cmp_conf, cmp_rsi, df_cmp)
            else:
                cmp_conf = 0.5
                cmp_sig  = "HOLD"

            # ── Section heading ─────────────────────────────────────────────
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            cmp_h1, cmp_h2 = st.columns(2)
            with cmp_h1:
                st.markdown(
                    f"<div style='font-size:1.1rem;font-weight:700;color:#e8ecf4'>{stock_name}</div>"
                    f"<div style='font-size:.78rem;color:#6b7494'>{ticker}</div>",
                    unsafe_allow_html=True,
                )
            with cmp_h2:
                st.markdown(
                    f"<div style='font-size:1.1rem;font-weight:700;color:#40c4ff'>{cmp_name}</div>"
                    f"<div style='font-size:.78rem;color:#6b7494'>{cmp_ticker_tab}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── Side-by-side metrics grid ───────────────────────────────────
            def _cmp_pair_row(label: str, v1: str, v2: str,
                               c1: str = "#e8ecf4", c2: str = "#e8ecf4") -> None:
                st.markdown(
                    f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                    f"gap:4px;padding:7px 14px;background:#12172a;"
                    f"border-bottom:1px solid #1e2235;font-size:.80rem;align-items:center'>"
                    f"<div style='color:#6b7494;font-size:.72rem;font-weight:600;"
                    f"text-transform:uppercase;letter-spacing:.6px'>{label}</div>"
                    f"<div style='color:{c1};font-weight:600'>{v1}</div>"
                    f"<div style='color:{c2};font-weight:600'>{v2}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Table header
            st.markdown(
                "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                "gap:4px;padding:8px 14px;background:#0e1220;border-radius:8px 8px 0 0;"
                "font-size:.68rem;color:#6b7494;font-weight:700;text-transform:uppercase;"
                "letter-spacing:.8px;border-bottom:1px solid #1e2235'>"
                f"<div>Metric</div><div style='color:#00d4aa'>{ticker}</div>"
                f"<div style='color:#40c4ff'>{cmp_ticker_tab}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            # Price & change
            p1c = "#00c853" if price_change >= 0 else "#ff1744"
            p2c = "#00c853" if cmp_chg_pct >= 0 else "#ff1744"
            _cmp_pair_row("Current Price",
                          f"{current_price:,.2f}", f"{cmp_price:,.2f}")
            _cmp_pair_row("Day Change",
                          f"{price_pct:+.2f}%", f"{cmp_chg_pct:+.2f}%", p1c, p2c)

            # AI signal
            _cmp_pair_row("AI Signal", signal, cmp_sig,
                          "#00c853" if "BUY" in signal else "#ff1744" if "SELL" in signal else "#ffd740",
                          "#00c853" if "BUY" in cmp_sig else "#ff1744" if "SELL" in cmp_sig else "#ffd740")
            _cmp_pair_row("AI Confidence",
                          f"{confidence*100:.1f}%", f"{cmp_conf*100:.1f}%",
                          "#00c853" if confidence >= 0.55 else "#ffd740",
                          "#00c853" if cmp_conf >= 0.55 else "#ffd740")

            # RSI
            rsi_c1 = "#ff1744" if rsi_val >= 70 else "#00c853" if rsi_val <= 30 else "#ffd740"
            rsi_c2 = "#ff1744" if cmp_rsi >= 70 else "#00c853" if cmp_rsi <= 30 else "#ffd740"
            _cmp_pair_row("RSI (14)",
                          f"{rsi_val:.1f}", f"{cmp_rsi:.1f}", rsi_c1, rsi_c2)

            # Moving averages
            ma50_c1  = "#00c853" if current_price > my_sma50 else "#ff1744"
            ma50_c2  = "#00c853" if cmp_price > cmp_sma50 else "#ff1744"
            ma200_c1 = "#00c853" if current_price > my_sma200 else "#ff1744"
            ma200_c2 = "#00c853" if cmp_price > cmp_sma200 else "#ff1744"
            _cmp_pair_row("vs MA50",
                          ("Above" if current_price > my_sma50 else "Below"),
                          ("Above" if cmp_price > cmp_sma50 else "Below"),
                          ma50_c1, ma50_c2)
            _cmp_pair_row("vs MA200",
                          ("Above" if current_price > my_sma200 else "Below"),
                          ("Above" if cmp_price > cmp_sma200 else "Below"),
                          ma200_c1, ma200_c2)

            # Risk & volatility
            rv1c = "#00c853" if risk_cat == "Low" else "#ffd740" if risk_cat == "Medium" else "#ff1744"
            rv2c = "#00c853" if cmp_risk == "Low" else "#ffd740" if cmp_risk == "Medium" else "#ff1744"
            _cmp_pair_row("Ann. Volatility",
                          f"{ann_vol:.1f}%", f"{cmp_vol:.1f}%")
            _cmp_pair_row("Risk Level",
                          f"{risk_cat} Risk", f"{cmp_risk} Risk", rv1c, rv2c)

            # Returns & growth
            cagr_c1 = "#00c853" if historical_cagr >= 0 else "#ff1744"
            cagr_c2 = "#00c853" if cmp_cagr >= 0 else "#ff1744"
            _cmp_pair_row("CAGR (since 2015)",
                          f"{historical_cagr*100:.1f}%", f"{cmp_cagr*100:.1f}%",
                          cagr_c1, cagr_c2)
            _cmp_pair_row("Cumulative Return",
                          f"{cum_ret*100:+.0f}%", f"{cmp_cumret*100:+.0f}%",
                          "#00c853" if cum_ret >= 0 else "#ff1744",
                          "#00c853" if cmp_cumret >= 0 else "#ff1744")
            dd_c1 = "#00c853" if max_dd >= -0.20 else "#ffd740" if max_dd >= -0.40 else "#ff1744"
            dd_c2 = "#00c853" if cmp_maxdd >= -0.20 else "#ffd740" if cmp_maxdd >= -0.40 else "#ff1744"
            _cmp_pair_row("Max Drawdown",
                          f"{max_dd*100:.1f}%", f"{cmp_maxdd*100:.1f}%", dd_c1, dd_c2)

            # Sentiment
            sent1 = ("Bullish" if sentiment_score > 0.55
                     else "Bearish" if sentiment_score < 0.45 else "Neutral")
            _cmp_hl2 = fetch_headlines(cmp_ticker_tab)
            if _cmp_hl2:
                _cmp_kw   = [_keyword_sentiment(h) for h in _cmp_hl2]
                _cmp_pos  = sum(1 for l, _ in _cmp_kw if l == "POSITIVE")
                cmp_sent  = 0.5 + (_cmp_pos / len(_cmp_kw) - 0.5) * 0.6
            else:
                cmp_sent = 0.5
            sent2 = ("Bullish" if cmp_sent > 0.55
                     else "Bearish" if cmp_sent < 0.45 else "Neutral")
            s1c = "#00c853" if sentiment_score > 0.55 else "#ff1744" if sentiment_score < 0.45 else "#ffd740"
            s2c = "#00c853" if cmp_sent > 0.55 else "#ff1744" if cmp_sent < 0.45 else "#ffd740"
            _cmp_pair_row("News Sentiment", sent1, sent2, s1c, s2c)

            st.markdown(
                "<div style='padding:4px 14px;background:#0e1220;border-radius:0 0 8px 8px;"
                "font-size:.68rem;color:#3d4566'>AI signals use ensemble GRU + XGBoost. "
                "Not financial advice.</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── Normalised performance chart ────────────────────────────────
            st.markdown(
                '<div class="section-header">📈 Normalised Historical Performance (base = 100)</div>',
                unsafe_allow_html=True,
            )
            cmp_lookback_opts = {
                "6 Months": 126, "1 Year": 252, "2 Years": 504, "5 Years": 1260,
            }
            cmp_lb_label = st.select_slider(
                "Comparison period",
                options=list(cmp_lookback_opts.keys()),
                value="2 Years",
                key="cmp_lb_slider",
            )
            cmp_lb_days = cmp_lookback_opts[cmp_lb_label]
            st.plotly_chart(
                make_normalised_compare_chart(df, ticker, df_cmp, cmp_ticker_tab, cmp_lb_days),
                use_container_width=True,
            )

            # ── Forecast comparison ─────────────────────────────────────────
            st.markdown(
                '<div class="section-header">🔮 Side-by-Side Forecast Comparison</div>',
                unsafe_allow_html=True,
            )

            # Build forecast for comparison stock
            cmp_ma_spread   = (cmp_ema20 - cmp_ema50) / cmp_ema50 if cmp_ema50 != 0 else 0
            cmp_trend_str   = float(np.clip(
                cmp_ma_spread * 5 + float(df_cmp["LogReturn"].mean()) * 50, -1, 1
            )) if "LogReturn" in df_cmp.columns else 0.0
            cmp_mkt_regime  = float(np.clip(
                df_cmp["CumLogRet30"].iloc[-1] if "CumLogRet30" in df_cmp.columns else 0.0,
                -0.15, 0.15,
            ))

            cmp_fc = dynamic_compound_forecast(
                current_price   = cmp_price,
                years           = 5,
                historical_cagr = cmp_cagr,
                confidence      = cmp_conf,
                trend_strength  = cmp_trend_str,
                ann_vol         = cmp_vol / 100.0,
                sentiment_score = cmp_sent,
                market_regime   = cmp_mkt_regime,
                frequency       = "monthly",
                n_simulations   = 200,
            )

            my_fc = dynamic_compound_forecast(
                current_price   = current_price,
                years           = 5,
                historical_cagr = historical_cagr,
                confidence      = confidence,
                trend_strength  = trend_strength,
                ann_vol         = ann_vol / 100.0,
                sentiment_score = sentiment_score,
                market_regime   = market_regime,
                frequency       = "monthly",
                n_simulations   = 200,
            )

            # Forecast table header
            st.markdown(
                "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                "gap:4px;padding:8px 14px;background:#0e1220;border-radius:8px 8px 0 0;"
                "font-size:.68rem;color:#6b7494;font-weight:700;text-transform:uppercase;"
                "letter-spacing:.8px;border-bottom:1px solid #1e2235'>"
                f"<div>Horizon</div><div style='color:#00d4aa'>{ticker}</div>"
                f"<div style='color:#40c4ff'>{cmp_ticker_tab}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            for label, my_val, cmp_val in [
                ("1 Year",  my_fc["f1y"], cmp_fc["f1y"]),
                ("3 Years", my_fc["f3y"], cmp_fc["f3y"]),
                ("5 Years", my_fc["f5y"], cmp_fc["f5y"]),
            ]:
                my_pnl  = (my_val  - current_price) / current_price * 100
                cmp_pnl = (cmp_val - cmp_price)     / cmp_price     * 100
                mc1 = "#00c853" if my_pnl  >= 0 else "#ff1744"
                mc2 = "#00c853" if cmp_pnl >= 0 else "#ff1744"
                st.markdown(
                    f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                    f"gap:4px;padding:9px 14px;background:#12172a;"
                    f"border-bottom:1px solid #1e2235;font-size:.80rem;align-items:center'>"
                    f"<div style='color:#6b7494;font-size:.72rem;font-weight:600;"
                    f"text-transform:uppercase;letter-spacing:.6px'>{label}</div>"
                    f"<div style='color:{mc1};font-weight:700'>{my_val:,.0f}"
                    f" <span style='font-size:.70rem'>({my_pnl:+.1f}%)</span></div>"
                    f"<div style='color:{mc2};font-weight:700'>{cmp_val:,.0f}"
                    f" <span style='font-size:.70rem'>({cmp_pnl:+.1f}%)</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Implied CAGR row
            my_cagr5  = ((my_fc["f5y"]  / current_price) ** (1/5) - 1) * 100
            cmp_cagr5 = ((cmp_fc["f5y"] / cmp_price)     ** (1/5) - 1) * 100
            c1c = "#00c853" if my_cagr5  >= 0 else "#ff1744"
            c2c = "#00c853" if cmp_cagr5 >= 0 else "#ff1744"
            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                f"gap:4px;padding:9px 14px;background:#12172a;"
                f"border-bottom:1px solid #1e2235;font-size:.80rem;align-items:center'>"
                f"<div style='color:#6b7494;font-size:.72rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:.6px'>Implied CAGR (5Y)</div>"
                f"<div style='color:{c1c};font-weight:700'>{my_cagr5:+.1f}%</div>"
                f"<div style='color:{c2c};font-weight:700'>{cmp_cagr5:+.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # AI confidence row
            conf_c1 = "#00c853" if confidence >= 0.55 else "#ffd740"
            conf_c2 = "#00c853" if cmp_conf >= 0.55 else "#ffd740"
            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                f"gap:4px;padding:9px 14px;background:#12172a;"
                f"border-bottom:1px solid #1e2235;font-size:.80rem;align-items:center'>"
                f"<div style='color:#6b7494;font-size:.72rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:.6px'>AI Confidence</div>"
                f"<div style='color:{conf_c1};font-weight:700'>{confidence*100:.1f}%</div>"
                f"<div style='color:{conf_c2};font-weight:700'>{cmp_conf*100:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='padding:4px 14px;background:#0e1220;border-radius:0 0 8px 8px;"
                "font-size:.68rem;color:#3d4566'>Forecast uses dynamic compounding engine. "
                "For educational use only — not financial advice.</div>",
                unsafe_allow_html=True,
            )


# TAB 3 — AI SCREENER
# ══════════════════════════════════════════════════════════

with tab_sc:
    st.markdown('<div class="section-header">🔍 AI Stock Screener</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6b7494;font-size:.82rem'>"
        "Scans all stocks in the universe and ranks them by AI signal strength, "
        "RSI, and trend. Results are cached for 10 minutes.</p>",
        unsafe_allow_html=True,
    )

    sector_options = ["All Sectors"] + list(SECTORS.keys())
    sc_sector = st.selectbox("Filter by Sector", sector_options, key="screener_sector")
    run_btn   = st.button("🔍 Run Full Screen", key="run_screener")

    # Filter tickers
    if sc_sector == "All Sectors":
        scan_tickers = [v for v in STOCK_LIST.values() if v]
    else:
        scan_tickers = SECTORS.get(sc_sector, [])

    if run_btn or ("screener_results" in st.session_state
                   and st.session_state.get("screener_sector_last") == sc_sector):

        if run_btn:
            results = []
            prog = st.progress(0, text="Scanning…")
            for i, t in enumerate(scan_tickers):
                prog.progress((i + 1) / len(scan_tickers), text=f"Scanning {t}…")
                try:
                    dft = fetch_data(t)
                    if dft is not None and not dft.empty:
                        row = quick_screen(dft, gru_model, xgb_model, scaler)
                        if row:
                            row["ticker"] = t
                            row["name"]   = TICKER_LABELS.get(t, t)
                            results.append(row)
                except Exception:
                    pass
            prog.empty()
            st.session_state["screener_results"]      = results
            st.session_state["screener_sector_last"]  = sc_sector

        results = st.session_state.get("screener_results", [])

        if results:
            # Sort by score desc
            results_sorted = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

            # Signal colour map
            sig_color = {
                "STRONG BUY": "#00c853", "BUY": "#00e676",
                "HOLD": "#ffd740",
                "SELL": "#ff6e40", "STRONG SELL": "#ff1744",
            }

            # Table header
            st.markdown(
                "<div style='display:grid;grid-template-columns:80px 140px 80px 80px 80px 70px 70px 90px;"
                "gap:4px;padding:8px 12px;background:#0e1220;border-radius:8px 8px 0 0;"
                "font-size:.68rem;color:#6b7494;font-weight:700;text-transform:uppercase;"
                "letter-spacing:.8px;border-bottom:1px solid #1e2235'>"
                "<div>Ticker</div><div>Name</div><div>Price</div><div>Signal</div>"
                "<div>Conf%</div><div>RSI</div><div>Mom%</div><div>Ann Vol%</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            for row in results_sorted:
                sig      = row["signal"]
                sclr     = sig_color.get(sig, "#6b7494")
                conf_c   = "#00c853" if row["confidence"] >= 0.55 else "#ffd740"
                rsi_c    = ("#ff1744" if row["rsi"] >= 70
                            else "#00c853" if row["rsi"] <= 30 else "#9ba3bf")
                mom_c    = "#00c853" if row["momentum"] >= 0 else "#ff1744"
                st.markdown(
                    f"<div style='display:grid;grid-template-columns:80px 140px 80px 80px 80px 70px 70px 90px;"
                    f"gap:4px;padding:9px 12px;background:#12172a;border-bottom:1px solid #1e2235;"
                    f"font-size:.78rem;align-items:center'>"
                    f"<div style='font-weight:700;color:#e8ecf4'>{row['ticker']}</div>"
                    f"<div style='color:#9ba3bf'>{row['name']}</div>"
                    f"<div style='font-weight:600'>{row['price']:,.1f}</div>"
                    f"<div style='color:{sclr};font-weight:700;font-size:.72rem'>{sig}</div>"
                    f"<div style='color:{conf_c}'>{row['confidence']*100:.0f}%</div>"
                    f"<div style='color:{rsi_c}'>{row['rsi']:.0f}</div>"
                    f"<div style='color:{mom_c}'>{row['momentum']:+.1f}%</div>"
                    f"<div style='color:#9ba3bf'>{row['ann_vol']:.0f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<div style='padding:4px 12px;background:#0e1220;border-radius:0 0 8px 8px;"
                "font-size:.68rem;color:#3d4566'>"
                f"Scanned {len(results)} stocks &nbsp;·&nbsp; "
                f"Strong Buy: {sum(1 for r in results if r['signal']=='STRONG BUY')} &nbsp;·&nbsp;"
                f" Buy: {sum(1 for r in results if r['signal']=='BUY')} &nbsp;·&nbsp;"
                f" Hold: {sum(1 for r in results if r['signal']=='HOLD')} &nbsp;·&nbsp;"
                f" Sell/Strong Sell: {sum(1 for r in results if 'SELL' in r['signal'])}"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No results. Click **Run Full Screen** to scan.")
    else:
        st.info("Select a sector and click **Run Full Screen** to scan the market.")


# ══════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ══════════════════════════════════════════════════════════

with tab_bt:
    st.markdown('<div class="section-header">⚡ Professional Backtest Engine</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6b7494;font-size:.82rem'>"
        "Walk-forward backtest with transaction costs and slippage. "
        "Buy when ensemble confidence ≥ threshold; sell when confidence ≤ (1 − threshold). "
        "Benchmark: buy-and-hold the same stock over the same period.</p>",
        unsafe_allow_html=True,
    )

    bt1, bt2, bt3, bt4 = st.columns(4)
    with bt1:
        bt_window = st.selectbox("Backtest window", ["1 Year (252d)", "18 Months (378d)", "2 Years (504d)"],
                                 key="bt_window")
        bt_days   = {"1 Year (252d)": 252, "18 Months (378d)": 378, "2 Years (504d)": 504}[bt_window]
    with bt2:
        bt_cost = st.number_input("Commission %", min_value=0.0, max_value=2.0, value=0.2, step=0.05,
                                   key="bt_cost") / 100
    with bt3:
        bt_slip = st.number_input("Slippage %", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                                   key="bt_slip") / 100
    with bt4:
        bt_thr  = st.slider("Signal threshold", 0.51, 0.75, 0.55, step=0.01, key="bt_thr")

    run_bt = st.button("⚡ Run Backtest", key="run_backtest")

    if run_bt or "backtest_result" in st.session_state:
        if run_bt:
            if not models_ready:
                st.warning("Train the model first — no trained ensemble available.")
            else:
                with st.spinner("Running backtest…"):
                    bt_result = run_backtest(
                        df, gru_model, xgb_model, scaler,
                        initial_capital=float(investment),
                        txn_cost=bt_cost,
                        slippage=bt_slip,
                        conf_threshold=bt_thr,
                        window_days=bt_days,
                    )
                st.session_state["backtest_result"] = bt_result

        bt_result = st.session_state.get("backtest_result", {})

        if bt_result:
            # Equity curve
            st.plotly_chart(make_equity_chart(bt_result, ticker), use_container_width=True)

            # Performance metrics
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            bm1, bm2, bm3, bm4, bm5, bm6 = st.columns(6)

            total_ret  = bt_result["total_return"]
            bh_final   = bt_result["bh_vals"][-1] if bt_result.get("bh_vals") else investment
            bh_ret     = (bh_final / investment - 1) * 100 if investment > 0 else 0
            sharpe     = bt_result["sharpe"]
            max_dd_bt  = bt_result["max_dd"]
            win_rate   = bt_result["win_rate"]
            n_trades   = bt_result["total_trades"]

            with bm1:
                rc = "#00c853" if total_ret >= 0 else "#ff1744"
                st.markdown(mc("Strategy Return", f"{total_ret:+.1f}%",
                               "AI ensemble trades", rc), unsafe_allow_html=True)
            with bm2:
                bh_c = "#00c853" if bh_ret >= 0 else "#ff1744"
                st.markdown(mc("Buy & Hold Return", f"{bh_ret:+.1f}%",
                               "Benchmark", bh_c), unsafe_allow_html=True)
            with bm3:
                alpha = total_ret - bh_ret
                ac = "#00c853" if alpha >= 0 else "#ff1744"
                st.markdown(mc("Alpha", f"{alpha:+.1f}%",
                               "vs buy-and-hold", ac), unsafe_allow_html=True)
            with bm4:
                sc_sh = "#00c853" if sharpe >= 1.0 else ("#ffd740" if sharpe >= 0.5 else "#ff1744")
                st.markdown(mc("Sharpe Ratio", f"{sharpe:.2f}",
                               "Risk-adjusted return", sc_sh), unsafe_allow_html=True)
            with bm5:
                dd_c = "#00c853" if max_dd_bt >= -20 else ("#ffd740" if max_dd_bt >= -35 else "#ff1744")
                st.markdown(mc("Max Drawdown", f"{max_dd_bt:.1f}%",
                               "Strategy worst decline", dd_c), unsafe_allow_html=True)
            with bm6:
                wc = "#00c853" if win_rate >= 55 else ("#ffd740" if win_rate >= 45 else "#ff1744")
                st.markdown(mc("Win Rate", f"{win_rate:.0f}%",
                               f"{n_trades} trades total", wc), unsafe_allow_html=True)

            # Trade log
            trades = bt_result.get("trades", [])
            if trades:
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">📋 Recent Trade Log</div>',
                            unsafe_allow_html=True)
                tl_header = (
                    "<div style='display:grid;grid-template-columns:100px 70px 120px 80px 80px;"
                    "gap:4px;padding:8px 12px;background:#0e1220;border-radius:8px 8px 0 0;"
                    "font-size:.68rem;color:#6b7494;font-weight:700;text-transform:uppercase'>"
                    "<div>Date</div><div>Type</div><div>Price</div><div>Conf</div><div>P&L %</div>"
                    "</div>"
                )
                st.markdown(tl_header, unsafe_allow_html=True)
                for tr in trades[-15:]:
                    tc   = "#00c853" if tr["type"] == "BUY" else "#ff1744"
                    pnl  = tr.get("pnl_pct", "—")
                    pnlt = f"{pnl:+.1f}%" if isinstance(pnl, float) else "—"
                    pc   = "#00c853" if isinstance(pnl, float) and pnl > 0 else "#ff1744" if isinstance(pnl, float) else "#6b7494"
                    st.markdown(
                        f"<div style='display:grid;grid-template-columns:100px 70px 120px 80px 80px;"
                        f"gap:4px;padding:7px 12px;background:#12172a;border-bottom:1px solid #1e2235;"
                        f"font-size:.78rem;align-items:center'>"
                        f"<div style='color:#9ba3bf'>{tr['date']}</div>"
                        f"<div style='color:{tc};font-weight:700'>{tr['type']}</div>"
                        f"<div style='color:#e8ecf4'>{tr['price']:,.2f}</div>"
                        f"<div style='color:#9ba3bf'>{tr['conf']:.2f}</div>"
                        f"<div style='color:{pc};font-weight:600'>{pnlt}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    "<div style='padding:4px 12px;background:#0e1220;border-radius:0 0 8px 8px;"
                    "font-size:.68rem;color:#3d4566'>Transaction costs + slippage applied to every trade</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Backtest returned no results. Ensure models are trained and there is enough data.")
    else:
        st.info("Configure settings above and click **Run Backtest** to start.")


# ══════════════════════════════════════════════════════════
# TAB 5 — FORECAST
# ══════════════════════════════════════════════════════════

with tab_fc:
    st.markdown('<div class="section-header">🔮 Dynamic Sequential Compounding Forecast & Portfolio Simulation</div>',
                unsafe_allow_html=True)

    freq_choice = st.select_slider(
        "Compounding frequency",
        options=["yearly", "monthly", "weekly"],
        value="monthly",
        help="How often growth is re-applied to the running price.",
    )

    fc = dynamic_compound_forecast(
        current_price   = current_price,
        years           = 5,
        historical_cagr = historical_cagr,
        confidence      = confidence,
        trend_strength  = trend_strength,
        ann_vol         = ann_vol / 100.0,
        sentiment_score = sentiment_score,
        market_regime   = market_regime,
        frequency       = freq_choice,
        n_simulations   = 300,
    )
    f1y = fc["f1y"]
    f3y = fc["f3y"]
    f5y = fc["f5y"]

    fc1, fc2 = st.columns([1.1, 0.9])

    with fc1:
        st.plotly_chart(make_compound_forecast_chart(fc), use_container_width=True)

        rates = fc.get("annual_rates", [])
        if rates:
            ppy        = fc["ppy"]
            early_rate = float(np.mean(rates[:ppy]))
            late_rate  = float(np.mean(rates[-ppy:]))
            mid_rate   = float(np.mean(rates[len(rates)//2 - ppy//2 : len(rates)//2 + ppy//2]))
            ra1, ra2, ra3 = st.columns(3)
            with ra1:
                st.markdown(mc("Year 1 Rate", f"{early_rate*100:+.1f}%", "AI + momentum",
                               "#00c853" if early_rate >= 0 else "#ff1744"), unsafe_allow_html=True)
            with ra2:
                st.markdown(mc("Mid-term Rate", f"{mid_rate*100:+.1f}%", "Mean reversion",
                               "#00c853" if mid_rate >= 0 else "#ff1744"), unsafe_allow_html=True)
            with ra3:
                st.markdown(mc("Year 5 Rate", f"{late_rate*100:+.1f}%", "CAGR-anchored",
                               "#00c853" if late_rate >= 0 else "#ff1744"), unsafe_allow_html=True)

    with fc2:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        pt1, pt2, pt3 = st.columns(3)
        for col, label, val in zip([pt1, pt2, pt3],
                                    ["1 Year", "3 Years", "5 Years"],
                                    [f1y, f3y, f5y]):
            pnl = (val - current_price) / current_price * 100
            clr = "#00c853" if pnl >= 0 else "#ff1744"
            with col:
                st.markdown(mc(label, f"{val:,.0f}", f"{pnl:+.1f}%", clr),
                            unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        implied_cagr_1y = (f1y / current_price - 1) * 100
        implied_cagr_5y = ((f5y / current_price) ** (1.0 / 5) - 1) * 100
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(mc("Implied CAGR 1Y", f"{implied_cagr_1y:+.1f}%", "Dynamic path",
                           "#00c853" if implied_cagr_1y >= 0 else "#ff1744"), unsafe_allow_html=True)
        with ic2:
            st.markdown(mc("Implied CAGR 5Y", f"{implied_cagr_5y:+.1f}%", "Sequential compound",
                           "#00c853" if implied_cagr_5y >= 0 else "#ff1744"), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        reg_label = ("Bull" if market_regime > 0.02
                     else "Bear" if market_regime < -0.02 else "Neutral")
        reg_clr   = ("#00c853" if market_regime > 0.02
                     else "#ff1744" if market_regime < -0.02 else "#ffd740")
        sent_label2 = ("Bullish" if sentiment_score > 0.55
                       else "Bearish" if sentiment_score < 0.45 else "Neutral")
        si1, si2 = st.columns(2)
        with si1:
            st.markdown(mc("Market Regime", reg_label,
                           f"30d log-ret: {market_regime*100:+.1f}%", reg_clr),
                        unsafe_allow_html=True)
        with si2:
            sc2 = ("#00c853" if sentiment_score > 0.55
                   else "#ff1744" if sentiment_score < 0.45 else "#ffd740")
            st.markdown(mc("News Sentiment", sent_label2,
                           f"Score: {sentiment_score:.2f}", sc2), unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:.77rem;color:#6b7494;font-weight:600;"
            "text-transform:uppercase;letter-spacing:1px'>Portfolio Simulation</p>",
            unsafe_allow_html=True,
        )
        for label, val in [("1 Year", f1y), ("3 Years", f3y), ("5 Years", f5y)]:
            port_val = investment * (val / current_price)
            profit   = port_val - investment
            clr      = "#00c853" if profit >= 0 else "#ff1744"
            st.markdown(
                f'<div class="forecast-row">'
                f'<div><div class="forecast-label">{label}</div>'
                f'<div class="forecast-val">{port_val:,.0f}</div></div>'
                f'<div style="text-align:right">'
                f'<div class="forecast-label">Invested {investment:,.0f}</div>'
                f'<div class="forecast-pnl" style="color:{clr}">{profit:+,.0f}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # Risk analysis sub-section inside Forecast
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📉 Risk Analysis</div>', unsafe_allow_html=True)
    rk1, rk2 = st.columns([1.6, 1.0])
    with rk1:
        st.plotly_chart(make_drawdown_chart(df, ticker), use_container_width=True)
    with rk2:
        # Value at Risk
        lr_series = df["LogReturn"].dropna() if "LogReturn" in df.columns else pd.Series([])
        if len(lr_series) > 30:
            var95 = float(np.percentile(lr_series, 5)) * current_price
            var99 = float(np.percentile(lr_series, 1)) * current_price
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(mc("Daily VaR (95%)", f"{abs(var95):,.0f}",
                           "Max 1-day loss at 95% confidence", "#ffd740"),
                        unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(mc("Daily VaR (99%)", f"{abs(var99):,.0f}",
                           "Max 1-day loss at 99% confidence", "#ff6e40"),
                        unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(mc("Annualised Vol", f"{ann_vol:.1f}%",
                           "Realised from log returns", "#9ba3bf"),
                        unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(mc("Calmar Ratio",
                           f"{(historical_cagr / max(abs(max_dd), 0.01)):.2f}",
                           "CAGR / Max Drawdown", "#40c4ff"),
                        unsafe_allow_html=True)

    # Feature Importance (XGBoost)
    if xgb_model is not None:
        st.markdown('<div class="section-header">🔬 XGBoost Feature Importance</div>',
                    unsafe_allow_html=True)
        fi_fig = make_feature_importance_chart(xgb_model)
        if fi_fig:
            st.plotly_chart(fi_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 6 — NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════

with tab_nw:
    st.markdown('<div class="section-header">📰 News Sentiment Analysis</div>',
                unsafe_allow_html=True)

    with st.spinner("Fetching headlines…"):
        headlines = fetch_headlines(ticker)

    if not headlines:
        st.info("No recent headlines found for this ticker.")
    else:
        with st.spinner("Scoring sentiment…"):
            news = run_sentiment(headlines)

        pos   = sum(1 for n in news if n["label"] == "POSITIVE")
        neg   = sum(1 for n in news if n["label"] == "NEGATIVE")
        neu   = sum(1 for n in news if n["label"] == "NEUTRAL")
        total = len(news)
        method       = news[0].get("method", "finbert") if news else "keyword"
        method_label = "FinBERT" if method == "finbert" else "Keyword AI"

        ns1, ns2 = st.columns([2, 1])
        with ns2:
            pos_pct = pos / total * 100
            clr     = "#00c853" if pos_pct >= 55 else ("#ff1744" if pos_pct < 35 else "#ffd740")
            overall = "Bullish" if pos_pct >= 55 else ("Bearish" if pos_pct < 35 else "Mixed")
            st.markdown(mc(f"{method_label} Verdict", overall,
                           f"🟢 {pos} · 🔴 {neg} · ⚪ {neu}  ({total} headlines)", clr),
                        unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(mc("Sentiment Score", f"{pos_pct:.0f}% Positive",
                           f"Scored by {method_label}", clr), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            # Sentiment bar
            bar_w = int(pos_pct)
            st.markdown(
                f"<div style='background:#1e2235;border-radius:6px;overflow:hidden;height:8px'>"
                f"<div style='width:{bar_w}%;height:8px;background:{clr};border-radius:6px'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with ns1:
            for item in news[:10]:
                if item["label"] == "POSITIVE":
                    lc, lt = "news-pos", "● POSITIVE"
                elif item["label"] == "NEGATIVE":
                    lc, lt = "news-neg", "● NEGATIVE"
                else:
                    lc, lt = "news-neu", "● NEUTRAL"
                st.markdown(
                    f'<div class="news-card">'
                    f'<div class="news-title">{item["headline"]}</div>'
                    f'<div class="{lc}">{lt} &nbsp;·&nbsp; {item["score"]*100:.0f}% confidence'
                    f' &nbsp;·&nbsp; {item.get("method","keyword").upper()}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#2e3450;font-size:.7rem'>"
    "PSX AI Quant Platform v4 &nbsp;·&nbsp; GRU + XGBoost Ensemble &nbsp;·&nbsp; "
    "FinBERT Sentiment &nbsp;·&nbsp; OBV · VWAP · Anomaly Detection &nbsp;·&nbsp; "
    "Walk-Forward Backtest &nbsp;·&nbsp; For educational use only — not financial advice</p>",
    unsafe_allow_html=True,
)
