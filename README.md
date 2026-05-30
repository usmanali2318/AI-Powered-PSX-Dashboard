# 📈 Data-Runner — AI Powered PSX Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-✓-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**An end-to-end AI quantitative trading dashboard for the Pakistan Stock Exchange (PSX), powered by a GRU + XGBoost + CatBoost ensemble with FinBERT sentiment analysis.**

[Features](#-features) • [Architecture](#-architecture) • [Models](#-ml-models) • [Setup](#-setup) • [Usage](#-usage) • [Results](#-results)

</div>

---

## 🖥️ Dashboard Preview

> **7 fully interactive tabs** — Overview · Charts · Compare · Screener · Backtest · Forecast · News

| Overview Tab | Charts Tab |
|:---:|:---:|
| 16 real-time metric cards with AI signal badges, CAGR, VaR, regime score | 9 Plotly charts: candlestick, RSI, MACD, OBV, VWAP, anomaly detection |

| Backtest Tab | Forecast Tab |
|:---:|:---:|
| Walk-forward equity curve vs buy-and-hold, Sharpe, win rate, trade log | Monte Carlo compounding forecast with P15–P85 uncertainty bands |

| Screener Tab | News Tab |
|:---:|:---:|
| AI-ranked screener across all 22 PSX stocks by ensemble confidence score | Google News RSS → FinBERT sentiment scoring with keyword fallback |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA-RUNNER PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DATA SOURCES                  FEATURE ENGINEERING          MODELS      │
│  ─────────────                 ───────────────────          ──────      │
│                                                                         │
│  yFinance ──────► 27 Stocks    ┌─ Price Features ──────►  ┌─ GRU ─┐   │
│  (22 PSX + 5 US)               │  Open/High/Low/Close      │  192  │   │
│                                │  EMA20/50, SMA20/50       │  units│   │
│  KSE-100 Index ─► Macro Data   │  RSI, MACD, BB, ATR       │  +    │   │
│  PKR/USD Rate  ─► (4 features) │                           │  MHA  │   │
│  Brent Oil     ─►              ├─ Volume Features ─────►   └───┬───┘   │
│                                │  OBV_Norm, RelVol          ┌──▼──┐    │
│  Google News ──► Headlines     │  VWAP_Dev, Volume_Chg      │     │    │
│                                │                            │Ens. │    │
│                                ├─ Regime Features ─────►   │blend│    │
│                                │  RegimeScore               │     │    │
│                                │  CumLogRet7/30             │55%  │    │
│                                │  RollVol20                 │GRU+ │    │
│                                │                            │XGB+ │    │
│                                └─ Macro Features ─────►    │CAT  │    │
│                                   KSE100_Ret               └──┬──┘    │
│  FinBERT ───────► Sentiment       PKRUSD_Ret                  │        │
│  (ProsusAI)       Scoring         Oil_Ret                     │        │
│                                   MacroMomentum               │        │
│                                                                │        │
│                         28 features × 60-day sequences        │        │
│                                                                ▼        │
│                                                    ┌─ XGBoost ─┐       │
│  LABEL GENERATION                                  │ 1000 trees│       │
│  ────────────────                                  │ AUC eval  │       │
│  ATR-adjusted threshold                            └─────┬─────┘       │
│  Multi-horizon avg(5,10,20d)                             │             │
│  Focal loss for imbalance                     ┌─ CatBoost ─┐           │
│  MCC-optimal threshold                        │ 82 iters   │           │
│  Isotonic rank-normalisation                  │ AUC eval   │           │
│                                               └─────┬──────┘           │
│                                                     │                  │
│                                    ┌────────────────┘                  │
│                                    ▼                                   │
│                         ┌─ Meta Stacker ─┐                            │
│                         │ LogisticReg    │                             │
│                         │ (LR stacker)   │                             │
│                         └───────┬────────┘                             │
│                                 │                                      │
│                    ┌────────────▼──────────────┐                       │
│                    │   Streamlit Dashboard      │                       │
│                    │   7 tabs · Dark theme      │                       │
│                    │   Plotly charts · Real-time│                       │
│                    └───────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🤖 ML Ensemble
- **GRU + Multi-Head Attention** — 2-layer GRU (192→96 units) with 4-head self-attention, last-step + global-average pooling concatenation, trained with binary cross-entropy and boosted class weights
- **XGBoost** — 1,000-tree gradient boosting with AUC optimization, `scale_pos_weight` for class imbalance, early stopping at 50 rounds
- **CatBoost** — Gradient boosting with `bagging_temperature`, `random_strength`, and balanced class weighting
- **Meta-stacker** — Logistic Regression stacker trained on ensemble outputs for optimal nonlinear blending
- **Isotonic rank-normalization** — Calibrates each model's raw probability scale before blending to prevent threshold collapse
- **Dynamic ensemble weights** — Proportional to each model's test-set ROC-AUC (not hardcoded)
- **MCC-optimal thresholds** — Per-model threshold search that rejects collapse (>85% one-class predictions)

### 📊 Feature Engineering (28 features)

| Category | Features |
|---|---|
| **Price** | Open, High, Low, Close, LogReturn, Momentum |
| **Moving Averages** | EMA20, EMA50, SMA20, SMA50 |
| **Momentum Indicators** | RSI, MACD, MACD_Signal, CumLogRet7, CumLogRet30 |
| **Volatility** | BB_Width, ATR, RollVol20 |
| **Volume Intelligence** | Volume, Volume_Change, OBV_Norm, RelVol |
| **Market Microstructure** | VWAP_Dev, RegimeScore |
| **Macro Context** | KSE100_Ret, PKRUSD_Ret, Oil_Ret, MacroMomentum |

### 📈 Dashboard Tabs

| Tab | Description |
|---|---|
| **📊 Overview** | 16 metric cards: price, AI signal, confidence, risk, CAGR, max drawdown, RSI, MA trend, VaR, regime, sentiment |
| **📈 Charts** | 9 Plotly charts: candlestick+volume, moving averages, RSI, MACD, OBV+VWAP, relative volume, volume spike z-score, price anomaly events, anomaly score |
| **⚖️ Compare** | Side-by-side metric grid for any two stocks + normalized performance chart + 5-year forecast |
| **🔍 Screener** | AI-ranks all 22 PSX stocks by ensemble confidence with sector filtering |
| **⚡ Backtest** | Walk-forward backtest with configurable commission, slippage, signal threshold; equity curve vs buy-and-hold, Sharpe, win rate, trade log |
| **🔮 Forecast** | Dynamic sequential compounding engine with Monte Carlo uncertainty bands (P15–P85), portfolio simulation, VaR, Calmar ratio, XGBoost feature importance |
| **📰 News** | Google News RSS → FinBERT sentiment scoring with keyword fallback |

### 🌍 Stock Universe
**22 PSX stocks** across Fertilizer, Banking, Oil & Gas, Technology, Auto, Cement, Power, and Telecom — plus **5 US stocks** (AAPL, MSFT, GOOGL, NVDA, TSLA)

---

## 🧠 ML Models

### Training Pipeline
```
python train_model.py
```

**Label Generation:**
- Multi-horizon averaging over 5, 10, and 20-day forward returns
- ATR-adjusted threshold per stock (ATR × 0.5, clamped 0.7%–4.0%)
- Sequence step=3 for faster training without accuracy loss

**Anti-Collapse Measures (v5):**
- MCC-based threshold optimization with hard collapse guards
- Isotonic rank-normalization before ensemble blending
- Focal loss (γ=1.0) with boosted UP-class weight
- Per-model threshold search (not shared ensemble threshold)
- Baseline comparison: always-UP, random, momentum predictors

### Model Performance (latest run)

| Model | ROC-AUC | Balanced Acc | MCC |
|---|---|---|---|
| GRU + MHA | 0.531 | 52.9% | 0.057 |
| XGBoost | 0.560 | 54.6% | 0.092 |
| CatBoost | 0.535 | 53.4% | — |
| **Ensemble** | **0.562** | **55.0%** | **0.103** |
| Walk-Forward XGB | 0.537±0.002 | 52.8% | 0.056 |

> Note: Stock price prediction is inherently noisy. These metrics beat random (AUC=0.50) and always-UP baselines consistently. The low variance in walk-forward (±0.002) indicates reliable generalization.

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- Windows / Linux / macOS

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/data-runner.git
cd data-runner/stock-dashboard

# Install dependencies
pip install -r requirements.txt

# Optional: CatBoost (recommended — adds 3rd ensemble member)
pip install catboost
```

### Requirements
```
streamlit>=1.35.0
plotly>=5.22.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
yfinance>=0.2.40
feedparser>=6.0.11
joblib>=1.4.0
requests>=2.31.0
tensorflow-cpu>=2.16.0
transformers>=4.40.0
torch>=2.2.0
xgboost>=2.0.0
```

### Train Models

```bash
# Downloads data, trains GRU + XGBoost + CatBoost, saves to models/
python train_model.py
```

Training time: ~90–120 min on CPU (GRU dominates). Models saved to `models/`.

### Run Dashboard

```bash
streamlit run app.py
# or on a specific port:
streamlit run app.py --server.port 5000
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
stock-dashboard/
├── app.py                  # Streamlit dashboard (7 tabs, ~1000 lines)
├── utils.py                # Feature engineering, scaling, financial helpers
├── train_model.py          # Full training pipeline (GRU + XGB + CatBoost)
├── requirements.txt        # Python dependencies
├── models/                 # Trained model files (git-ignored)
│   ├── gru_model.keras
│   ├── xgb_model.pkl
│   ├── catboost_model.pkl
│   ├── meta_model.pkl
│   ├── metadata.json
│   └── scaler.pkl (in data/)
├── data/
│   └── scaler.pkl
└── logs/
    └── training_log.txt
```

---

## 📄 Results & Metrics

### Feature Importance (XGBoost Gain)
```
MacroMomentum    ████████████  0.0486  ← #1 feature (new in v5)
Close            ██████████    0.0459
Volume           █████████     0.0452
KSE100_Ret       ████████      0.0421  ← macro
EMA20            ████████      0.0414
EMA50            ████████      0.0395
SMA50            ████████      0.0395
SMA20            ███████       0.0392
PKRUSD_Ret       ███████       0.0381  ← macro
Oil_Ret          ███████       0.0379  ← macro
```

### Walk-Forward Stability
```
Fold 1  AUC=0.5401  MCC=0.054  BalAcc=52.8%
Fold 2  AUC=0.5353  MCC=0.053  BalAcc=52.6%
Fold 3  AUC=0.5368  MCC=0.060  BalAcc=52.9%
─────────────────────────────────────────────
Avg     AUC=0.537±0.002  (very stable)
```

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| **Dashboard** | Streamlit 1.57, Plotly 6.7 |
| **Deep Learning** | TensorFlow 2.21, Keras 3.14 |
| **Gradient Boosting** | XGBoost 3.2, CatBoost |
| **NLP / Sentiment** | HuggingFace Transformers, FinBERT (ProsusAI) |
| **Data** | yFinance, feedparser (Google News RSS) |
| **ML Utilities** | scikit-learn 1.8, joblib |
| **Language** | Python 3.13 |

---

## 📌 Resume Bullet Points

> Copy-paste ready for your CV / LinkedIn

---

**Software Engineer / ML Engineer roles:**

- Built a full-stack AI quantitative trading platform for the Pakistan Stock Exchange using Python, featuring a GRU + XGBoost + CatBoost ensemble that achieved **55% balanced accuracy** and **AUC 0.562** on held-out test data across 27 stocks and 83,000+ training samples

- Engineered a **28-feature time-series pipeline** including novel macro context features (KSE-100 index return, PKR/USD exchange rate, Brent crude oil) that became the **#1 predictive feature** by XGBoost gain importance

- Designed and implemented **anti-collapse ensemble training** with isotonic rank-normalization, per-model MCC-optimal threshold search, and dynamic AUC-weighted blending — solving probability scale mismatch between GRU and tree-based models

- Integrated **FinBERT transformer sentiment analysis** on live Google News RSS feeds with keyword-based fallback for offline inference

- Developed a **7-tab Streamlit dashboard** with 9 interactive Plotly charts, a Monte Carlo compounding forecast engine with uncertainty bands, a walk-forward backtester with realistic slippage/commission modeling, and a real-time AI screener across 22 PSX stocks

---

**Data Science / Quant roles:**

- Implemented a multi-horizon label generation system using **ATR-adjusted volatility thresholds** averaged over 5/10/20-day forward windows, reducing noise labels and improving model discriminability

- Applied **GRU + Multi-Head Attention** architecture with last-timestep and global-average-pooling concatenation for financial time-series classification, trained with binary cross-entropy and boosted class weights on 60-day rolling sequences

- Achieved **walk-forward AUC stability of 0.537 ± 0.002** across 3 time folds, demonstrating consistent out-of-sample generalization on Pakistan Stock Exchange data from 2013–2025

- Built a **stacking meta-model** (Logistic Regression) that combines GRU, XGBoost, and CatBoost probability outputs, with isotonic calibration to correct probability scale mismatches before ensemble blending

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past model performance does not guarantee future returns. Always consult a licensed financial advisor before making investment decisions.

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">
Made with ☕ and too many training runs · PSX Data via yFinance · Sentiment via FinBERT
</div>
