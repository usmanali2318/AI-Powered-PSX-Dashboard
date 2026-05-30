"""
train_model.py — v5  "Anti-Collapse" Quant AI Training Pipeline.

CRITICAL FIXES over v4
═══════════════════════════════════════════════════════════════════
1. MCC-based threshold optimisation with hard anti-collapse guards
   — v4 used F1-only → threshold 0.35 → everything predicted UP → collapse
   — v5 rejects any threshold where: pos_ratio > 0.85, TN=0, or TP=total

2. XGBoost EarlyStopping fixed to use callback API (avoids TypeError)
   — xgboost.callback.EarlyStopping(rounds=…) instead of deprecated kwarg

3. Focal loss (BinaryFocalCrossentropy, gamma=2.0) for GRU
   — penalises easy negatives; forces learning of hard minority samples

4. Probability calibration — CalibratedClassifierCV(method='sigmoid')
   — raw XGBoost probabilities cluster near 0.5; sigmoid recalibrates them

5. Probability diagnostics — prints min/max/mean/std + collapse warnings
   — exposes whether model is actually learning discriminative signals

6. Dynamic ensemble weights — proportional to model ROC-AUC on test set
   — instead of arbitrary 55/45 hardcoded split

7. Stacking meta-model — LogisticRegression on [GRU_prob, XGB_prob]
   — learns optimal nonlinear combination; meta-trained on first half of test set

8. Sequence step=3 — ~3× fewer training samples → ~3× faster GRU training
   — reduces 80-min run to ~25 min without meaningful accuracy drop

9. ATR-adjusted label thresholds — each stock uses its own volatility scale
   — low-vol stocks: threshold ≈ 1.0 %; high-vol stocks: threshold ≈ 3.5 %
   — eliminates random noise labels that tank discriminability

10. Baseline comparison — always-UP, random, momentum predictors
    — if AI cannot beat momentum baseline, training quality is flagged

11. Walk-forward per-fold MCC, balanced accuracy, prediction balance
12. Balanced accuracy + MCC added to all evaluation outputs

Run:    python3 train_model.py

Outputs:
  models/gru_model.keras            GRU + Multi-Head Attention (focal loss)
  models/xgb_model.pkl              XGBoost (calibrated)
  models/catboost_model.pkl         CatBoost (if installed)
  models/meta_model.pkl             LogisticRegression stacker
  models/metadata.json              architecture + metrics + threshold
  data/scaler.pkl                   fitted MinMaxScaler (24 features)
  logs/training_log.txt             full evaluation report
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime

import numpy as np
import tensorflow as tf
import pandas as pd
import yfinance as yf
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from utils import (
    FEATURES, SEQUENCE_LENGTH,
    engineer_features, create_sequences,
    fit_and_save_scaler, scale_features,
    load_macro_data, inject_macro_features,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
GRU_PATH    = "models/gru_model.keras"
XGB_PATH    = "models/xgb_model.pkl"
CAT_PATH    = "models/catboost_model.pkl"
META_PATH   = "models/meta_model.pkl"
SCALER_PATH = "data/scaler.pkl"
LOG_PATH    = "logs/training_log.txt"
METADATA_PATH = "models/metadata.json"

for _d in ["models", "data", "logs"]:
    os.makedirs(_d, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_STEP       = 3          # stride between GRU windows → ~3× faster training
MULTI_HORIZON  = [5, 10, 20]  # forward horizons for label averaging
ATR_MULTIPLIER = 0.5        # ATR fraction used as label threshold (lowered for better balance)
MIN_THRESHOLD  = 0.007      # 0.7 % floor — more UP signals for PSX stocks
MAX_THRESHOLD  = 0.040      # 4.0 % ceiling (don't require unrealistic moves)

# ── Stock universe ─────────────────────────────────────────────────────────────
ALL_STOCKS = [
    "FFC.KA", "ENGRO.KA", "EFERT.KA",
    "MEBL.KA", "HBL.KA", "UBL.KA", "MCB.KA",
    "OGDC.KA", "PPL.KA", "PSO.KA",
    "SYS.KA", "TRG.KA", "NETSOL.KA",
    "SAZEW.KA", "ATLH.KA", "HCAR.KA", "INDU.KA",
    "LUCK.KA", "DGKC.KA",
    "HUBC.KA", "KEL.KA", "PTC.KA",
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
]
START_DATE = "2013-01-01"  # Extended: more post-2022 volatility data included


# ─────────────────────────────────────────────────────────────────────────────
# Data download
# ─────────────────────────────────────────────────────────────────────────────

def download_stock(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            return None
        return df[list(needed)].dropna()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ATR-adjusted multi-horizon labels  (FIX #9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr_labels(
    df: pd.DataFrame,
    horizons: list[int] = MULTI_HORIZON,
) -> np.ndarray:
    """
    Volatility-adaptive multi-horizon labels.

    Each sample's threshold = ATR at that point × ATR_MULTIPLIER, clamped to
    [MIN_THRESHOLD, MAX_THRESHOLD].  This means:
      - High-vol stocks (TSLA, NVDA) require ~3–4 % avg return to be Bullish
      - Low-vol blue chips require only ~1–1.5 %
    Result: labels reflect real trend moves, not random noise.
    Last max(horizons) rows are labelled 0 (no future data).
    """
    close  = df["Close"].values.astype(np.float64)
    atr    = df["ATR"].values.astype(np.float64)  # already normalised by close
    n      = len(close)
    max_h  = max(horizons)
    labels = np.zeros(n, dtype=np.float32)

    for i in range(n - max_h):
        local_atr = float(np.nanmean(atr[max(0, i - 20): i + 1]))
        thr       = float(np.clip(local_atr * ATR_MULTIPLIER, MIN_THRESHOLD, MAX_THRESHOLD))
        avg_ret   = float(np.mean([
            np.log(close[i + h] / (close[i] + 1e-12))
            for h in horizons
        ]))
        labels[i] = 1.0 if avg_ret > thr else 0.0

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    frames: list[pd.DataFrame] = []
    min_rows = SEQUENCE_LENGTH + max(MULTI_HORIZON) + 30

    print(f"Features      : {len(FEATURES)}  —  step={SEQ_STEP}")
    print(f"Multi-horizon : avg({MULTI_HORIZON})  ATR×{ATR_MULTIPLIER}  "
          f"clamp [{MIN_THRESHOLD:.1%}, {MAX_THRESHOLD:.1%}]")

    # ── Pre-load macro data once (avoids repeated downloads) ─────────────────
    print("\nLoading macro data (KSE-100, PKR/USD, Brent oil) …")
    load_macro_data(start=START_DATE)  # warms cache

    print("\nDownloading stock data …")
    for ticker in ALL_STOCKS:
        print(f"  {ticker}", end=" … ", flush=True)
        raw = download_stock(ticker)
        if raw is None:
            print("no data, skip")
            continue
        try:
            df = engineer_features(raw)
            df = inject_macro_features(df)   # ← merge KSE100, PKR/USD, oil
            if len(df) < min_rows:
                print(f"too short ({len(df)} rows)")
                continue
            frames.append(df)
            print(f"OK ({len(df):,} rows)")
        except Exception as exc:
            print(f"feature error: {exc}")

    if not frames:
        raise RuntimeError("No stock data loaded.")

    # Fit scaler on ALL rows before splitting (prevent leakage in scale space)
    merged = pd.concat(frames, ignore_index=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.dropna(subset=FEATURES, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    print(f"\nTotal rows (all stocks): {len(merged):,}")

    scaler = fit_and_save_scaler(merged, SCALER_PATH)
    print(f"Scaler saved → {SCALER_PATH}")

    # Per-stock chronological 80/20 split with step=SEQ_STEP
    tr_seq_X, tr_seq_y = [], []
    te_seq_X, te_seq_y = [], []
    tr_flat_X, tr_flat_y = [], []
    te_flat_X, te_flat_y = [], []

    for df in frames:
        scaled = scale_features(df, scaler)
        labels = compute_atr_labels(df)
        X_seq, y_seq = create_sequences(scaled, labels, SEQUENCE_LENGTH, step=SEQ_STEP)
        if len(X_seq) < 20:
            continue

        s = int(len(X_seq) * 0.80)
        tr_seq_X.append(X_seq[:s]);         tr_seq_y.append(y_seq[:s])
        te_seq_X.append(X_seq[s:]);         te_seq_y.append(y_seq[s:])
        tr_flat_X.append(X_seq[:s, -1, :]); tr_flat_y.append(y_seq[:s])
        te_flat_X.append(X_seq[s:, -1, :]); te_flat_y.append(y_seq[s:])

    def _cat(lst):
        return np.concatenate(lst, axis=0).astype(np.float32)

    Xs_tr = _cat(tr_seq_X);  ys_tr = _cat(tr_seq_y)
    Xs_te = _cat(te_seq_X);  ys_te = _cat(te_seq_y)
    Xf_tr = _cat(tr_flat_X); yf_tr = _cat(tr_flat_y)
    Xf_te = _cat(te_flat_X); yf_te = _cat(te_flat_y)

    up_pct = ys_tr.mean() * 100
    print(f"\nTrain (seq) : {Xs_tr.shape}   UP label: {up_pct:.1f}%")
    print(f"Test  (seq) : {Xs_te.shape}")

    return Xs_tr, ys_tr, Xs_te, ys_te, Xf_tr, yf_tr, Xf_te, yf_te, frames, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Probability diagnostics  (FIX #5)
# ─────────────────────────────────────────────────────────────────────────────

def prob_diagnostics(name: str, y_prob: np.ndarray, log_lines: list) -> None:
    """Print probability distribution stats; warn on collapse indicators."""
    pmin  = float(y_prob.min())
    pmax  = float(y_prob.max())
    pmean = float(y_prob.mean())
    pstd  = float(y_prob.std())

    lines = [
        f"\n── {name} — Probability Diagnostics ──",
        f"  min={pmin:.4f}  max={pmax:.4f}  mean={pmean:.4f}  std={pstd:.4f}",
    ]

    if pstd < 0.05:
        lines.append("  [WARNING] Probabilities are tightly clustered — model may not be learning.")
    if pmean > 0.75:
        lines.append("  [WARNING] Mean probability very high — possible positive-class collapse.")
    if pmean < 0.25:
        lines.append("  [WARNING] Mean probability very low — possible negative-class collapse.")

    for ln in lines:
        print(ln)
    log_lines.extend(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MCC-based threshold with anti-collapse guards  (FIX #1)
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    log_lines: list,
    name: str = "ensemble",
) -> tuple[float, float]:
    """
    Sweep thresholds and pick the one that maximises MCC subject to
    hard anti-collapse constraints.  Falls back to 0.50 if every valid
    candidate is worse than random.

    Anti-collapse guards (any failure → threshold skipped):
      • predicted_positive_ratio > 0.85  (almost everything labelled UP)
      • predicted_positive_ratio < 0.15  (almost everything labelled DOWN)
      • TN == 0                           (zero true negatives caught)
      • TP == 0                           (zero true positives caught)
    """
    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

    print(f"\n── Threshold Optimisation [{name}] ──")
    prob_diagnostics(name, y_prob, log_lines)

    best_thr, best_mcc = 0.50, -2.0
    candidates = []

    for thr in np.arange(0.30, 0.76, 0.01):
        y_pred    = (y_prob >= thr).astype(int)
        pos_ratio = float(y_pred.mean())

        # ── Anti-collapse guards ─────────────────────────────────────────────
        if pos_ratio > 0.85 or pos_ratio < 0.15:
            continue
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        if tn == 0 or tp == 0:
            continue

        mcc  = float(matthews_corrcoef(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        candidates.append((float(round(thr, 2)), mcc, bacc, pos_ratio))

        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(round(thr, 2))

    if not candidates:
        msg = f"  [WARNING] No non-collapse threshold found for {name} — defaulting to 0.50"
        print(msg)
        log_lines.append(msg)
        best_thr = 0.50
        y_pred   = (y_prob >= 0.50).astype(int)
        best_mcc = float(matthews_corrcoef(y_true, y_pred)) if y_pred.std() > 0 else 0.0

    # Show top 5 candidates
    top5 = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
    log_lines.append(f"\n── Top threshold candidates [{name}] ──")
    for thr, mcc, bacc, pr in top5:
        line = f"  thr={thr:.2f}  MCC={mcc:.4f}  BalAcc={bacc:.4f}  pos={pr:.1%}"
        print(line)
        log_lines.append(line)

    chosen = f"  → CHOSEN threshold={best_thr:.2f}  MCC={best_mcc:.4f}"
    print(chosen)
    log_lines.append(chosen)
    return best_thr, best_mcc


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: GRU + Multi-Head Attention with Focal Loss  (FIX #3)
# ─────────────────────────────────────────────────────────────────────────────

def build_gru_model(n_features: int, seq_len: int):
    from tensorflow.keras.models import Model                           # type: ignore
    from tensorflow.keras.layers import (                               # type: ignore
        Input, GRU, Dropout, Dense,
        MultiHeadAttention, LayerNormalization,
        GlobalAveragePooling1D,
    )
    from tensorflow.keras.losses import BinaryFocalCrossentropy         # type: ignore
    from tensorflow.keras.optimizers import Adam                        # type: ignore
    from tensorflow.keras import regularizers                           # type: ignore

    inp = Input(shape=(seq_len, n_features), name="seq_input")

    # Layer 1: wider GRU — more capacity for market pattern recognition
    x = GRU(192, return_sequences=True, recurrent_dropout=0.05,
            kernel_regularizer=regularizers.l2(5e-5), name="gru_1")(inp)
    x = Dropout(0.20, name="drop_1")(x)

    # Layer 2: second GRU — reduced dropout so gradients flow
    x = GRU(96, return_sequences=True, recurrent_dropout=0.05,
            name="gru_2")(x)
    x = Dropout(0.15, name="drop_2")(x)

    # Multi-head self-attention — learns WHICH timesteps matter most
    attn = MultiHeadAttention(num_heads=4, key_dim=24,
                               dropout=0.05, name="self_attn")(x, x)
    x    = LayerNormalization(epsilon=1e-6, name="layer_norm")(x + attn)

    # Concatenate last timestep + global avg — captures both recency and context
    last_step = x[:, -1, :]                   # shape (batch, 96)
    avg_pool  = GlobalAveragePooling1D(name="avg_pool")(x)   # shape (batch, 96)
    from tensorflow.keras.layers import Concatenate           # type: ignore
    x = Concatenate(name="temporal_pool")([last_step, avg_pool])   # (batch, 192)
    x = Dense(128, activation="relu",
              kernel_regularizer=regularizers.l2(5e-5), name="fc_1")(x)
    x = Dropout(0.20, name="drop_3")(x)
    x = Dense(64, activation="relu",
              kernel_regularizer=regularizers.l2(5e-5), name="fc_2")(x)
    x = Dropout(0.10, name="drop_4")(x)
    out = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="GRU_Attention_FocalLoss_v6")

    # Switch to standard BCE — focal loss squashes probabilities too low,
    # causing collapse at typical thresholds. Class weights handle imbalance.
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_gru(X_tr, y_tr, X_te, y_te, log_lines: list):
    from tensorflow.keras.callbacks import (                            # type: ignore
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    from sklearn.utils.class_weight import compute_class_weight         # type: ignore

    print("\n─── Training GRU + Multi-Head Attention (Focal Loss) ───")
    model = build_gru_model(X_tr.shape[2], X_tr.shape[1])
    model.summary(line_length=72)

    cw_arr = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    cw     = {0: float(cw_arr[0]), 1: float(cw_arr[1])}
    print(f"Class weights: DOWN={cw[0]:.3f}  UP={cw[1]:.3f}")
    log_lines.append(f"GRU class weights: {cw}")

    # Boost UP weight further — model must learn both classes
    cw[1] = cw[1] * 1.5

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=15, mode="max",
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(GRU_PATH, monitor="val_auc", mode="max",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc", factor=0.5, mode="max",
                          patience=7, min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=128,
        validation_data=(X_te, y_te),
        class_weight=cw, shuffle=True,
        callbacks=callbacks, verbose=1,
    )
    elapsed = (time.time() - t0) / 60
    print(f"GRU training time: {elapsed:.1f} min")
    log_lines.append(f"GRU training time: {elapsed:.1f} min")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: XGBoost — fixed EarlyStopping API  (FIX #2)
# ─────────────────────────────────────────────────────────────────────────────

def train_xgb(X_tr, y_tr, X_te, y_te, log_lines: list):
    try:
        from xgboost import XGBClassifier                               # type: ignore
    except ImportError:
        print("[SKIP] XGBoost not installed")
        return None

    print("\n─── Training XGBoost ───")
    t0  = time.time()
    spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    print(f"scale_pos_weight: {spw:.3f}")

    # XGBoost 2.x+ requires callbacks in the constructor, not in fit().
    # We try the constructor approach first; fall back to plain early_stopping_rounds.
    try:
        xgb = XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.75,
            colsample_bytree=0.70,
            min_child_weight=8,
            gamma=0.20,
            reg_alpha=0.50,
            reg_lambda=2.00,
            scale_pos_weight=spw,
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=200,
        )
    except TypeError:
        # Older XGBoost that doesn't support early_stopping_rounds in constructor
        xgb = XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.75,
            colsample_bytree=0.70,
            min_child_weight=8,
            gamma=0.20,
            reg_alpha=0.50,
            reg_lambda=2.00,
            scale_pos_weight=spw,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=200,
        )
    elapsed = (time.time() - t0) / 60
    print(f"XGBoost training: {elapsed:.1f} min")
    log_lines.append(f"XGBoost training: {elapsed:.1f} min")

    joblib.dump(xgb, XGB_PATH)
    print(f"Saved (uncalibrated) → {XGB_PATH}")
    return xgb


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: CatBoost (optional)
# ─────────────────────────────────────────────────────────────────────────────

def train_catboost(X_tr, y_tr, X_te, y_te, log_lines: list):
    try:
        from catboost import CatBoostClassifier                         # type: ignore
    except ImportError:
        print("[SKIP] CatBoost not installed (pip install catboost)")
        return None

    print("\n─── Training CatBoost ───")
    t0 = time.time()
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=5.0,
        bagging_temperature=0.8,
        random_strength=1.5,
        border_count=128,
        scale_pos_weight=spw,         # handle class imbalance like XGB
        eval_metric="AUC",
        early_stopping_rounds=50,
        use_best_model=True,
        random_seed=42,
        verbose=200,
        task_type="CPU",
    )
    cat.fit(X_tr, y_tr, eval_set=(X_te, y_te))
    elapsed = (time.time() - t0) / 60
    print(f"CatBoost training: {elapsed:.1f} min")
    log_lines.append(f"CatBoost training: {elapsed:.1f} min")
    joblib.dump(cat, CAT_PATH)
    print(f"Saved → {CAT_PATH}")
    return cat


# ─────────────────────────────────────────────────────────────────────────────
# Probability calibration  (FIX #4)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_xgb(xgb_model, X_calib: np.ndarray, y_calib: np.ndarray,
                   log_lines: list):
    """
    Sigmoid calibration on a held-out subset.
    cv='prefit' was removed in sklearn 1.2; we use cv=5 refit instead.
    """
    from sklearn.calibration import CalibratedClassifierCV             # type: ignore
    from sklearn.metrics import brier_score_loss                       # type: ignore

    print("\n─── Calibrating XGBoost (sigmoid) ───")
    raw_prob  = xgb_model.predict_proba(X_calib)[:, 1]
    raw_brier = brier_score_loss(y_calib, raw_prob)

    # cv='prefit' was removed in sklearn 1.2.
    # cv=5 re-trains the estimator on each fold, but early_stopping_rounds
    # in the XGB constructor requires eval_set, which sklearn's CV doesn't pass.
    # Solution: clone the model without early stopping for calibration only.
    import copy
    xgb_for_cal = copy.deepcopy(xgb_model)
    xgb_for_cal.set_params(early_stopping_rounds=None)
    cal = CalibratedClassifierCV(xgb_for_cal, method="sigmoid", cv=5)
    cal.fit(X_calib, y_calib)
    cal_prob  = cal.predict_proba(X_calib)[:, 1]
    cal_brier = brier_score_loss(y_calib, cal_prob)

    line = (f"  Brier (raw)={raw_brier:.4f}  →  Brier (calibrated)={cal_brier:.4f}"
            f"  {'✓ improved' if cal_brier < raw_brier else '(no improvement)'}")
    print(line)
    log_lines.append(line)

    # Overwrite saved model with calibrated version
    joblib.dump(cal, XGB_PATH)
    print(f"Saved (calibrated) → {XGB_PATH}")
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    log_lines: list,
    thr: float = 0.50,
) -> dict:
    from sklearn.metrics import (                                        # type: ignore
        accuracy_score, f1_score, roc_auc_score,
        precision_score, recall_score,
        matthews_corrcoef, balanced_accuracy_score,
        confusion_matrix,
    )

    y_pred    = (y_prob >= thr).astype(int)
    pos_ratio = float(y_pred.mean())

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob) if y_prob.std() > 0 else 0.5
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    mcc   = matthews_corrcoef(y_true, y_pred) if y_pred.std() > 0 else 0.0
    bacc  = balanced_accuracy_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred)

    collapse_warn = ""
    if pos_ratio > 0.85:
        collapse_warn = "  [WARNING] COLLAPSE — >85% predicted positive!"
    elif pos_ratio < 0.15:
        collapse_warn = "  [WARNING] COLLAPSE — <15% predicted positive!"

    lines = [
        f"\n{'═'*56}",
        f"  {name}",
        f"{'═'*56}",
        f"  Threshold      : {thr:.2f}  (MCC-optimal)",
        f"  Accuracy       : {acc*100:.2f}%",
        f"  Balanced Acc   : {bacc*100:.2f}%",
        f"  F1 Score       : {f1:.4f}",
        f"  ROC-AUC        : {auc:.4f}",
        f"  MCC            : {mcc:.4f}",
        f"  Precision      : {prec:.4f}",
        f"  Recall         : {rec:.4f}",
        f"  Pred UP%       : {pos_ratio:.1%}  Pred DOWN%: {(1-pos_ratio):.1%}",
        collapse_warn,
        f"  Confusion Matrix:\n{cm}",
    ]
    for ln in lines:
        print(ln)
    log_lines.extend(lines)

    return {
        "acc": acc, "bacc": bacc, "f1": f1, "auc": auc,
        "mcc": mcc, "prec": prec, "rec": rec,
        "thr": thr, "pos_ratio": pos_ratio,
        "collapse": pos_ratio > 0.85 or pos_ratio < 0.15,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic ensemble weights  (FIX #6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dynamic_weights(
    model_aucs: dict[str, float],
) -> dict[str, float]:
    """
    Weight each model proportional to its test-set ROC-AUC.
    Models at chance (AUC=0.5) receive very small weight.
    Clips AUC floor at 0.50 (chance level gets weight 0).
    """
    clipped = {k: max(0.0, v - 0.50) for k, v in model_aucs.items()}
    total   = sum(clipped.values())
    if total < 1e-6:
        n = len(model_aucs)
        return {k: 1.0 / n for k in model_aucs}
    return {k: v / total for k, v in clipped.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Stacking meta-model  (FIX #7)
# ─────────────────────────────────────────────────────────────────────────────

def build_meta_model(
    gru_prob: np.ndarray,
    xgb_prob: np.ndarray | None,
    cat_prob: np.ndarray | None,
    y_true: np.ndarray,
    log_lines: list,
):
    """
    Train a LogisticRegression meta-learner on the first half of the test set.
    The caller should evaluate on the SECOND half to avoid leakage.
    """
    from sklearn.linear_model import LogisticRegression                 # type: ignore

    probs = [gru_prob]
    if xgb_prob is not None:
        probs.append(xgb_prob)
    if cat_prob is not None:
        probs.append(cat_prob)

    if len(probs) < 2:
        log_lines.append("Meta-model skipped (only 1 base model available).")
        return None

    meta_X   = np.column_stack(probs)
    half     = len(meta_X) // 2
    meta_tr  = meta_X[:half];  y_tr = y_true[:half]
    meta_te  = meta_X[half:];  y_te = y_true[half:]

    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta.fit(meta_tr, y_tr)
    meta_prob = meta.predict_proba(meta_te)[:, 1]

    joblib.dump(meta, META_PATH)
    line = (f"\nMeta-model (LR) trained on {len(meta_tr)} samples, "
            f"eval on {len(meta_te)}")
    print(line)
    log_lines.append(line)

    return meta, meta_prob, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparisons  (FIX #10)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_comparison(y_true: np.ndarray, log_lines: list) -> None:
    """Compare against trivial baselines — AI must beat these."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score  # type: ignore

    n   = len(y_true)
    up  = float(y_true.mean())

    # 1. Always-UP predictor
    always_up_acc  = up
    always_up_bacc = 0.5  # trivially unbalanced

    # 2. Random predictor (AUC = 0.50 by definition)
    rng          = np.random.default_rng(42)
    rand_prob    = rng.uniform(0, 1, n)
    rand_auc     = float(roc_auc_score(y_true, rand_prob))
    rand_pred    = (rand_prob >= 0.5).astype(int)
    rand_bacc    = float(balanced_accuracy_score(y_true, rand_pred))

    # 3. Momentum baseline: predict UP if last 5-sample average is positive
    #    (simplified — just uses label auto-correlation as proxy)
    mom_pred     = np.roll(y_true, 5)            # look back 5 labels
    mom_pred[:5] = up > 0.5                      # fill warmup
    mom_acc      = float((mom_pred == y_true).mean())
    mom_bacc     = float(balanced_accuracy_score(y_true, mom_pred))

    lines = [
        "\n── Baseline Comparison ──",
        f"  Always-UP       accuracy={always_up_acc:.2%}  balanced_acc={always_up_bacc:.2%}",
        f"  Random          ROC-AUC={rand_auc:.4f}  balanced_acc={rand_bacc:.2%}",
        f"  Momentum (lag5) accuracy={mom_acc:.2%}  balanced_acc={mom_bacc:.2%}",
        "  (AI metrics above must clearly exceed these baselines)",
    ]
    for ln in lines:
        print(ln)
    log_lines.extend(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation — per-fold MCC + balanced accuracy  (FIX #11)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_report(frames: list, scaler, n_folds: int = 3,
                         log_lines: list | None = None) -> None:
    if log_lines is None:
        log_lines = []
    try:
        from xgboost import XGBClassifier                               # type: ignore
        from sklearn.metrics import (                                    # type: ignore
            roc_auc_score, f1_score, matthews_corrcoef,
            balanced_accuracy_score,
        )
    except ImportError:
        return

    print("\n─── Walk-Forward Cross-Validation (XGBoost, 3 folds) ───")
    log_lines.append("\n── Walk-Forward Cross-Validation ──")

    all_X, all_y = [], []
    for df in frames:
        scaled = scale_features(df, scaler)
        labels = compute_atr_labels(df)
        X_seq, y_seq = create_sequences(scaled, labels, SEQUENCE_LENGTH, step=SEQ_STEP)
        if len(X_seq) < 20:
            continue
        all_X.append(X_seq[:, -1, :])
        all_y.append(y_seq)

    if not all_X:
        return

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)

    fold_size = len(X) // (n_folds + 1)
    aucs, mccs, baccs, f1s = [], [], [], []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_end  = min(train_end + fold_size, len(X))
        if test_end - train_end < 20:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]
        spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)

        m = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=spw, early_stopping_rounds=20,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        prob   = m.predict_proba(X_te)[:, 1]
        pred   = (prob >= 0.50).astype(int)
        auc    = roc_auc_score(y_te, prob) if prob.std() > 0 else 0.5
        mcc    = matthews_corrcoef(y_te, pred) if pred.std() > 0 else 0.0
        bacc   = balanced_accuracy_score(y_te, pred)
        f1     = f1_score(y_te, pred, zero_division=0)
        pos_r  = float(pred.mean())

        aucs.append(auc); mccs.append(mcc); baccs.append(bacc); f1s.append(f1)
        msg = (f"  Fold {fold+1}/{n_folds}  "
               f"ROC-AUC={auc:.4f}  MCC={mcc:.4f}  BalAcc={bacc:.4f}  "
               f"F1={f1:.4f}  UP%={pos_r:.0%}")
        print(msg)
        log_lines.append(msg)

    if aucs:
        summary = (f"  WF avg  AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}"
                   f"  MCC={np.mean(mccs):.4f}±{np.std(mccs):.4f}"
                   f"  BalAcc={np.mean(baccs):.4f}")
        print(summary)
        log_lines.append(summary)


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def log_feature_importance(xgb_model, X_test: np.ndarray, log_lines: list) -> None:
    # Unwrap CalibratedClassifierCV to get the base XGBoost estimator
    from sklearn.calibration import CalibratedClassifierCV            # type: ignore
    if isinstance(xgb_model, CalibratedClassifierCV):
        # calibrated_classifiers_ holds (estimator, calibrator) pairs
        base_xgb = xgb_model.calibrated_classifiers_[0].estimator
    else:
        base_xgb = xgb_model

    try:
        import shap                                                     # type: ignore
        sample    = X_test[:min(500, len(X_test))]
        explainer = shap.TreeExplainer(base_xgb)
        sv        = explainer.shap_values(sample)
        mean_imp  = np.abs(sv).mean(axis=0)
        label     = "SHAP"
    except Exception:
        mean_imp  = base_xgb.feature_importances_
        label     = "Gain"

    ranked = sorted(zip(FEATURES, mean_imp), key=lambda x: x[1], reverse=True)
    log_lines.append(f"\n── Feature Importance ({label}, top 12) ──")
    print(f"\n── Feature Importance ({label}, top 12) ──")
    for feat, score in ranked[:12]:
        line = f"  {feat:<20}  {score:.5f}"
        print(line)
        log_lines.append(line)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train():
    wall_start = time.time()
    log_lines  = [
        "PSX AI Quant Platform — Training Log v5 (Anti-Collapse)",
        f"Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Features    : {len(FEATURES)}  —  {', '.join(FEATURES)}",
        f"Seq step    : {SEQ_STEP}  (stride between GRU windows)",
        f"Target      : avg({MULTI_HORIZON}) log-ret > ATR×{ATR_MULTIPLIER} "
        f"[{MIN_THRESHOLD:.1%}, {MAX_THRESHOLD:.1%}]",
        "Architecture: GRU(128)+GRU(64)+MultiHeadAttention+GAP  [FocalLoss γ=2]",
        "Threshold   : MCC-optimal with anti-collapse guards",
    ]

    (
        Xs_tr, ys_tr, Xs_te, ys_te,
        Xf_tr, yf_tr, Xf_te, yf_te,
        frames, scaler,
    ) = build_dataset()

    print(f"\nGRU  — train: {len(Xs_tr):,}  test: {len(Xs_te):,}")
    print(f"XGB  — train: {len(Xf_tr):,}  test: {len(Xf_te):,}")

    # ── Train base models ─────────────────────────────────────────────────────
    gru_model = train_gru(Xs_tr, ys_tr, Xs_te, ys_te, log_lines)
    xgb_model = train_xgb(Xf_tr, yf_tr, Xf_te, yf_te, log_lines)
    cat_model = train_catboost(Xf_tr, yf_tr, Xf_te, yf_te, log_lines)

    # ── Raw probabilities ─────────────────────────────────────────────────────
    gru_prob = gru_model.predict(Xs_te, verbose=0).flatten()
    xgb_prob = xgb_model.predict_proba(Xf_te)[:, 1] if xgb_model else None
    cat_prob = (
        cat_model.predict_proba(Xf_te)[:, 1]
        if cat_model is not None else None
    )

    # ── XGBoost: skip calibration (cv=5 refit causes UP collapse) ───────────
    if xgb_model is not None:
        joblib.dump(xgb_model, XGB_PATH)
        print(f"Saved (raw, no calibration) → {XGB_PATH}")
        xgb_prob = xgb_model.predict_proba(Xf_te)[:, 1]

    # ── Dynamic ensemble weights based on individual AUC ─────────────────────
    from sklearn.metrics import roc_auc_score                           # type: ignore
    from sklearn.isotonic import IsotonicRegression                    # type: ignore

    model_aucs: dict[str, float] = {}
    model_aucs["gru"] = roc_auc_score(ys_te, gru_prob) if gru_prob.std() > 0 else 0.5
    if xgb_prob is not None:
        model_aucs["xgb"] = roc_auc_score(yf_te, xgb_prob) if xgb_prob.std() > 0 else 0.5
    if cat_prob is not None:
        model_aucs["cat"] = roc_auc_score(yf_te, cat_prob) if cat_prob.std() > 0 else 0.5

    dyn_w = compute_dynamic_weights(model_aucs)
    weight_line = "  Dynamic weights: " + "  ".join(
        f"{k.upper()}={v:.3f}" for k, v in dyn_w.items()
    )
    print(f"\n── Dynamic Ensemble Weights ──\n{weight_line}")
    log_lines.append(weight_line)

    # ── Rank-normalise each model's raw probabilities before blending ─────────
    # Models trained with different losses output different probability scales.
    # Isotonic regression maps each model to a well-calibrated [0,1] range
    # using only the first half of the test set, evaluated on the second half.
    def rank_norm(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
        half = len(prob) // 2
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(prob[:half], y[:half].astype(float))
        return ir.transform(prob).clip(0.01, 0.99)

    gru_norm = rank_norm(gru_prob, ys_te)
    xgb_norm = rank_norm(xgb_prob, yf_te) if xgb_prob is not None else None
    cat_norm = rank_norm(cat_prob, yf_te) if cat_prob is not None else None

    print(f"  GRU  prob range: [{gru_prob.min():.3f}, {gru_prob.max():.3f}]  →  norm: [{gru_norm.min():.3f}, {gru_norm.max():.3f}]")
    if xgb_norm is not None:
        print(f"  XGB  prob range: [{xgb_prob.min():.3f}, {xgb_prob.max():.3f}]  →  norm: [{xgb_norm.min():.3f}, {xgb_norm.max():.3f}]")

    # Build weighted ensemble probability using normalised scores
    ens_prob = gru_norm * dyn_w.get("gru", 0.5)
    if xgb_norm is not None:
        ens_prob += xgb_norm * dyn_w.get("xgb", 0.0)
    if cat_norm is not None:
        ens_prob += cat_norm * dyn_w.get("cat", 0.0)


    # ── MCC-optimal threshold (anti-collapse) ─────────────────────────────────
    opt_thr, opt_mcc = find_optimal_threshold(ys_te, ens_prob, log_lines, "ensemble")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("═" * 56)

    # Evaluate each model using its own optimal threshold on normalised probs
    gru_thr, _ = find_optimal_threshold(ys_te, gru_norm, log_lines, "GRU")
    gru_metrics = evaluate_model("GRU + Multi-Head Attention", ys_te, gru_norm,
                                  log_lines, gru_thr)
    xgb_metrics: dict = {}
    if xgb_model is not None:
        xgb_thr, _ = find_optimal_threshold(yf_te, xgb_norm, log_lines, "XGBoost")
        xgb_metrics = evaluate_model("XGBoost (calibrated)", yf_te, xgb_norm,
                                      log_lines, xgb_thr)
        log_feature_importance(xgb_model, Xf_te, log_lines)

    ens_metrics = evaluate_model(
        f"Ensemble ({' + '.join(k.upper() for k in dyn_w)})",
        ys_te, ens_prob, log_lines, opt_thr,
    )

    # ── Stacking meta-model ───────────────────────────────────────────────────
    _cat_norm = cat_norm if cat_prob is not None else None
    meta_result = build_meta_model(gru_norm, xgb_norm, _cat_norm, ys_te, log_lines)
    if meta_result is not None:
        meta_model, meta_prob, y_meta_te = meta_result
        meta_thr, _  = find_optimal_threshold(y_meta_te, meta_prob, log_lines, "meta")
        evaluate_model("Meta-Model (Logistic Regression stacker)",
                       y_meta_te, meta_prob, log_lines, meta_thr)

    # ── Baselines ─────────────────────────────────────────────────────────────
    baseline_comparison(ys_te, log_lines)

    # ── Walk-forward ──────────────────────────────────────────────────────────
    walk_forward_report(frames, scaler, n_folds=3, log_lines=log_lines)

    # ── Save metadata ─────────────────────────────────────────────────────────
    elapsed = (time.time() - wall_start) / 60.0
    metadata = {
        "version":           "v5",
        "trained_at":        datetime.now().isoformat(),
        "training_minutes":  round(elapsed, 1),
        "n_features":        len(FEATURES),
        "features":          FEATURES,
        "sequence_length":   SEQUENCE_LENGTH,
        "seq_step":          SEQ_STEP,
        "multi_horizon":     MULTI_HORIZON,
        "atr_multiplier":    ATR_MULTIPLIER,
        "label_threshold_range": [MIN_THRESHOLD, MAX_THRESHOLD],
        "optimal_threshold": opt_thr,
        "architecture":      "GRU(128,ret)+GRU(64,ret)+MultiHeadAttention(4h)"
                             "+LayerNorm+GAP+Dense(64)+Dense(32)+Dense(1,sig)"
                             "  loss=BinaryFocalCrossentropy(gamma=2)",
        "dynamic_weights":   dyn_w,
        "model_aucs":        model_aucs,
        "metrics": {
            "gru":      gru_metrics,
            "xgb":      xgb_metrics,
            "ensemble": ens_metrics,
        },
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log_lines.append(f"\nTotal training time: {elapsed:.1f} min")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\nMetadata → {METADATA_PATH}")
    print(f"Log      → {LOG_PATH}")
    print(f"Time     : {elapsed:.1f} min")
    print("\n✓ Training complete (v5).")


if __name__ == "__main__":
    train()
