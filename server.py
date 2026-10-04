import os, warnings, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
import joblib, numpy as np, pandas as pd, yfinance as yf, feedparser
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from utils import (engineer_features, load_scaler, ensemble_predict, quick_screen, compute_cagr,
                   compute_max_drawdown, compute_cumulative_return, compute_annual_volatility,
                   dynamic_compound_forecast)

warnings.filterwarnings("ignore")
B = Path(__file__).parent
PSX = {"FFC": ("Fauji Fertilizer", "Fertilizer"), "ENGRO": ("Engro Corp", "Fertilizer"), "EFERT": ("Engro Fertilizers", "Fertilizer"),
       "MEBL": ("Meezan Bank", "Banking"), "HBL": ("Habib Bank", "Banking"), "UBL": ("United Bank", "Banking"), "MCB": ("MCB Bank", "Banking"),
       "OGDC": ("Oil & Gas Dev Corp", "Oil & Gas"), "PPL": ("Pakistan Petroleum", "Oil & Gas"), "PSO": ("Pakistan State Oil", "Oil & Gas"),
       "SYS": ("Systems Ltd", "Technology"), "TRG": ("TRG Pakistan", "Technology"), "NETSOL": ("NetSol Technologies", "Technology"),
       "SAZEW": ("Pak Suzuki", "Auto"), "ATLH": ("Atlas Honda", "Auto"), "HCAR": ("Honda Atlas Cars", "Auto"), "INDU": ("Indus Motor", "Auto"),
       "LUCK": ("Lucky Cement", "Cement"), "DGKC": ("D.G. Khan Cement", "Cement"), "HUBC": ("Hub Power", "Power"), "KEL": ("K-Electric", "Power"),
       "PTC": ("Pakistan Telecom", "Telecom")}
INTL = {"AAPL": ("Apple", "International"), "MSFT": ("Microsoft", "International"), "GOOGL": ("Alphabet", "International"),
        "NVDA": ("NVIDIA", "International"), "TSLA": ("Tesla", "International")}
ST = {**PSX, **INTL}
POS = "surge gain rise rally profit growth record beat upgrade jump strong dividend".split()
NEG = "fall drop loss decline plunge crash cut downgrade weak default probe fraud slump".split()
_c, M = {}, {}


def cached(k, ttl, fn):
    v = _c.get(k)
    if v and time() - v[0] < ttl:
        return v[1]
    r = fn()
    _c[k] = (time(), r)
    return r


def load(code):
    def f():
        d = yf.download(code + ".KA" if code in PSX else code, start="2015-01-01", progress=False, auto_adjust=True)
        if d is None or d.empty:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return engineer_features(d[["Open", "High", "Low", "Close", "Volume"]].dropna())
    return cached("df" + code, 300, f)


def models():
    if not M:
        g = x = s = None
        try: s = load_scaler(str(B / "data/scaler.pkl"))
        except Exception: pass
        try: x = joblib.load(B / "models/xgb_model.pkl")
        except Exception: pass
        if (B / "models/gru_model.keras").exists():
            try:
                from tensorflow.keras.models import load_model
                g = load_model(B / "models/gru_model.keras")
            except Exception: pass
        M.update(g=g, x=x, s=s)
    return M


def signal(conf, rsi, df):
    e20, e50, m = (float(df[k].iloc[-1]) for k in ("EMA20", "EMA50", "MACD"))
    s = 2 if conf >= .65 else 1 if conf >= .55 else -2 if conf <= .35 else -1 if conf <= .45 else 0
    s += 2 if rsi <= 30 else 1 if rsi <= 45 else -2 if rsi >= 70 else -1 if rsi >= 55 else 0
    s += (1 if e20 > e50 else -1) + (1 if m > 0 else -1)
    return "STRONG BUY" if s >= 5 else "BUY" if s >= 2 else "HOLD" if s >= -1 else "SELL" if s >= -4 else "STRONG SELL"


def news(code):
    def f():
        q = urllib.parse.quote(f"{ST[code][0]} {code} stock")
        out = []
        for e in feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en").entries[:10]:
            w = e.title.lower()
            sc = sum(x in w for x in POS) - sum(x in w for x in NEG)
            out.append({"title": e.title, "link": e.link, "tone": "pos" if sc > 0 else "neg" if sc < 0 else "neu"})
        return out
    return cached("n" + code, 600, f)


app = FastAPI(title="PSX AI")


@app.get("/api/health")
def health():
    m = models()
    return {"ok": True, "scaler": m["s"] is not None, "xgb": m["x"] is not None, "gru": m["g"] is not None}


@app.get("/api/stocks")
def stocks():
    return [{"code": c, "name": n, "sector": s} for c, (n, s) in ST.items()]


@app.get("/api/news/{code}")
def news_ep(code: str):
    if code.upper() not in ST:
        raise HTTPException(404)
    return news(code.upper())


@app.get("/api/stock/{code}")
def stock(code: str):
    code = code.upper()
    if code not in ST:
        raise HTTPException(404, "Unknown ticker")
    df = load(code)
    if df is None:
        raise HTTPException(502, "No market data")
    m = models()
    ready = m["s"] is not None and (m["g"] is not None or m["x"] is not None)
    e = ensemble_predict(df, m["g"], m["x"], m["s"]) if ready else {"confidence": .5, "dl": .5, "xgb": .5}
    p, pv = float(df.Close.iloc[-1]), float(df.Close.iloc[-2])
    rsi, vol = float(df.RSI.iloc[-1]), compute_annual_volatility(df) * 100
    cagr, mdd = compute_cagr(df), compute_max_drawdown(df)
    trend = float(np.clip((df.EMA20.iloc[-1] / df.EMA50.iloc[-1] - 1) * 5 + df.LogReturn.mean() * 50, -1, 1))
    regime = float(np.clip(df.CumLogRet30.iloc[-1], -.15, .15))
    hl = news(code)
    sent = .5 + (sum(h["tone"] == "pos" for h in hl) / len(hl) - .5) * .6 if hl else .5
    fc = dynamic_compound_forecast(p, 5, cagr, e["confidence"], trend, vol / 100, sent, regime, "monthly", 200)
    lr = df.LogReturn.dropna()
    t = df.tail(1260)
    L = lambda s: [None if pd.isna(x) else round(float(x), 3) for x in s]
    return {
        "code": code, "name": ST[code][0], "sector": ST[code][1], "price": p, "chg": (p / pv - 1) * 100,
        "signal": signal(e["confidence"], rsi, df), "conf": e["confidence"], "gru": e["dl"], "xgb": e["xgb"], "ready": ready,
        "vol": vol, "risk": "Low" if vol < 20 else "Medium" if vol < 40 else "High", "cagr": cagr * 100, "mdd": mdd * 100,
        "cum": compute_cumulative_return(df) * 100, "rsi": rsi, "mom": float(df.Momentum.iloc[-1]) * 100,
        "bull": bool(df.EMA20.iloc[-1] > df.EMA50.iloc[-1]), "sent": sent, "var95": abs(float(np.percentile(lr, 5))) * p,
        "calmar": cagr / max(abs(mdd), .01),
        "fc": {"path": L(fc["path"]), "up": L(fc["upper"]), "lo": L(fc["lower"]), "f1": fc["f1y"], "f3": fc["f3y"], "f5": fc["f5y"]},
        "s": {"d": t.Date.dt.strftime("%Y-%m-%d").tolist(), "o": L(t.Open), "h": L(t.High), "l": L(t.Low), "c": L(t.Close),
              "v": L(t.Volume), "e20": L(t.EMA20), "e50": L(t.EMA50), "rsi": L(t.RSI), "macd": L(t.MACD * 100), "msig": L(t.MACD_Signal * 100)},
    }


@app.get("/api/screen")
def screen():
    def f():
        m = models()
        def one(c):
            d = load(c)
            r = quick_screen(d, m["g"], m["x"], m["s"]) if d is not None else None
            return {**r, "code": c, "name": ST[c][0], "sector": ST[c][1],
                    "chg": float(d.Close.iloc[-1] / d.Close.iloc[-2] - 1) * 100} if r else None
        with ThreadPoolExecutor(6) as ex:
            return [r for r in ex.map(one, ST) if r]
    return sorted(cached("scr", 600, f), key=lambda r: -r["score"])


app.mount("/", StaticFiles(directory=B / "static", html=True), name="static")
