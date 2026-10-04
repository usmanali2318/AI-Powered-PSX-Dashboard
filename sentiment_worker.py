"""
sentiment_worker.py - scores headlines with FinBERT in its own process.

Why a separate process: TensorFlow (GRU model) and PyTorch (FinBERT) loaded into the
same Python process can crash it with a segmentation fault, which Streamlit cannot
catch. Running torch here keeps a crash isolated; app.py falls back to keyword
scoring if this script fails or times out.

Usage: echo '["headline one", "headline two"]' | python sentiment_worker.py
Prints a JSON list of {"label": "POSITIVE|NEGATIVE|NEUTRAL", "score": float}.
"""
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main() -> None:
    headlines = json.loads(sys.stdin.read() or "[]")
    if not headlines:
        print("[]")
        return
    from transformers import pipeline  # imported here so torch never loads in the app process
    nlp = pipeline("text-classification", model="ProsusAI/finbert",
                   truncation=True, max_length=128, framework="pt")
    out = []
    for r in nlp(headlines):
        label = str(r["label"]).upper()
        if label not in ("POSITIVE", "NEGATIVE"):
            label = "NEUTRAL"
        out.append({"label": label, "score": float(r["score"])})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
