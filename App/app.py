# -*- coding: utf-8 -*-
"""
Rasd (رصد) — classify customer reviews (Saudi Arabic) by intent and sentiment.

Runs locally: reviews are processed on this machine by the two fine-tuned MARBERT models.

    python app.py            # then open http://127.0.0.1:8000

Environment variables (all optional):
    RASD_MODELS_DIR         folder that contains saudi_marbert_intent/ and saudi_marbert_sentiment/  (default: ./models)
    RASD_DEMO=1             start WITHOUT the models, using a trivial keyword classifier (UI preview only!)
    RASD_REVIEW_THRESHOLD   confidence below which a review is flagged for a human (default 0.60)
    RASD_MAX_LEN            max tokens per review at inference time (default 64)
    RASD_MAX_ROWS           max reviews analysed per request (default 3000)
    RASD_DEVICE             force "cpu", "cuda" or "mps" (default: cuda if available, else cpu)
    RASD_HOST / RASD_PORT   where to listen (default 127.0.0.1 / 8000)
"""
import csv
import io
import os
import re
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("RASD_MODELS_DIR", BASE_DIR / "models"))
INTENT_MODEL = "LeenAlsahli/saudi-marbert-intent"
SENTIMENT_MODEL = "LeenAlsahli/saudi-marbert-sentiment"
DEMO = os.environ.get("RASD_DEMO", "0") == "1"
REVIEW_THRESHOLD = float(os.environ.get("RASD_REVIEW_THRESHOLD", "0.60"))
INFER_MAX_LEN = int(os.environ.get("RASD_MAX_LEN", "64"))
MAX_ROWS = int(os.environ.get("RASD_MAX_ROWS", "3000"))
MAX_UPLOAD_MB = 10
MAX_TEXT_CHARS = 2000
BATCH_SIZE = 32
SENTIMENTS = ("Positive", "Neutral", "Negative")

# ---------------------------------------------------------------------------
# Text cleaning — MUST match the cleaning used to build the `text_clean` column
# the models were trained on.
# ---------------------------------------------------------------------------
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def clean_text(t) -> str:
    t = str(t)
    t = _DIACRITICS.sub("", t)                 # diacritics + tatweel
    t = re.sub(r"[أإآ]", "ا", t)               # normalise alef forms
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)       # 3+ repeated chars -> 2
    t = re.sub(r"['\"“”‘’]", "", t)            # quote marks
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _result(intent, ci, sentiment, cs):
    return {
        "intent": intent,
        "intent_confidence": round(float(ci), 3),
        "sentiment": sentiment,
        "sentiment_confidence": round(float(cs), 3),
        "needs_review": bool(min(ci, cs) < REVIEW_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
class TransformerClassifier:
    """The real thing: two fine-tuned MARBERT models."""

    is_demo = False
    name = "MARBERT"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(
            os.environ.get("RASD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tok = AutoTokenizer.from_pretrained(INTENT_MODEL)
        self.models, self.labels = {}, {}
        for task, model_name in (
            ("intent", INTENT_MODEL),
            ("sentiment", SENTIMENT_MODEL)
        ):
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
            
            self.models[task] = model
            self.labels[task] = {int(k): v for k, v in model.config.id2label.items()}

    def _probs(self, task, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=INFER_MAX_LEN,
                       return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.models[task](**enc).logits
        return self.torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, texts: List[str]) -> List[dict]:
        cleaned = [clean_text(t) for t in texts]
        out = [None] * len(texts)
        idx = [i for i, c in enumerate(cleaned) if c]
        for s in range(0, len(idx), BATCH_SIZE):
            chunk = idx[s:s + BATCH_SIZE]
            batch = [cleaned[i] for i in chunk]
            p_int, p_sen = self._probs("intent", batch), self._probs("sentiment", batch)
            for row, i in enumerate(chunk):
                ki, ks = int(p_int[row].argmax()), int(p_sen[row].argmax())
                out[i] = _result(self.labels["intent"][ki], p_int[row][ki],
                                 self.labels["sentiment"][ks], p_sen[row][ks])
        return out


class DemoClassifier:
    """UI PREVIEW ONLY — keyword rules, NOT the trained model. Never use for real analysis."""

    is_demo = True
    name = "demo"

    _PAY = ("دفع", "خصم", "انخصم", "فيزا", "بطاقة", "بطاقتي", "مدى", "استرجاع المبلغ", "رسوم", "الفاتورة")
    _TECH = ("التطبيق", "الموقع", "يهنق", "يعلق", "يقفل", "خطا", "خطأ", "تسجيل", "تحديث", "ما يفتح")
    _NEG = ("تاخر", "تأخر", "سيء", "اسوا", "أسوأ", "مكسور", "خربان", "ما وصل", "ما رد", "رديء", "مشكلة")
    _POS = ("ممتاز", "رائع", "شكرا", "شكراً", "حلو", "سريع", "تسلمون", "ابدعتوا", "راقي", "تحسن")
    _REQ = ("ابغى", "ابي", "أبغى", "أبي", "اريد")
    _SUG = ("اقترح", "أقترح", "ياليت", "ليت", "يا ليت")
    _Q = ("؟", "?", "كيف", "هل", "وش", "متى", "كم ")

    def predict(self, texts: List[str]) -> List[dict]:
        out = []
        for raw in texts:
            t = clean_text(raw)
            if not t:
                out.append(None)
                continue
            has = lambda words: any(w in t for w in words)
            is_question = has(self._Q)
            if has(self._PAY) and (has(self._NEG) or "مرتين" in t or "ما " in t or "رفض" in t):
                out.append(_result("Payment Problem", 0.82, "Negative", 0.80))
            elif has(self._TECH) and has(self._NEG + ("ما ", "كل ")):
                out.append(_result("Technical Issue", 0.78, "Negative", 0.79))
            elif is_question:
                out.append(_result("Inquiry", 0.75, "Neutral", 0.74))
            elif has(self._POS) and not has(self._NEG):
                out.append(_result("Positive Feedback", 0.86, "Positive", 0.88))
            elif has(self._NEG):
                out.append(_result("Complaint", 0.84, "Negative", 0.83))
            elif has(self._SUG):
                out.append(_result("Suggestion", 0.74, "Neutral", 0.70))
            elif has(self._REQ):
                out.append(_result("Request", 0.72, "Neutral", 0.71))
            else:
                out.append(_result("Inquiry", 0.42, "Neutral", 0.45))
        return out


CLASSIFIER = None


def missing_models_message() -> str:
    return (
        "\n[رصد] تعذر تحميل الموديلات من Hugging Face.\n"
        "تأكد من الاتصال بالإنترنت وصحة أسماء الموديلات.\n"
        f"Intent model: {INTENT_MODEL}\n"
        f"Sentiment model: {SENTIMENT_MODEL}\n"
    )

def load_classifier():
    if DEMO:
        return DemoClassifier()

    return TransformerClassifier()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CLASSIFIER
    CLASSIFIER = load_classifier()
    yield


app = FastAPI(title="Rasd", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Table reading helpers
# ---------------------------------------------------------------------------
TEXT_HINTS = ("text", "review", "comment", "message", "feedback", "opinion",
              "نص", "تعليق", "رأي", "رأى", "تقييم", "رسالة", "ملاحظ", "محتوى")
DATE_HINTS = ("date", "time", "created", "timestamp", "تاريخ", "وقت")


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(data), dtype=str).fillna("")
        except ImportError:
            raise HTTPException(400, "لقراءة ملفات Excel ثبّت المكتبة: pip install openpyxl")
        except Exception:
            raise HTTPException(400, "تعذّرت قراءة ملف Excel. جرّب حفظه بصيغة CSV.")
    for enc in ("utf-8-sig", "cp1256"):
        try:
            try:
                df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc,
                                 dtype=str, keep_default_na=False)
            except (pd.errors.ParserError, csv.Error):
                df = pd.read_csv(io.BytesIO(data), encoding=enc, dtype=str, keep_default_na=False)
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise HTTPException(400, "الملف فارغ.")
        except Exception:
            raise HTTPException(400, "تعذّرت قراءة الملف. تأكد أنه CSV أو Excel سليم.")
    raise HTTPException(400, "ترميز الملف غير مدعوم. احفظه بترميز UTF-8.")


def parse_dates(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pd.to_datetime(series, errors="coerce")
        b = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return a if a.notna().sum() >= b.notna().sum() else b


def suggest_text_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if any(h in str(c).lower() for h in TEXT_HINTS):
            return c
    if len(df.columns) == 0:
        return None
    lengths = {c: df[c].astype(str).str.len().mean() for c in df.columns}
    return max(lengths, key=lengths.get)


def suggest_date_column(df: pd.DataFrame, text_col: Optional[str]) -> Optional[str]:
    for c in df.columns:
        if c != text_col and any(h in str(c).lower() for h in DATE_HINTS):
            return c
    for c in df.columns:
        if c == text_col:
            continue
        sample = df[c].astype(str).head(200)
        if sample.str.contains(r"[-/]", regex=True).mean() > 0.8:
            if parse_dates(sample).notna().mean() >= 0.8:
                return c
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _net(g: pd.DataFrame) -> float:
    n = len(g)
    if n == 0:
        return 0.0
    return round(((g["sentiment"] == "Positive").sum() - (g["sentiment"] == "Negative").sum()) / n * 100, 1)


def build_trend(df: pd.DataFrame):
    d = df.dropna(subset=["date_dt"]).sort_values("date_dt")
    if len(d) < 10:
        return None
    span = (d["date_dt"].max() - d["date_dt"].min()).days
    if span <= 21:
        freq, gran = "D", "day"
    elif span <= 200:
        freq, gran = "W-SAT", "week"      # weeks start on Sunday
    else:
        freq, gran = "M", "month"
    d = d.assign(period=d["date_dt"].dt.to_period(freq))
    points = []
    for period, g in d.groupby("period"):
        n = len(g)
        points.append({
            "period": period.start_time.strftime("%Y-%m-%d"),
            "n": int(n),
            "net": _net(g),
            "neg_share": round((g["sentiment"] == "Negative").sum() / n * 100, 1),
            "pos_share": round((g["sentiment"] == "Positive").sum() / n * 100, 1),
        })
    half = len(d) // 2
    first, second = d.iloc[:half], d.iloc[half:]
    verdict = {"status": "insufficient"}
    if min(len(first), len(second)) >= 10:
        delta = round(_net(second) - _net(first), 1)
        status = "improved" if delta >= 5 else "declined" if delta <= -5 else "stable"
        verdict = {
            "status": status,
            "delta": delta,
            "first": {"n": int(len(first)), "net": _net(first),
                      "from": first["date_dt"].min().strftime("%Y-%m-%d"),
                      "to": first["date_dt"].max().strftime("%Y-%m-%d")},
            "second": {"n": int(len(second)), "net": _net(second),
                       "from": second["date_dt"].min().strftime("%Y-%m-%d"),
                       "to": second["date_dt"].max().strftime("%Y-%m-%d")},
        }
    return {"granularity": gran, "points": points, "verdict": verdict}


def build_report(texts: List[str], dates: Optional[pd.Series] = None, truncated: bool = False) -> dict:
    texts = [str(t)[:MAX_TEXT_CHARS] for t in texts]
    preds = CLASSIFIER.predict(texts)
    rows = []
    for i, (t, p) in enumerate(zip(texts, preds)):
        if p is None:
            continue
        dt = dates.iloc[i] if dates is not None else pd.NaT
        rows.append({"text": t, **p, "date_dt": dt})
    if not rows:
        raise HTTPException(400, "لم أجد أي نص صالح للتحليل.")
    df = pd.DataFrame(rows)
    total = len(df)
    sent_counts = {s: int((df["sentiment"] == s).sum()) for s in SENTIMENTS}
    intent_counts = {k: int(v) for k, v in df["intent"].value_counts().items()}
    index = round((sent_counts["Positive"] - sent_counts["Negative"]) / total * 100, 1)
    dated = int(df["date_dt"].notna().sum()) if dates is not None else 0
    trend = build_trend(df) if dates is not None else None

    out_rows = df.drop(columns=["date_dt"]).to_dict("records")
    for r, dt in zip(out_rows, df["date_dt"]):
        r["date"] = dt.strftime("%Y-%m-%d") if pd.notna(dt) else None
    return {
        "total": total,
        "sentiment": sent_counts,
        "intent": intent_counts,
        "satisfaction_index": index,
        "needs_review": int(df["needs_review"].sum()),
        "dated_rows": dated,
        "trend": trend,
        "truncated": truncated,
        "skipped_empty": len(texts) - total,
        "demo": CLASSIFIER.is_demo,
        "rows": out_rows,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class TextsIn(BaseModel):
    texts: List[str]


def _limit(texts: List[str]):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise HTTPException(400, "اكتب رأيًا واحدًا على الأقل.")
    truncated = len(texts) > MAX_ROWS
    return texts[:MAX_ROWS], truncated


async def _read_upload(file: UploadFile) -> pd.DataFrame:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"حجم الملف أكبر من {MAX_UPLOAD_MB} ميغابايت.")
    df = read_table(data, file.filename)
    if df.empty or len(df.columns) == 0:
        raise HTTPException(400, "الملف لا يحتوي على بيانات.")
    return df


@app.get("/api/status")
def status():
    return {"demo": CLASSIFIER.is_demo, "model": CLASSIFIER.name,
            "review_threshold": REVIEW_THRESHOLD, "max_rows": MAX_ROWS}


@app.post("/api/predict")
def predict(body: TextsIn):
    texts, _ = _limit(body.texts)
    texts = [t[:MAX_TEXT_CHARS] for t in texts][:200]
    return {"results": [r for r in CLASSIFIER.predict(texts)], "demo": CLASSIFIER.is_demo}


@app.post("/api/analyze-text")
def analyze_text(body: TextsIn):
    texts, truncated = _limit(body.texts)
    return build_report(texts, None, truncated)


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)):
    df = await _read_upload(file)
    text_col = suggest_text_column(df)
    date_col = suggest_date_column(df, text_col)
    return {
        "filename": file.filename,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "text_column": None if text_col is None else str(text_col),
        "date_column": None if date_col is None else str(date_col),
        "max_rows": MAX_ROWS,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), text_column: str = Form(...), date_column: str = Form("")):
    df = await _read_upload(file)
    if text_column not in df.columns:
        raise HTTPException(400, "عمود النص غير موجود في الملف.")
    if date_column and date_column not in df.columns:
        raise HTTPException(400, "عمود التاريخ غير موجود في الملف.")
    df = df[df[text_column].astype(str).str.strip() != ""]
    if df.empty:
        raise HTTPException(400, "عمود النص فارغ.")
    truncated = len(df) > MAX_ROWS
    df = df.head(MAX_ROWS).reset_index(drop=True)
    dates = parse_dates(df[date_column]) if date_column else None
    return build_report(df[text_column].astype(str).tolist(), dates, truncated)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("RASD_HOST", "127.0.0.1")
    port = int(os.environ.get("RASD_PORT", "8000"))
    print(f"\n[رصد] افتح المتصفح على: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)
