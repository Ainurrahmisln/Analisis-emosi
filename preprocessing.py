# Preprocessing
import re
import io
import unicodedata
import pandas as pd
import streamlit as st
from config import DEFAULT_KAMUS_NORMALISASI

# =========================
# PREPROCESSING
# =========================
ZWSP_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
URL_RE  = re.compile(r"http[s]?://\S+|www\.\S+")
TAG_RE  = re.compile(r"<[^>]+>")
WS_RE   = re.compile(r"\s+")

def cleaning_text(teks: str) -> str:
    teks = "" if pd.isna(teks) else str(teks)
    teks = unicodedata.normalize("NFKC", teks)
    teks = ZWSP_RE.sub("", teks)
    teks = TAG_RE.sub(" ", teks)
    teks = URL_RE.sub(" ", teks)
    teks = WS_RE.sub(" ", teks).strip()
    return teks

def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())

def build_normalizer(kamus: dict):
    if not kamus:
        return None
    pola = re.compile(r"\b(" + "|".join(map(re.escape, kamus.keys())) + r")\b")
    def normalisasi(teks: str) -> str:
        teks = "" if pd.isna(teks) else str(teks)
        return pola.sub(lambda m: kamus.get(m.group(1), m.group(1)), teks)
    return normalisasi

NORMALIZER = build_normalizer(DEFAULT_KAMUS_NORMALISASI)

def preprocess_text_pipeline(text: str, do_clean=True, do_lower=True, do_norm=True) -> str:
    t = "" if pd.isna(text) else str(text)
    if do_clean:  t = cleaning_text(t)
    if do_lower:  t = t.lower()
    if do_norm and NORMALIZER is not None: t = NORMALIZER(t)
    return t

# =========================
# LOAD CSV
# =========================
@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes, encoding="utf-8"):
    return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
