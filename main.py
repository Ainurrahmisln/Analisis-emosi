# Main
import hashlib
import json
import os
import streamlit as st

from config import _IMPORT_ERRORS
from styles import inject_styles
from plots import setup_matplotlib_theme
from preprocessing import load_csv_from_bytes, preprocess_text_pipeline, word_count
from ui import render_sidebar, render_tab_summary, render_tab_eda

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Analisis Emosi · Tragedi Mei 1998",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
setup_matplotlib_theme()

# =========================
# HEADER
# =========================
st.markdown('<span class="section-eyebrow">Sistem Analisis NLP · Sejarah Digital</span>', unsafe_allow_html=True)
st.title("Analisis Emosi Komentar YouTube\nTerkait Kekerasan Seksual Mei 1998")
st.markdown(
    '<span class="title-caption">'
    'Upload CSV &rarr; EDA &rarr; Preprocessing &rarr; Distribusi Emosi &rarr; '
    'Train IndoBERT &amp; GRU (PyTorch) &rarr; Evaluasi (TEST SET)'
    '</span>',
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================
config = render_sidebar()

# =========================
# GUARD: FILE UPLOAD
# =========================
if config["uploaded"] is None:
    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)
    st.info("Upload file CSV melalui sidebar untuk memulai analisis.")
    st.stop()

file_bytes = config["uploaded"].getvalue()
file_hash  = hashlib.md5(file_bytes).hexdigest()
df_raw     = load_csv_from_bytes(file_bytes, encoding=config["encoding"])

# =========================
# PREPROCESS + FILTER
# =========================
TEXT_COL = config["TEXT_COL"]
if TEXT_COL not in df_raw.columns:
    st.error(f"Kolom komentar `{TEXT_COL}` tidak ditemukan.")
    st.stop()

df           = df_raw.copy()
df[TEXT_COL] = df[TEXT_COL].astype(str)

if config["drop_duplicates"]:
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

df["text_norm"] = df[TEXT_COL].apply(
    lambda x: preprocess_text_pipeline(
        x,
        do_clean=config["do_clean"],
        do_lower=config["do_lower"],
        do_norm=config["do_norm"],
    )
).astype(str)

if config["use_word_filter"]:
    df["_len_word"] = df["text_norm"].apply(word_count)
    df = df[
        (df["_len_word"] > int(config["min_words"])) &
        (df["_len_word"] <= int(config["max_words"]))
    ].copy()
    df.drop(columns=["_len_word"], inplace=True, errors="ignore")

st.session_state["df_proc"] = df

# =========================
# CACHE INVALIDATION
# =========================
run_config_for_hash         = {k: v for k, v in config.items() if k != "uploaded"}
run_config_for_hash["file_hash"] = file_hash
run_hash = hashlib.md5(
    json.dumps(run_config_for_hash, sort_keys=True, default=str).encode()
).hexdigest()

if st.session_state.get("run_hash") != run_hash:
    st.session_state["run_hash"] = run_hash
    st.session_state.pop("results", None)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["RINGKASAN", "EDA DETAIL"])

with tab1:
    render_tab_summary(df_raw, df, config)

with tab2:
    render_tab_eda(df_raw, config)