# Tampilan Sidebar
import os
import streamlit as st
from config import _IMPORT_ERRORS
from models import torch, transformers, accelerate


def render_sidebar():
    with st.sidebar:
        st.markdown("### Konfigurasi")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        encoding = st.selectbox("Encoding CSV", ["utf-8", "latin1", "utf-8-sig"], index=0)

        st.divider()
        st.markdown("#### Kolom CSV")
        TEXT_COL  = st.text_input("Kolom komentar", value="Komentar")
        LABEL_COL = st.text_input("Kolom label emosi (true)", value="label_emosi_autofixed")

        st.divider()
        st.markdown("#### EDA & Cleaning")
        show_eda_detail = st.checkbox("Tampilkan EDA detail", value=True)
        drop_duplicates = st.checkbox("Drop duplikasi komentar", value=True)

        st.divider()
        st.markdown("#### Preprocessing")
        do_clean = st.checkbox("Cleaning", value=True)
        do_lower = st.checkbox("Lowercase", value=True)
        do_norm  = st.checkbox("Normalisasi kamus", value=True)

        st.divider()
        st.markdown("#### Filter Kata")
        use_word_filter = st.checkbox("Aktifkan filter 4-50 kata", value=True)
        min_words = st.number_input("Min kata (>)", min_value=0, value=4, step=1)
        max_words = st.number_input("Max kata (<=)", min_value=1, value=50, step=1)

        st.divider()
        st.markdown("#### Model Config")
        MODEL_NAME = st.text_input("Base model IndoBERT", value="indobenchmark/indobert-base-p1")
        max_len    = st.number_input("Max length token", min_value=32, value=128, step=8)

        st.divider()
        st.markdown("#### Mode Cepat")
        fast_mode         = st.checkbox("Aktifkan mode cepat", value=True)
        max_train_samples = st.number_input("Batas data training (0 = semua)", min_value=0, value=3000, step=500)
        bert_epochs       = st.number_input("Epoch IndoBERT", min_value=1, value=1, step=1)
        gru_epochs        = st.number_input("Epoch GRU",      min_value=1, value=3, step=1)
        gru_batch         = st.number_input("Batch GRU",      min_value=8, value=128, step=8)

        st.divider()
        st.markdown("#### Run Training")
        auto_train    = st.checkbox("Auto-train setelah upload", value=False)
        force_retrain = st.checkbox("Paksa train ulang", value=False)

        st.divider()
        if st.button("Reset Semua"):
            st.cache_data.clear()
            for k in list(st.session_state.keys()):
                st.session_state.pop(k, None)
            st.success("Reset selesai. Upload ulang file.")
            st.stop()

        with st.expander("Diagnostik Library"):
            st.write("Python:", os.sys.version)
            if torch is not None:
                st.write("torch:", getattr(torch, "__version__", "unknown"))
            else:
                st.error("torch gagal import")
            if transformers is not None:
                st.write("transformers:", getattr(transformers, "__version__", "unknown"))
            else:
                st.error("transformers gagal import")
            if accelerate is not None:
                st.write("accelerate:", getattr(accelerate, "__version__", "unknown"))
            else:
                st.warning("accelerate gagal import")
            if _IMPORT_ERRORS:
                st.code("\n\n".join(
                    [f"[{k}] Hint: {v[0]}\n{v[1]}" for k, v in _IMPORT_ERRORS.items()]
                ))

    return {
        "uploaded": uploaded, "encoding": encoding,
        "TEXT_COL": TEXT_COL, "LABEL_COL": LABEL_COL,
        "show_eda_detail": show_eda_detail, "drop_duplicates": drop_duplicates,
        "do_clean": do_clean, "do_lower": do_lower, "do_norm": do_norm,
        "use_word_filter": use_word_filter, "min_words": min_words, "max_words": max_words,
        "MODEL_NAME": MODEL_NAME, "max_len": max_len,
        "fast_mode": fast_mode, "max_train_samples": max_train_samples,
        "bert_epochs": bert_epochs, "gru_epochs": gru_epochs, "gru_batch": gru_batch,
        "auto_train": auto_train, "force_retrain": force_retrain,
    }
