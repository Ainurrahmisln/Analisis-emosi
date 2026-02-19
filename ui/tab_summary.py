# Tampilan Tab Summary
import pandas as pd
import streamlit as st
import traceback
from config import get_emoji
from plots import plot_confusion
from training import run_all_training


def render_tab_summary(df_raw, df, config):
    LABEL_COL = config["LABEL_COL"]

    # =========================
    # DATASET OVERVIEW
    # =========================
    st.markdown('<span class="section-eyebrow">Dataset Overview</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Jumlah Baris",   f"{len(df_raw):,}")
    with c2: st.metric("Jumlah Kolom",   int(df_raw.shape[1]))
    with c3: st.metric("Missing Values", f"{int(df_raw.isna().sum().sum()):,}")

    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-eyebrow">Preview Data Mentah</span>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(15), use_container_width=True)

    # =========================
    # EMOTION DISTRIBUTION
    # =========================
    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-eyebrow">Distribusi Emosi</span>', unsafe_allow_html=True)

    if LABEL_COL not in df.columns:
        st.warning(f"Kolom label `{LABEL_COL}` tidak ada - distribusi emosi & training tidak tersedia.")
    else:
        s     = df[LABEL_COL].astype(str).str.strip().str.lower()
        s     = s[(s.notna()) & (s != "") & (s != "nan")]
        vc    = s.value_counts()
        total = int(vc.sum())
        pct   = (vc / total * 100).round(2) if total else vc * 0

        cols = st.columns(min(5, len(vc)))
        for i, (label, count) in enumerate(vc.head(5).items()):
            with cols[i]:
                st.markdown(f"""
                <div class="emo-card">
                    <div class="emo-icon">{get_emoji(label)}</div>
                    <div class="emo-label">{label}</div>
                    <div class="emo-count">{count:,}</div>
                    <div class="emo-pct">{pct[label]}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="gap-sm"></div>', unsafe_allow_html=True)
        dist_df = pd.DataFrame({
            "emoji":  [get_emoji(x) for x in vc.index],
            "emosi":  vc.index,
            "jumlah": vc.values,
            "persen": pct.values,
        })
        st.dataframe(dist_df, use_container_width=True)

    # =========================
    # POST-PREPROCESSING STATS
    # =========================
    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-eyebrow">Setelah EDA & Preprocessing</span>', unsafe_allow_html=True)
    removed = int(len(df_raw) - len(df))
    c1, c2  = st.columns(2)
    with c1: st.metric("Data setelah preprocessing", f"{len(df):,}")
    with c2: st.metric("Baris dihapus", f"{removed:,}", delta=f"-{removed}", delta_color="inverse")

    # =========================
    # TRAINING
    # =========================
    st.divider()
    st.markdown('<span class="section-eyebrow">Evaluasi Model · Test Set</span>', unsafe_allow_html=True)

    if LABEL_COL not in df.columns:
        st.info("Training tidak tersedia karena kolom label true tidak ditemukan.")
        return

    run_now = config["auto_train"] or st.button("Jalankan Training & Evaluasi", type="primary")

    if run_now:
        if (st.session_state.get("results") is None) or config["force_retrain"]:
            with st.spinner("Sedang training IndoBERT & GRU (PyTorch)..."):
                try:
                    results = run_all_training(
                        df_in=df,
                        text_col="text_norm",
                        label_col=LABEL_COL,
                        model_name=config["MODEL_NAME"],
                        max_len=int(config["max_len"]),
                        seed=42,
                        bert_epochs=int(config["bert_epochs"]),
                        gru_epochs=int(config["gru_epochs"]),
                        gru_batch=int(config["gru_batch"]),
                        fast_mode=bool(config["fast_mode"]),
                        max_train_samples=int(config["max_train_samples"]),
                    )
                    st.session_state["results"] = results
                except Exception as e:
                    st.error(f"Gagal training/evaluasi: {e}")
                    st.code(traceback.format_exc())
                    return

    if st.session_state.get("results") is None:
        st.warning("Belum ada hasil. Klik tombol Run atau aktifkan Auto-train di sidebar.")
        return

    res = st.session_state["results"]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("IndoBERT · Accuracy", f"{res['bert']['test_accuracy']:.4f}")
    with c2: st.metric("GRU · Accuracy",      f"{res['gru']['test_accuracy']:.4f}")
    with c3: st.metric("Split train/val/test",
                        f"{res['sizes']['train']} / {res['sizes']['val']} / {res['sizes']['test']}")

    st.markdown('<div class="gap-sm"></div>', unsafe_allow_html=True)

    summary_df = pd.DataFrame([
        {"model": "IndoBERT Fine-tuning",
         "test_accuracy": f"{res['bert']['test_accuracy']:.4f}",
         "test_macro_f1": f"{res['bert']['test_macro_f1']:.4f}"},
        {"model": "GRU (PyTorch)",
         "test_accuracy": f"{res['gru']['test_accuracy']:.4f}",
         "test_macro_f1": f"{res['gru']['test_macro_f1']:.4f}"},
    ])
    st.dataframe(summary_df, use_container_width=True)

    with st.expander("DETAIL EVALUASI · Classification Report & Confusion Matrix"):
        st.markdown("**IndoBERT - Classification Report (TEST)**")
        st.dataframe(pd.DataFrame(res["bert"]["report"]).T, use_container_width=True)
        st.markdown('<div class="gap-sm"></div>', unsafe_allow_html=True)
        st.markdown("**IndoBERT - Confusion Matrix (TEST)**")
        plot_confusion(res["bert"]["cm"], res["label_names"], "IndoBERT · Confusion Matrix (TEST)")

        st.divider()
        st.markdown("**GRU (PyTorch) - Classification Report (TEST)**")
        st.dataframe(pd.DataFrame(res["gru"]["report"]).T, use_container_width=True)
        st.markdown('<div class="gap-sm"></div>', unsafe_allow_html=True)
        st.markdown("**GRU (PyTorch) - Confusion Matrix (TEST)**")
        plot_confusion(res["gru"]["cm"], res["label_names"], "GRU (PyTorch) · Confusion Matrix (TEST)")

    with st.expander("KONFIGURASI RUN"):
        st.json(res["meta"])