# Tampilan Tab EDA
import pandas as pd
import streamlit as st
from plots import plot_hist


def render_tab_eda(df_raw, config):
    show_eda_detail = config["show_eda_detail"]
    TEXT_COL        = config["TEXT_COL"]

    if not show_eda_detail:
        st.info("EDA detail dimatikan. Aktifkan di sidebar.")
        return

    # =========================
    # MISSING VALUES
    # =========================
    st.markdown('<span class="section-eyebrow">Missing Values per Kolom</span>', unsafe_allow_html=True)
    miss_df = pd.DataFrame({
        "dtype":   df_raw.dtypes.astype(str),
        "missing": df_raw.isna().sum(),
    }).sort_values("missing", ascending=False)
    st.dataframe(miss_df, use_container_width=True)

    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)

    if TEXT_COL not in df_raw.columns:
        st.warning(f"Kolom `{TEXT_COL}` tidak ada - EDA panjang komentar tidak tersedia.")
        return

    # =========================
    # DUPLICATES
    # =========================
    dup_count = int(df_raw.duplicated(subset=[TEXT_COL]).sum())
    st.metric("Duplikasi komentar", f"{dup_count:,}")

    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)

    # =========================
    # LENGTH STATS
    # =========================
    tmp             = df_raw.copy()
    tmp[TEXT_COL]   = tmp[TEXT_COL].astype(str).fillna("")
    tmp["len_char"] = tmp[TEXT_COL].apply(len)
    tmp["len_word"] = tmp[TEXT_COL].apply(lambda x: len(x.split()))

    st.markdown('<span class="section-eyebrow">Statistik Panjang Komentar (Raw)</span>', unsafe_allow_html=True)
    st.dataframe(tmp[["len_char", "len_word"]].describe(), use_container_width=True)

    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-eyebrow">Distribusi Panjang Komentar (Jumlah Kata)</span>', unsafe_allow_html=True)
    plot_hist(tmp["len_word"], "Distribusi Panjang Komentar (Jumlah Kata)", "Jumlah kata")