# Plots
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import streamlit as st

def setup_matplotlib_theme():
    # =========================
    # MATPLOTLIB THEME
    # =========================
    mpl.rcParams.update({
        "figure.facecolor":  "#161412",
        "axes.facecolor":    "#1e1b18",
        "axes.edgecolor":    "#2e2a25",
        "axes.labelcolor":   "#8a8070",
        "axes.titlecolor":   "#e8e0d0",
        "axes.titlesize":    11,
        "axes.labelsize":    9,
        "xtick.color":       "#8a8070",
        "ytick.color":       "#8a8070",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        "#e8e0d0",
        "grid.color":        "#2e2a25",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "lines.color":       "#8b1a1a",
        "patch.facecolor":   "#8b1a1a",
    })

def plot_hist(series, title, xlabel):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    vals = series.dropna().values
    ax.hist(vals, bins=30, color="#8b1a1a", edgecolor="#1e1b18", linewidth=0.5, alpha=0.9)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frekuensi")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def plot_confusion(cm, labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Reds")
    ax.set_title(title, pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#8a8070", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
