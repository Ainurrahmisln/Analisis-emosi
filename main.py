# main.py
import io
import re
import unicodedata
import hashlib
import json
import os
import gc

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================
# STREAMLIT CLOUD / RAM SAVER (global)
# =========================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =========================
# OPTIONAL LIBS (biar app tetap kebuka walau belum install)
# =========================
try:
    import torch
except Exception:
    torch = None

try:
    import accelerate  # noqa: F401
except Exception:
    accelerate = None

try:
    from datasets import Dataset
except Exception:
    Dataset = None

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
    )
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    DataCollatorWithPadding = None
    TrainingArguments = None
    Trainer = None

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import layers, models
except Exception:
    tf = None
    pad_sequences = None
    layers = None
    models = None


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Analisis Emosi (EDA + Evaluasi IndoBERT & GRU)", layout="wide")

# =========================
# ✅ THEME CSS (PUTIH + PINK BORDER, TAPI WIDGET DEFAULT)
# =========================
st.markdown(
    """
<style>
:root{
  --pink:#ED5D96;
  --darkred:#7B0101;
  --softpink: rgba(237,93,150,0.12);
  --softred: rgba(123,1,1,0.10);
  --text:#111111;
  --border: rgba(237,93,150,0.45);
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main{
  background: #ffffff !important;
  color: var(--text) !important;
}
[data-testid="stHeader"]{ background: rgba(0,0,0,0) !important; }
section[data-testid="stSidebar"]{
  background: #ffffff !important;
  border-right: 2px solid var(--pink) !important;
}
section[data-testid="stSidebar"] > div{ background: #ffffff !important; }
section[data-testid="stSidebar"] *{ color: var(--text) !important; }
.main .block-container{
  background: #ffffff !important;
  border: 2px solid var(--pink);
  border-radius: 22px;
  padding: 2rem 2rem 3rem;
  box-shadow: 0 10px 30px var(--softpink);
}
div[data-testid="metric-container"]{
  background: linear-gradient(135deg, var(--softpink), var(--softred)) !important;
  border: 2px solid var(--border) !important;
  border-radius: 18px !important;
  padding: 12px 14px !important;
}
div[data-testid="metric-container"] *{ color: var(--text) !important; }
div[data-testid="stDataFrame"]{
  background: #ffffff !important;
  border: 1.5px solid rgba(237,93,150,0.35) !important;
  border-radius: 14px !important;
  padding: 6px !important;
}
div[data-testid="stExpander"]{
  border: 1.5px solid rgba(237,93,150,0.35) !important;
  border-radius: 14px !important;
  background: #ffffff !important;
}
.card-pink{
  background: radial-gradient(circle at center,
    rgba(123,1,1,0.18) 0%,
    rgba(237,93,150,0.14) 55%,
    rgba(255,255,255,1) 100%) !important;
  border: 2px solid rgba(237,93,150,0.60) !important;
  border-radius: 14px !important;
  padding: 14px !important;
  box-shadow: 0 8px 18px rgba(237,93,150,0.18) !important;
}
.gap-emosi-table{ height: 22px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("ANALISIS EMOSI KOMENTAR YOUTUBE TERKAIT KEKERASAN SEKSUAL MEI 1998")
st.caption("Upload CSV → EDA → Preprocessing → Distribusi Emosi → Train IndoBERT & GRU → Evaluasi (TEST)")

# =========================
# EMOJI MAP
# =========================
EMOJI_MAP = {
    "marah": "😡",
    "sedih": "😢",
    "takut": "😨",
    "bahagia": "😄",
    "cinta": "🥰",
    "senang": "😄",
    "netral": "😐",
    "lainnya": "❓",
    "other": "❓",
}


def get_emoji(label: str) -> str:
    key = "" if label is None else str(label).strip().lower()
    return EMOJI_MAP.get(key, "❓")


# =========================
# PREPROCESSING
# =========================
ZWSP_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

DEFAULT_KAMUS_NORMALISASI = {
    "yg": "yang",
    "yng": "yang",
    "tdk": "tidak",
    "dgn": "dengan",
    "dr": "dari",
    "krn": "karena",
    "bgt": "banget",
    "udh": "udah",
    "udah": "sudah",
    "kalo": "kalau",
    "klo": "kalau",
    "sm": "sama",
    "gw": "saya",
    "gue": "saya",
    "lu": "kamu",
    "kmu": "kamu",
    "sy": "saya",
    "pls": "please",
    "utk": "untuk",
    "dlm": "dalam",
    "sdh": "sudah",
    "dg": "dengan",
    "jd": "jadi",
    "gt": "gitu",
    "jg": "juga",
    "tp": "tapi",
    "hrs": "harus",
    "mnrt": "menurut",
    "dl": "dulu",
    "lg": "lagi",
    "bnyk": "banyak",
    "bkn": "bukan",
    "br": "baru",
    "bhs": "bahasa",
    "kpd": "kepada",
    "sbg": "sebagai",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "dah": "sudah",
    "pake": "pakai",
}


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
    if do_clean:
        t = cleaning_text(t)
    if do_lower:
        t = t.lower()
    if do_norm and NORMALIZER is not None:
        t = NORMALIZER(t)
    return t


# =========================
# LOAD CSV (cache by bytes)
# =========================
@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes, encoding="utf-8"):
    return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)


# =========================
# PLOT HELPERS
# =========================
def plot_hist(series, title, xlabel):
    fig = plt.figure(figsize=(8, 3.5))
    vals = series.dropna().values
    plt.hist(vals, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frekuensi")
    st.pyplot(fig, clear_figure=True)


def plot_confusion(cm, labels, title="Confusion Matrix"):
    fig = plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Input & Settings")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    encoding = st.selectbox("Encoding CSV", ["utf-8", "latin1", "utf-8-sig"], index=0)

    st.divider()
    st.subheader("Kolom CSV")
    TEXT_COL = st.text_input("Kolom komentar", value="Komentar")
    LABEL_COL = st.text_input("Kolom label emosi (true)", value="label_emosi_autofixed")

    st.divider()
    st.subheader("EDA & Cleaning")
    show_eda_detail = st.checkbox("Tampilkan EDA detail", value=True)
    drop_duplicates = st.checkbox("Drop duplikasi komentar (untuk proses)", value=True)

    st.divider()
    st.subheader("Preprocessing")
    do_clean = st.checkbox("Cleaning", value=True)
    do_lower = st.checkbox("Lowercase", value=True)
    do_norm = st.checkbox("Normalisasi kamus", value=True)

    st.divider()
    st.subheader("Filter kata")
    use_word_filter = st.checkbox("Aktifkan filter 4–50 kata", value=True)
    min_words = st.number_input("Min kata (>)", min_value=0, value=4, step=1)
    max_words = st.number_input("Max kata (<=)", min_value=1, value=50, step=1)

    st.divider()
    st.subheader("Model config")
    MODEL_NAME = st.text_input("Base model IndoBERT", value="indobenchmark/indobert-base-p1")
    max_len = st.number_input("Max length token", min_value=32, value=128, step=8)

    st.divider()
    st.subheader("⚡ Mode cepat (biar nggak lama)")
    fast_mode = st.checkbox("Aktifkan mode cepat", value=True)
    max_train_samples = st.number_input("Batas data training (0 = semua)", min_value=0, value=2000, step=500)
    bert_epochs = st.number_input("Epoch IndoBERT", min_value=1, value=1 if fast_mode else 3, step=1)
    gru_epochs = st.number_input("Epoch GRU", min_value=1, value=2 if fast_mode else 10, step=1)
    gru_batch = st.number_input("Batch GRU", min_value=16, value=64 if fast_mode else 64, step=16)

    st.divider()
    st.subheader("Run Training")
    auto_train = st.checkbox("Auto-train setelah upload / ganti setting", value=False)
    force_retrain = st.checkbox("Paksa train ulang (abaikan cache)", value=False)

    st.divider()
    if st.button("🔄 Reset (hapus cache & hasil)"):
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            st.session_state.pop(k, None)
        st.success("Reset done. Upload ulang file.")
        st.stop()


# =========================
# GUARD: file upload
# =========================
if uploaded is None:
    st.info("Upload file CSV dulu ya.")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

df_raw = load_csv_from_bytes(file_bytes, encoding=encoding)

# =========================
# PREPROCESS + FILTER (buat pipeline kerja)
# =========================
if TEXT_COL not in df_raw.columns:
    st.error(f"Kolom komentar `{TEXT_COL}` tidak ditemukan.")
    st.stop()

df = df_raw.copy()
df[TEXT_COL] = df[TEXT_COL].astype(str)

if drop_duplicates:
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

df["text_norm"] = df[TEXT_COL].apply(lambda x: preprocess_text_pipeline(x, do_clean, do_lower, do_norm))

if use_word_filter:
    df["_len_word"] = df["text_norm"].apply(word_count)
    df = df[(df["_len_word"] > int(min_words)) & (df["_len_word"] <= int(max_words))].copy()
    df.drop(columns=["_len_word"], inplace=True, errors="ignore")

if len(df) == 0:
    st.error("Data habis setelah preprocessing/filter. Coba matikan filter kata atau ubah min/max.")
    st.stop()

st.session_state["df_proc"] = df

# =========================
# INVALIDATION: kalau file/setting berubah → hasil harus berubah
# =========================
run_config = {
    "file_hash": file_hash,
    "encoding": encoding,
    "TEXT_COL": TEXT_COL,
    "LABEL_COL": LABEL_COL,
    "drop_duplicates": drop_duplicates,
    "do_clean": do_clean,
    "do_lower": do_lower,
    "do_norm": do_norm,
    "use_word_filter": use_word_filter,
    "min_words": int(min_words),
    "max_words": int(max_words),
    "MODEL_NAME": MODEL_NAME,
    "max_len": int(max_len),
    "fast_mode": bool(fast_mode),
    "max_train_samples": int(max_train_samples),
    "bert_epochs": int(bert_epochs),
    "gru_epochs": int(gru_epochs),
    "gru_batch": int(gru_batch),
}
run_hash = hashlib.md5(json.dumps(run_config, sort_keys=True).encode("utf-8")).hexdigest()

if st.session_state.get("run_hash") != run_hash:
    st.session_state["run_hash"] = run_hash
    st.session_state.pop("results", None)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🏠 Ringkas (seperti gambar)", "📊 EDA Detail"])


# =========================
# TRAINING + EVALUATION
# =========================
def require_libs_or_stop():
    if (torch is None) or (AutoTokenizer is None) or (Trainer is None) or (Dataset is None):
        st.error("Library IndoBERT belum siap. Install: torch, transformers, datasets, accelerate.")
        st.stop()
    if accelerate is None:
        st.error("Library IndoBERT belum siap. Install: accelerate.")
        st.stop()
    if (tf is None) or (pad_sequences is None):
        st.error("Library GRU belum siap. Install: tensorflow (disarankan tensorflow-cpu di cloud).")
        st.stop()


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def run_all_training(
    df_in: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str,
    max_len: int,
    seed: int,
    bert_epochs: int,
    gru_epochs: int,
    gru_batch: int,
    fast_mode: bool,
    max_train_samples: int,
):
    require_libs_or_stop()

    # Hemat CPU
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # ---------- Data prepare ----------
    data = df_in[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str).str.strip()
    data[label_col] = data[label_col].astype(str).str.strip().str.lower()

    if len(data) < 30:
        raise RuntimeError("Data terlalu sedikit untuk training yang stabil. Minimal disarankan > 30 baris.")

    if max_train_samples and max_train_samples > 0 and len(data) > max_train_samples:
        data = data.sample(n=int(max_train_samples), random_state=seed).reset_index(drop=True)

    # cek distribusi label
    vc = data[label_col].value_counts()
    min_count = int(vc.min())
    do_stratify = min_count >= 2  # kalau ada label cuma 1, stratify pasti error

    le = LabelEncoder()
    data["label_id"] = le.fit_transform(data[label_col])
    label_names = list(le.classes_)
    num_labels = len(label_names)

    # ✅ split lebih robust: 80/20. stratify kalau memungkinkan.
    try:
        train_df, test_df = train_test_split(
            data,
            test_size=0.2,
            random_state=seed,
            stratify=data["label_id"] if do_stratify else None,
        )
    except Exception:
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=seed, stratify=None)

    # class weights full (panjang = num_labels)
    classes_in_train = np.unique(train_df["label_id"].values)
    cw_part = compute_class_weight(
        class_weight="balanced",
        classes=classes_in_train,
        y=train_df["label_id"].values,
    )
    cw_full = np.ones(num_labels, dtype="float32")
    for c, w in zip(classes_in_train, cw_part):
        cw_full[int(c)] = float(w)

    class_weights_torch = torch.tensor(cw_full, dtype=torch.float)
    class_weight_dict_tf = {i: float(cw_full[i]) for i in range(num_labels)}

    # ---------- Tokenizer + HF Datasets ----------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = Dataset.from_pandas(train_df[[text_col, "label_id"]].rename(columns={"label_id": "labels"}))
    test_ds = Dataset.from_pandas(test_df[[text_col, "label_id"]].rename(columns={"label_id": "labels"}))

    # ✅ fast_mode: paksa max_len kecil biar RAM aman
    effective_max_len = int(max_len)
    if fast_mode and effective_max_len > 64:
        effective_max_len = 64

    def tokenize_fn(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=effective_max_len)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    # RAM saver
    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # fast_mode: freeze encoder, train head saja
    if fast_mode:
        base = getattr(model, model.base_model_prefix, None)
        if base is not None:
            for p in base.parameters():
                p.requires_grad = False

    use_cuda = (torch is not None) and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir="indobert_out",
        learning_rate=5e-5 if fast_mode else 2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=int(bert_epochs),
        weight_decay=0.01,
        evaluation_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_steps=50,
        seed=seed,
        report_to="none",
        fp16=bool(use_cuda),
        disable_tqdm=True,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_torch,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=data_collator,
    )

    # ---------- IndoBERT train + TEST eval ----------
    trainer.train()

    test_pred = trainer.predict(test_ds)
    bert_logits = test_pred.predictions
    bert_y_true = test_pred.label_ids
    bert_y_pred = np.argmax(bert_logits, axis=-1)

    bert_acc = float(accuracy_score(bert_y_true, bert_y_pred))
    bert_f1m = float(f1_score(bert_y_true, bert_y_pred, average="macro", zero_division=0))
    bert_cm = confusion_matrix(bert_y_true, bert_y_pred, labels=list(range(num_labels)))
    bert_report = classification_report(
        bert_y_true, bert_y_pred, target_names=label_names, output_dict=True, zero_division=0
    )

    # ✅ hemat RAM: jangan load AutoModel lagi → ambil embedding dari model yang sudah ada
    W = model.get_input_embeddings().weight.detach().cpu().numpy().astype("float32")
    vocab_size, emb_dim = W.shape

    # bersihin torch side
    del trainer
    del model
    gc.collect()
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ---------- GRU (TEST eval) ----------
    # clear TF session biar gak numpuk memori saat rerun
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    X_train_ids = pad_sequences(
        train_ds["input_ids"],
        maxlen=effective_max_len,
        padding="post",
        truncating="post",
        value=pad_id,
    )
    X_test_ids = pad_sequences(
        test_ds["input_ids"],
        maxlen=effective_max_len,
        padding="post",
        truncating="post",
        value=pad_id,
    )

    y_train = np.array(train_ds["labels"])
    y_test = np.array(test_ds["labels"])

    gru_units = 64 if fast_mode else 128
    gru_dropout = 0.2 if fast_mode else 0.3

    # ✅ batch jangan kegedean di cloud
    effective_gru_batch = int(min(int(gru_batch), 64))

    model_gru = models.Sequential(
        [
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=emb_dim,
                weights=[W],
                trainable=False,
                mask_zero=True,
            ),
            layers.Bidirectional(
                layers.GRU(gru_units, dropout=gru_dropout, recurrent_dropout=0.0)
            ),
            layers.Dropout(gru_dropout),
            layers.Dense(num_labels, activation="softmax"),
        ]
    )

    model_gru.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_gru.fit(
        X_train_ids,
        y_train,
        validation_split=0.1,  # ✅ aman walau kelas kecil (dibanding 80/10/10 stratify)
        epochs=int(gru_epochs),
        batch_size=effective_gru_batch,
        class_weight=class_weight_dict_tf,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
        verbose=0,
    )

    probs = model_gru.predict(X_test_ids, verbose=0)
    gru_y_pred = np.argmax(probs, axis=1)

    gru_acc = float(accuracy_score(y_test, gru_y_pred))
    gru_f1m = float(f1_score(y_test, gru_y_pred, average="macro", zero_division=0))
    gru_cm = confusion_matrix(y_test, gru_y_pred, labels=list(range(num_labels)))
    gru_report = classification_report(
        y_test, gru_y_pred, target_names=label_names, output_dict=True, zero_division=0
    )

    return {
        "label_names": label_names,
        "sizes": {"train": int(len(train_df)), "val": 0, "test": int(len(test_df))},
        "bert": {"test_accuracy": bert_acc, "test_macro_f1": bert_f1m, "cm": bert_cm, "report": bert_report},
        "gru": {"test_accuracy": gru_acc, "test_macro_f1": gru_f1m, "cm": gru_cm, "report": gru_report},
        "meta": {
            "fast_mode": bool(fast_mode),
            "min_label_count": int(min_count),
            "stratify_used": bool(do_stratify),
            "max_train_samples": int(max_train_samples),
            "bert_epochs": int(bert_epochs),
            "gru_epochs": int(gru_epochs),
            "gru_batch": int(gru_batch),
            "effective_gru_batch": int(effective_gru_batch),
            "torch_cuda": bool(torch is not None and torch.cuda.is_available()),
            "effective_max_len": int(effective_max_len),
            "bert_batch": 2,
            "grad_accum": 8,
        },
    }


# =========================
# TAB 1: RINGKAS
# =========================
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Jumlah baris", int(len(df_raw)))
    with c2:
        st.metric("Jumlah kolom", int(df_raw.shape[1]))
    with c3:
        st.metric("Missing (total)", int(df_raw.isna().sum().sum()))

    st.subheader("Preview Data")
    st.dataframe(df_raw.head(15), use_container_width=True)

    # Distribusi emosi
    st.subheader("Jumlah emosi")
    if LABEL_COL not in df.columns:
        st.warning(f"Kolom label `{LABEL_COL}` tidak ada → distribusi emosi & training tidak bisa jalan.")
    else:
        s = df[LABEL_COL].astype(str).str.strip().str.lower()
        s = s[(s.notna()) & (s != "") & (s != "nan")]
        vc = s.value_counts()
        total = int(vc.sum())
        pct = (vc / total * 100).round(2) if total else vc * 0

        dist_df = pd.DataFrame(
            {
                "emoji": [get_emoji(x) for x in vc.index],
                "emosi": vc.index,
                "jumlah": vc.values,
                "persen": pct.values,
            }
        )

        cols = st.columns(min(5, len(dist_df)))
        for i, row in enumerate(dist_df.head(5).itertuples(index=False)):
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="card-pink">
                        <div style="font-size:34px;line-height:1;">{row.emoji}</div>
                        <div style="font-size:18px;font-weight:700;margin-top:6px;">{row.emosi}</div>
                        <div style="font-size:16px;margin-top:6px;">{int(row.jumlah)} ({row.persen}%)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='gap-emosi-table'></div>", unsafe_allow_html=True)
        st.dataframe(dist_df, use_container_width=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.subheader("Total data setelah EDA & Preprocessing")
    st.metric("Total data (setelah preprocessing)", int(len(df)))
    removed = int(len(df_raw) - len(df))
    st.caption(f"Data berkurang: {removed} baris (dari {len(df_raw)} → {len(df)})")

    st.divider()
    st.subheader("Hasil prediksi model (Evaluasi TEST)")

    if LABEL_COL not in df.columns:
        st.info("Tidak bisa training/evaluasi karena kolom label true tidak ada.")
        st.stop()

    run_now = auto_train or st.button("▶️ Jalankan Training & Evaluasi (TEST)", type="primary")

    if run_now:
        if (st.session_state.get("results") is None) or force_retrain:
            with st.spinner("Training IndoBERT & GRU sedang berjalan..."):
                try:
                    results = run_all_training(
                        df_in=st.session_state["df_proc"],
                        text_col="text_norm",
                        label_col=LABEL_COL,
                        model_name=MODEL_NAME,
                        max_len=int(max_len),
                        seed=42,
                        bert_epochs=int(bert_epochs),
                        gru_epochs=int(gru_epochs),
                        gru_batch=int(gru_batch),
                        fast_mode=bool(fast_mode),
                        max_train_samples=int(max_train_samples),
                    )
                    st.session_state["results"] = results
                except Exception as e:
                    st.exception(e)  # tampilkan traceback lengkap
                    st.stop()

    if st.session_state.get("results") is None:
        st.warning("Belum ada hasil. Klik tombol Run atau aktifkan Auto-train di sidebar.")
    else:
        res = st.session_state["results"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("IndoBERT Test Accuracy", f"{res['bert']['test_accuracy']:.4f}")
        with c2:
            st.metric("GRU Test Accuracy", f"{res['gru']['test_accuracy']:.4f}")
        with c3:
            st.metric("Split (train/test)", f"{res['sizes']['train']}/{res['sizes']['test']}")

        summary_df = pd.DataFrame(
            [
                {
                    "model": "IndoBERT Fine-tuning",
                    "test_accuracy": res["bert"]["test_accuracy"],
                    "test_macro_f1": res["bert"]["test_macro_f1"],
                },
                {
                    "model": "GRU",
                    "test_accuracy": res["gru"]["test_accuracy"],
                    "test_macro_f1": res["gru"]["test_macro_f1"],
                },
            ]
        )
        st.dataframe(summary_df, use_container_width=True)

        with st.expander("📄 Detail evaluasi (classification report & confusion matrix)"):
            st.write("**IndoBERT — Classification Report (TEST)**")
            st.dataframe(pd.DataFrame(res["bert"]["report"]).T, use_container_width=True)
            st.write("**IndoBERT — Confusion Matrix (TEST)**")
            plot_confusion(res["bert"]["cm"], res["label_names"], "IndoBERT - Confusion Matrix (TEST)")

            st.write("**GRU — Classification Report (TEST)**")
            st.dataframe(pd.DataFrame(res["gru"]["report"]).T, use_container_width=True)
            st.write("**GRU — Confusion Matrix (TEST)**")
            plot_confusion(res["gru"]["cm"], res["label_names"], "GRU - Confusion Matrix (TEST)")

        with st.expander("Detail konfigurasi run"):
            st.json(res["meta"])


# =========================
# TAB 2: EDA DETAIL
# =========================
with tab2:
    if not show_eda_detail:
        st.info("EDA detail dimatikan di sidebar.")
    else:
        st.subheader("EDA Detail")

        st.write("**Missing per kolom**")
        miss_df = pd.DataFrame({"dtype": df_raw.dtypes.astype(str), "missing": df_raw.isna().sum()}).sort_values(
            "missing", ascending=False
        )
        st.dataframe(miss_df, use_container_width=True)

        if TEXT_COL in df_raw.columns:
            dup_count = int(df_raw.duplicated(subset=[TEXT_COL]).sum())
            st.metric("Duplikasi komentar (berdasarkan kolom komentar)", dup_count)

        if TEXT_COL in df_raw.columns:
            tmp = df_raw.copy()
            tmp[TEXT_COL] = tmp[TEXT_COL].astype(str).fillna("")
            tmp["len_char"] = tmp[TEXT_COL].apply(len)
            tmp["len_word"] = tmp[TEXT_COL].apply(lambda x: len(x.split()))

            st.write("**Statistik panjang komentar (raw)**")
            st.dataframe(tmp[["len_char", "len_word"]].describe(), use_container_width=True)

            st.write("**Histogram panjang komentar (jumlah kata)**")
            plot_hist(tmp["len_word"], "Distribusi Panjang Komentar (Jumlah Kata)", "Jumlah kata")
        else:
            st.warning(f"Kolom `{TEXT_COL}` tidak ada, jadi EDA panjang komentar tidak ditampilkan.")
