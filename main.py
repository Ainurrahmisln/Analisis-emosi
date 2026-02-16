# main.py
import io
import re
import unicodedata
import hashlib
import json
import os
import gc
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# =========================
# STREAMLIT CLOUD / RAM SAVER
# =========================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =========================
# OPTIONAL IMPORTS (tampilkan error aslinya kalau gagal)
# =========================
_IMPORT_ERRORS = {}

def _optional_import(name: str, hint: str = ""):
    try:
        mod = __import__(name)
        return mod
    except Exception:
        _IMPORT_ERRORS[name] = (hint, traceback.format_exc())
        return None

torch = _optional_import("torch", "pip install torch")
transformers = _optional_import("transformers", "pip install transformers accelerate sentencepiece")

if transformers is not None:
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            TrainingArguments,
            Trainer,
        )
    except Exception:
        _IMPORT_ERRORS["transformers_subimports"] = ("pip install transformers", traceback.format_exc())
        AutoTokenizer = None
        AutoModelForSequenceClassification = None
        DataCollatorWithPadding = None
        TrainingArguments = None
        Trainer = None
else:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    DataCollatorWithPadding = None
    TrainingArguments = None
    Trainer = None

accelerate = _optional_import("accelerate", "pip install accelerate")


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Analisis Emosi (EDA + Evaluasi IndoBERT & GRU)", layout="wide")

# =========================
# THEME CSS (punya kamu)
# =========================
st.markdown("""
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
""", unsafe_allow_html=True)

st.title("ANALISIS EMOSI KOMENTAR YOUTUBE TERKAIT KEKERASAN SEKSUAL MEI 1998")
st.caption("Upload CSV → EDA → Preprocessing → Distribusi Emosi → Train IndoBERT & GRU (PyTorch) → Evaluasi (TEST)")


# =========================
# DIAGNOSTIC (biar jelas kenapa dianggap 'belum siap')
# =========================
with st.expander("🧪 Diagnostik library (kalau error import)"):
    st.write("Kalau deploy gagal/anggap library belum siap, cek detail di sini.")
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
        st.warning("accelerate gagal import (Trainer butuh accelerate)")

    if _IMPORT_ERRORS:
        st.code("\n\n".join(
            [f"[{k}] Hint: {v[0]}\n{v[1]}" for k, v in _IMPORT_ERRORS.items()]
        ))


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
ZWSP_RE  = re.compile(r"[\u200b\u200c\u200d\ufeff]")
URL_RE   = re.compile(r"http[s]?://\S+|www\.\S+")
TAG_RE   = re.compile(r"<[^>]+>")
WS_RE    = re.compile(r"\s+")

DEFAULT_KAMUS_NORMALISASI = {
    "yg": "yang", "yng": "yang", "tdk": "tidak", "dgn": "dengan", "dr": "dari", "krn": "karena",
    "bgt": "banget", "udh": "udah", "udah": "sudah", "kalo": "kalau", "klo": "kalau", "sm": "sama",
    "gw": "saya", "gue": "saya", "lu": "kamu", "kmu": "kamu", "sy": "saya", "pls": "please",
    "utk": "untuk", "dlm": "dalam", "sdh": "sudah", "dg": "dengan", "jd": "jadi", "gt": "gitu",
    "jg": "juga", "tp": "tapi", "hrs": "harus", "mnrt": "menurut", "dl": "dulu", "lg": "lagi",
    "bnyk": "banyak", "bkn": "bukan", "br": "baru", "bhs": "bahasa", "kpd": "kepada", "sbg": "sebagai",
    "ga": "tidak", "gak": "tidak", "nggak": "tidak", "dah": "sudah", "pake": "pakai",
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
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
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
    TEXT_COL  = st.text_input("Kolom komentar", value="Komentar")
    LABEL_COL = st.text_input("Kolom label emosi (true)", value="label_emosi_autofixed")

    st.divider()
    st.subheader("EDA & Cleaning")
    show_eda_detail = st.checkbox("Tampilkan EDA detail", value=True)
    drop_duplicates = st.checkbox("Drop duplikasi komentar (untuk proses)", value=True)

    st.divider()
    st.subheader("Preprocessing")
    do_clean = st.checkbox("Cleaning", value=True)
    do_lower = st.checkbox("Lowercase", value=True)
    do_norm  = st.checkbox("Normalisasi kamus", value=True)

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
    max_train_samples = st.number_input("Batas data training (0 = semua)", min_value=0, value=3000, step=500)
    bert_epochs = st.number_input("Epoch IndoBERT", min_value=1, value=1 if fast_mode else 3, step=1)
    gru_epochs  = st.number_input("Epoch GRU", min_value=1, value=3 if fast_mode else 10, step=1)
    gru_batch   = st.number_input("Batch GRU", min_value=8, value=128 if fast_mode else 64, step=8)

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
# PREPROCESS + FILTER
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

st.session_state["df_proc"] = df


# =========================
# INVALIDATION
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
# TRAINING / EVAL HELPERS
# =========================
def require_libs_or_stop():
    if torch is None:
        st.error("PyTorch (torch) belum siap / gagal import. Lihat expander Diagnostik di atas.")
        st.stop()
    if transformers is None or AutoTokenizer is None or Trainer is None or TrainingArguments is None:
        st.error("Transformers/Trainer belum siap. Pastikan: transformers + accelerate + sentencepiece.")
        st.stop()
    if accelerate is None:
        st.error("accelerate belum siap (Trainer butuh accelerate). Install: accelerate.")
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

class TorchTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.labels = labels
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=int(max_len),
            padding=False,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: self.enc[k][idx] for k in self.enc.keys()}
        item["labels"] = int(self.labels[idx])
        return item

class GRUClassifier(torch.nn.Module):
    def __init__(self, emb_weight, pad_id: int, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings=emb_weight,
            freeze=True,
            padding_idx=pad_id
        )
        emb_dim = emb_weight.shape[1]
        self.gru = torch.nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # [B, T, E]

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1).to("cpu")
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, h = self.gru(packed)
        else:
            out, h = self.gru(x)

        # h: [num_directions, B, H] karena 1 layer
        # bidirectional => 2, jadi concat
        h_fwd = h[0]
        h_bwd = h[1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 2H]

        z = self.dropout(h_cat)
        logits = self.fc(z)
        return logits

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
    max_train_samples: int
):
    require_libs_or_stop()

    # hemat CPU
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    data = df_in[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str).str.strip()
    data[label_col] = data[label_col].astype(str).str.strip().str.lower()

    if max_train_samples and max_train_samples > 0 and len(data) > max_train_samples:
        data = data.sample(n=int(max_train_samples), random_state=seed).reset_index(drop=True)

    le = LabelEncoder()
    data["label_id"] = le.fit_transform(data[label_col])
    label_names = list(le.classes_)
    num_labels = len(label_names)

    train_df, temp_df = train_test_split(
        data, test_size=0.2, random_state=seed, stratify=data["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label_id"]
    )

    classes = np.unique(train_df["label_id"].values)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["label_id"].values)
    class_weights_torch = torch.tensor(cw, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    effective_max_len = int(max_len)
    if fast_mode and effective_max_len > 96:
        effective_max_len = 96

    train_ds = TorchTextDataset(train_df[text_col].tolist(), train_df["label_id"].tolist(), tokenizer, effective_max_len)
    val_ds   = TorchTextDataset(val_df[text_col].tolist(),   val_df["label_id"].tolist(),   tokenizer, effective_max_len)
    test_ds  = TorchTextDataset(test_df[text_col].tolist(),  test_df["label_id"].tolist(),  tokenizer, effective_max_len)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    # hemat mem
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    if fast_mode:
        base = getattr(model, model.base_model_prefix, None)
        if base is not None:
            for p in base.parameters():
                p.requires_grad = False

    use_cuda = bool(torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir="indobert_out",
        learning_rate=5e-5 if fast_mode else 2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=int(bert_epochs),
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_steps=50,
        seed=seed,
        report_to="none",
        fp16=bool(use_cuda),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_torch,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()

    # eval BERT
    test_pred = trainer.predict(test_ds)
    bert_logits = test_pred.predictions
    bert_y_true = test_pred.label_ids
    bert_y_pred = np.argmax(bert_logits, axis=-1)

    bert_acc = float(accuracy_score(bert_y_true, bert_y_pred))
    bert_f1m = float(f1_score(bert_y_true, bert_y_pred, average="macro", zero_division=0))
    bert_cm = confusion_matrix(bert_y_true, bert_y_pred, labels=list(range(num_labels)))
    bert_report = classification_report(bert_y_true, bert_y_pred, target_names=label_names, output_dict=True, zero_division=0)

    # bersihin trainer pred buffer
    gc.collect()
    if use_cuda:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # =========================
    # GRU (PyTorch) pakai embedding IndoBERT
    # =========================
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # embedding weight langsung dari model fine-tuned
    emb_weight = model.get_input_embeddings().weight.detach().cpu()

    # tokenize untuk GRU (pakai padding max_length biar gampang)
    def encode_for_gru(texts):
        enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=effective_max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]

    X_train_ids, M_train = encode_for_gru(train_df[text_col].tolist())
    X_val_ids,   M_val   = encode_for_gru(val_df[text_col].tolist())
    X_test_ids,  M_test  = encode_for_gru(test_df[text_col].tolist())

    y_train = torch.tensor(train_df["label_id"].values, dtype=torch.long)
    y_val   = torch.tensor(val_df["label_id"].values, dtype=torch.long)
    y_test  = torch.tensor(test_df["label_id"].values, dtype=torch.long)

    device = torch.device("cuda" if use_cuda else "cpu")
    hidden = 64 if fast_mode else 128
    drop = 0.2 if fast_mode else 0.3

    gru_model = GRUClassifier(emb_weight, pad_id=pad_id, hidden_size=hidden, num_labels=num_labels, dropout=drop).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_torch.to(device))
    opt = torch.optim.Adam(gru_model.parameters(), lr=1e-4)

    def make_loader(X, M, y, batch):
        ds = torch.utils.data.TensorDataset(X, M, y)
        return torch.utils.data.DataLoader(ds, batch_size=int(batch), shuffle=True)

    train_loader = make_loader(X_train_ids, M_train, y_train, gru_batch)
    val_loader   = make_loader(X_val_ids,   M_val,   y_val,   gru_batch)
    test_loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_ids, M_test, y_test),
        batch_size=int(gru_batch),
        shuffle=False
    )

    best_val = float("inf")
    patience = 1
    bad = 0

    for _epoch in range(int(gru_epochs)):
        gru_model.train()
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = gru_model(xb, attention_mask=mb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # val
        gru_model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, mb, yb in val_loader:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                logits = gru_model(xb, attention_mask=mb)
                loss = loss_fn(logits, yb)
                val_loss += float(loss.item()) * yb.size(0)
                n += yb.size(0)
        val_loss = val_loss / max(n, 1)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
        else:
            bad += 1
            if bad > patience:
                break

    # test GRU
    gru_model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, mb, yb in test_loader:
            xb, mb = xb.to(device), mb.to(device)
            logits = gru_model(xb, attention_mask=mb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(yb.numpy().tolist())

    gru_acc = float(accuracy_score(all_true, all_pred))
    gru_f1m = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    gru_cm  = confusion_matrix(all_true, all_pred, labels=list(range(num_labels)))
    gru_report = classification_report(all_true, all_pred, target_names=label_names, output_dict=True, zero_division=0)

    return {
        "label_names": label_names,
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "bert": {"test_accuracy": bert_acc, "test_macro_f1": bert_f1m, "cm": bert_cm, "report": bert_report},
        "gru":  {"test_accuracy": gru_acc,  "test_macro_f1": gru_f1m,  "cm": gru_cm,  "report": gru_report},
        "meta": {
            "fast_mode": fast_mode,
            "max_train_samples": int(max_train_samples),
            "bert_epochs": int(bert_epochs),
            "gru_epochs": int(gru_epochs),
            "gru_batch": int(gru_batch),
            "torch_cuda": bool(use_cuda),
            "effective_max_len": int(effective_max_len),
            "bert_batch": 2,
            "grad_accum": 8,
        }
    }


# =========================
# TAB 1: RINGKAS
# =========================
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Jumlah baris", int(len(df_raw)))
    with c2: st.metric("Jumlah kolom", int(df_raw.shape[1]))
    with c3: st.metric("Missing (total)", int(df_raw.isna().sum().sum()))

    st.subheader("Preview Data")
    st.dataframe(df_raw.head(15), use_container_width=True)

    st.subheader("Jumlah emosi")
    if LABEL_COL not in df.columns:
        st.warning(f"Kolom label `{LABEL_COL}` tidak ada → distribusi emosi & training tidak bisa jalan.")
    else:
        s = df[LABEL_COL].astype(str).str.strip().str.lower()
        s = s[(s.notna()) & (s != "") & (s != "nan")]
        vc = s.value_counts()
        total = int(vc.sum())
        pct = (vc / total * 100).round(2) if total else vc * 0

        dist_df = pd.DataFrame({
            "emoji": [get_emoji(x) for x in vc.index],
            "emosi": vc.index,
            "jumlah": vc.values,
            "persen": pct.values
        })

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
                    unsafe_allow_html=True
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
            with st.spinner("Training IndoBERT & GRU (PyTorch) sedang berjalan..."):
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
                    st.error(f"Gagal training/evaluasi: {e}")
                    st.code(traceback.format_exc())
                    st.stop()

    if st.session_state.get("results") is None:
        st.warning("Belum ada hasil. Klik tombol Run atau aktifkan Auto-train di sidebar.")
    else:
        res = st.session_state["results"]

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("IndoBERT Test Accuracy", f"{res['bert']['test_accuracy']:.4f}")
        with c2: st.metric("GRU (PyTorch) Test Accuracy", f"{res['gru']['test_accuracy']:.4f}")
        with c3: st.metric("Split (train/val/test)", f"{res['sizes']['train']}/{res['sizes']['val']}/{res['sizes']['test']}")

        summary_df = pd.DataFrame([
            {"model": "IndoBERT Fine-tuning", "test_accuracy": res["bert"]["test_accuracy"], "test_macro_f1": res["bert"]["test_macro_f1"]},
            {"model": "GRU (PyTorch)",        "test_accuracy": res["gru"]["test_accuracy"],  "test_macro_f1": res["gru"]["test_macro_f1"]},
        ])
        st.dataframe(summary_df, use_container_width=True)

        with st.expander("📄 Detail evaluasi (classification report & confusion matrix)"):
            st.write("**IndoBERT — Classification Report (TEST)**")
            st.dataframe(pd.DataFrame(res["bert"]["report"]).T, use_container_width=True)
            st.write("**IndoBERT — Confusion Matrix (TEST)**")
            plot_confusion(res["bert"]["cm"], res["label_names"], "IndoBERT - Confusion Matrix (TEST)")

            st.write("**GRU (PyTorch) — Classification Report (TEST)**")
            st.dataframe(pd.DataFrame(res["gru"]["report"]).T, use_container_width=True)
            st.write("**GRU (PyTorch) — Confusion Matrix (TEST)**")
            plot_confusion(res["gru"]["cm"], res["label_names"], "GRU (PyTorch) - Confusion Matrix (TEST)")

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
        miss_df = pd.DataFrame({
            "dtype": df_raw.dtypes.astype(str),
            "missing": df_raw.isna().sum()
        }).sort_values("missing", ascending=False)
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


