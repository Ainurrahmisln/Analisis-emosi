# Training
import gc
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from models import (
    torch, transformers, accelerate,
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer,
    WeightedTrainer, TorchTextDataset, GRUClassifier
)

# =========================
# TRAINING / EVAL HELPERS
# =========================
def require_libs_or_stop():
    if torch is None:
        st.error("PyTorch (torch) belum siap / gagal import. Lihat Diagnostik di sidebar.")
        st.stop()
    if transformers is None or AutoTokenizer is None or Trainer is None or TrainingArguments is None:
        st.error("Transformers/Trainer belum siap. Pastikan: transformers + accelerate + sentencepiece.")
        st.stop()
    if accelerate is None:
        st.error("accelerate belum siap. Install: pip install accelerate")
        st.stop()

def run_all_training(df_in, text_col, label_col, model_name, max_len, seed,
                     bert_epochs, gru_epochs, gru_batch, fast_mode, max_train_samples):
    require_libs_or_stop()
    try: torch.set_num_threads(1)
    except Exception: pass

    data = df_in[[text_col, label_col]].dropna().copy()
    data[text_col]  = data[text_col].astype(str).str.strip()
    data[label_col] = data[label_col].astype(str).str.strip().str.lower()

    if max_train_samples and max_train_samples > 0 and len(data) > max_train_samples:
        data = data.sample(n=int(max_train_samples), random_state=seed).reset_index(drop=True)

    le = LabelEncoder()
    data["label_id"] = le.fit_transform(data[label_col])
    label_names = list(le.classes_)
    num_labels  = len(label_names)

    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=seed, stratify=data["label_id"])
    val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label_id"])

    classes = np.unique(train_df["label_id"].values)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["label_id"].values)
    class_weights_torch = torch.tensor(cw, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    effective_max_len = min(int(max_len), 96) if fast_mode else int(max_len)

    train_ds = TorchTextDataset(train_df[text_col].tolist(), train_df["label_id"].tolist(), tokenizer, effective_max_len)
    val_ds   = TorchTextDataset(val_df[text_col].tolist(),   val_df["label_id"].tolist(),   tokenizer, effective_max_len)
    test_ds  = TorchTextDataset(test_df[text_col].tolist(),  test_df["label_id"].tolist(),  tokenizer, effective_max_len)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)

    try: model.config.use_cache = False
    except: pass
    try: model.gradient_checkpointing_enable()
    except: pass

    if fast_mode:
        base = getattr(model, model.base_model_prefix, None)
        if base is not None:
            for p in base.parameters(): p.requires_grad = False

    use_cuda = bool(torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir="indobert_out", learning_rate=5e-5 if fast_mode else 2e-5,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, num_train_epochs=int(bert_epochs),
        weight_decay=0.01, eval_strategy="no", save_strategy="no",
        load_best_model_at_end=False, dataloader_num_workers=0,
        dataloader_pin_memory=False, logging_steps=50, seed=seed,
        report_to="none", fp16=bool(use_cuda),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_torch, model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=None, data_collator=data_collator,
    )
    trainer.train()

    test_pred   = trainer.predict(test_ds)
    bert_y_true = test_pred.label_ids
    bert_y_pred = np.argmax(test_pred.predictions, axis=-1)
    bert_acc    = float(accuracy_score(bert_y_true, bert_y_pred))
    bert_f1m    = float(f1_score(bert_y_true, bert_y_pred, average="macro", zero_division=0))
    bert_cm     = confusion_matrix(bert_y_true, bert_y_pred, labels=list(range(num_labels)))
    bert_report = classification_report(bert_y_true, bert_y_pred, target_names=label_names, output_dict=True, zero_division=0)

    gc.collect()
    if use_cuda:
        try: torch.cuda.empty_cache()
        except: pass

    # GRU
    pad_id     = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    emb_weight = model.get_input_embeddings().weight.detach().cpu()

    def encode_for_gru(texts):
        enc = tokenizer(list(texts), truncation=True, max_length=effective_max_len, padding="max_length", return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    X_train_ids, M_train = encode_for_gru(train_df[text_col].tolist())
    X_val_ids,   M_val   = encode_for_gru(val_df[text_col].tolist())
    X_test_ids,  M_test  = encode_for_gru(test_df[text_col].tolist())
    y_train = torch.tensor(train_df["label_id"].values, dtype=torch.long)
    y_val   = torch.tensor(val_df["label_id"].values,   dtype=torch.long)
    y_test  = torch.tensor(test_df["label_id"].values,  dtype=torch.long)

    device    = torch.device("cuda" if use_cuda else "cpu")
    gru_model = GRUClassifier(emb_weight, pad_id=pad_id,
                               hidden_size=64 if fast_mode else 128,
                               num_labels=num_labels,
                               dropout=0.2 if fast_mode else 0.3).to(device)
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights_torch.to(device))
    opt       = torch.optim.Adam(gru_model.parameters(), lr=1e-4)

    def make_loader(X, M, y, batch, shuffle=True):
        ds = torch.utils.data.TensorDataset(X, M, y)
        return torch.utils.data.DataLoader(ds, batch_size=int(batch), shuffle=shuffle)

    train_loader = make_loader(X_train_ids, M_train, y_train, gru_batch)
    val_loader   = make_loader(X_val_ids,   M_val,   y_val,   gru_batch, shuffle=False)
    test_loader  = make_loader(X_test_ids,  M_test,  y_test,  gru_batch, shuffle=False)

    best_val = float("inf"); bad = 0; patience = 1
    for _epoch in range(int(gru_epochs)):
        gru_model.train()
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(gru_model(xb, attention_mask=mb), yb)
            loss.backward(); opt.step()
        gru_model.eval()
        val_loss = 0.0; n = 0
        with torch.no_grad():
            for xb, mb, yb in val_loader:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                l = loss_fn(gru_model(xb, attention_mask=mb), yb)
                val_loss += float(l.item()) * yb.size(0); n += yb.size(0)
        val_loss /= max(n, 1)
        if val_loss < best_val - 1e-6: best_val = val_loss; bad = 0
        else:
            bad += 1
            if bad > patience: break

    gru_model.eval()
    all_pred = []; all_true = []
    with torch.no_grad():
        for xb, mb, yb in test_loader:
            xb, mb = xb.to(device), mb.to(device)
            all_pred.extend(torch.argmax(gru_model(xb, attention_mask=mb), dim=1).cpu().numpy().tolist())
            all_true.extend(yb.numpy().tolist())

    gru_acc    = float(accuracy_score(all_true, all_pred))
    gru_f1m    = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    gru_cm     = confusion_matrix(all_true, all_pred, labels=list(range(num_labels)))
    gru_report = classification_report(all_true, all_pred, target_names=label_names, output_dict=True, zero_division=0)

    return {
        "label_names": label_names,
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "bert": {"test_accuracy": bert_acc, "test_macro_f1": bert_f1m, "cm": bert_cm, "report": bert_report},
        "gru":  {"test_accuracy": gru_acc,  "test_macro_f1": gru_f1m,  "cm": gru_cm,  "report": gru_report},
        "meta": {
            "fast_mode": fast_mode, "max_train_samples": int(max_train_samples),
            "bert_epochs": int(bert_epochs), "gru_epochs": int(gru_epochs),
            "gru_batch": int(gru_batch), "torch_cuda": bool(use_cuda),
            "effective_max_len": int(effective_max_len), "bert_batch": 2, "grad_accum": 8,
        }
    }
