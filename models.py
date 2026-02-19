# Models
import traceback
from config import _optional_import, _IMPORT_ERRORS

# =========================
# OPTIONAL IMPORTS
# =========================
torch         = _optional_import("torch",          "pip install torch")
transformers  = _optional_import("transformers",   "pip install transformers accelerate sentencepiece")
accelerate    = _optional_import("accelerate",     "pip install accelerate")

AutoTokenizer                      = None
AutoModelForSequenceClassification = None
DataCollatorWithPadding            = None
TrainingArguments                  = None
Trainer                            = None

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
        _IMPORT_ERRORS["transformers_subimports"] = (
            "pip install transformers", traceback.format_exc()
        )

# =========================
# WEIGHTED TRAINER
# =========================
if Trainer is not None:
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss
else:
    class WeightedTrainer:
        pass

# =========================
# DATASET & GRU
# =========================
if torch is not None:
    class TorchTextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.labels = labels
            self.enc    = tokenizer(list(texts), truncation=True, max_length=int(max_len), padding=False)

        def __len__(self): return len(self.labels)

        def __getitem__(self, idx):
            item = {k: self.enc[k][idx] for k in self.enc.keys()}
            item["labels"] = int(self.labels[idx])
            return item

    class GRUClassifier(torch.nn.Module):
        def __init__(self, emb_weight, pad_id, hidden_size, num_labels, dropout):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(emb_weight, freeze=True, padding_idx=pad_id)
            emb_dim  = emb_weight.shape[1]
            self.gru = torch.nn.GRU(input_size=emb_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            self.dropout = torch.nn.Dropout(dropout)
            self.fc      = torch.nn.Linear(hidden_size * 2, num_labels)

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).clamp(min=1).to("cpu")
                packed  = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                _, h = self.gru(packed)
            else:
                _, h = self.gru(x)
            return self.fc(self.dropout(torch.cat([h[0], h[1]], dim=1)))
else:
    class TorchTextDataset: pass
    class GRUClassifier:    pass