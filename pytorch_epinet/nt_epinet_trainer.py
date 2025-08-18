# nt_epinet_trainer.py
# Trainer-compatible NT + Epinet with PEFT LoRA (base-only).
# - Base: AutoModelForMaskedLM (we only use hidden states; no MLM loss)
# - Epinet: your EpinetWrapper from epinet.py (pooled hidden -> logits add-on)
# - Trainer: standard HF Trainer API, toy dataset for a quick smoke test.

import os
import math
import random
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, TaskType, get_peft_model

# custom imports
from epinet import EpinetWrapper, EpinetConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
NUM_LABELS = 2

# ---------------------------
# Base model + small classifier head
# ---------------------------
class NTWithHead(nn.Module):
    """
    Load the MLM backbone to match checkpoint shapes; we won't use its lm_head.
    We'll pool its last hidden state and pass through a small classifier.
    """
    def __init__(self, name: str = MODEL_NAME, num_classes: int = NUM_LABELS):
        super().__init__()
        self.base = AutoModelForMaskedLM.from_pretrained(
            name, trust_remote_code=True, output_hidden_states=True
        )
        H = self.base.config.hidden_size
        self.classifier = nn.Linear(H, num_classes)

def nt_feature_fn_seqclf(m: NTWithHead, batch: Dict[str, torch.Tensor]):
    """
    Return (base_logits, pooled_hidden) for EpinetWrapper.
    - base_logits: from our small classifier on pooled hidden states.
    - pooled_hidden: masked mean over sequence positions.
    """
    inputs = batch.data if hasattr(batch, "data") else batch
    out = m.base(**inputs)                      # has .hidden_states (list of [B, L, H])
    last = out.hidden_states[-1]                # [B, L, H]
    mask = inputs.get("attention_mask", None)   # [B, L] or None

    if mask is None:
        pooled = last.mean(dim=1)               # [B, H]
    else:
        msk = mask.unsqueeze(-1).to(last.dtype) # [B, L, 1]
        denom = msk.sum(dim=1).clamp_min(1e-6)  # [B, 1]
        pooled = (last * msk).sum(dim=1) / denom

    mu = m.classifier(pooled)                   # [B, C]
    return mu, pooled

# ---------------------------
# Trainer-compatible wrapper around EpinetWrapper
# ---------------------------
class HFEpinetSeqClassifier(nn.Module):
    """
    Makes EpinetWrapper look like a HF model: accepts (input_ids, attention_mask, labels)
    and returns dict(loss=..., logits=...).
    """
    def __init__(self, wrapper: EpinetWrapper, k_train: int = 1, k_eval: int = 8):
        super().__init__()
        self.wrapper = wrapper
        self.k_train = k_train
        self.k_eval = k_eval

    @property
    def cfg(self):
        return self.wrapper.cfg

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        K = self.k_train if self.training else self.k_eval
        logits = self.wrapper(batch, n_index_samples=K)   # [B, C]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

# ---------------------------
# Tiny toy dataset (torch Dataset)
# ---------------------------
class ToySeqDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, sequences: List[str], labels: List[int], max_len: Optional[int] = 512):
        enc = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(max_len, tokenizer.model_max_length) if max_len else tokenizer.model_max_length,
        )
        self.enc = {k: v for k, v in enc.items()}   # tensors
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item

# ---------------------------
# Build everything
# ---------------------------
def build_model_and_tokenizer() -> (HFEpinetSeqClassifier, AutoTokenizer):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Base+head
    base = NTWithHead(num_classes=NUM_LABELS)

    # PEFT LoRA on backbone only (classifier & epinet train normally)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8, lora_alpha=16, lora_dropout=0.05,
        # match actual leaf Linear names; ESM-style usually has these:
        target_modules=["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
        bias="none",
    )
    base.base = get_peft_model(base.base, lora_cfg)

    # Epinet config (pooled hidden only)
    epi_cfg = EpinetConfig(
        num_classes=NUM_LABELS
   )

    wrapper = EpinetWrapper(base, nt_feature_fn_seqclf, epi_cfg).to(DEVICE)
    model = HFEpinetSeqClassifier(wrapper, k_train=1, k_eval=8).to(DEVICE)
    return model, tok

# ---------------------------
# Metrics (F1 with sklearn if available; otherwise accuracy)
# ---------------------------
def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(axis=-1)
    labels = eval_pred.label_ids
    try:
        from sklearn.metrics import f1_score
        return {"f1_score": f1_score(labels, preds)}
    except Exception:
        import numpy as np
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

# ---------------------------
# Main
# ---------------------------
def main():
    model, tok = build_model_and_tokenizer()

    # --- Warm-up forward (builds any lazy epinet cores), then re-freeze prior core just in case ---
    with torch.no_grad():
        demo = tok(["ATTCCGATTCCGATTCCG", "GATTACA"], return_tensors="pt", padding=True, truncation=True)
        demo = {k: v.to(DEVICE) for k, v in demo.items()}
        _ = model(**demo)  # builds epinet MLP cores on first call
    # Ensure prior head is frozen after lazy build (harmless if already frozen)
    if hasattr(model.wrapper.epinet.prior_head, "core") and model.wrapper.epinet.prior_head.core is not None:
        model.wrapper.epinet.prior_head.core.requires_grad_(False)

    # --- Toy data ---
    sequences = [
        "ATTCCGATTCCGATTCCG",
        "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT",
        "GATTACA",
        "ATGCGTATGCGTATGCGT",
        "TTTTTTTTTTTTTTTTTT",
        "ACGTACGTACGTACGT",
    ]
    labels = [0, 1, 0, 1, 0, 1]  # toy binary
    # small split
    train_idx = [0, 1, 2, 3]
    val_idx   = [4, 5]

    train_ds = ToySeqDataset(tok, [sequences[i] for i in train_idx], [labels[i] for i in train_idx], max_len=256)
    val_ds   = ToySeqDataset(tok, [sequences[i] for i in val_idx],   [labels[i] for i in val_idx],   max_len=256)

    # --- Trainer ---
    args = TrainingArguments(
        output_dir="nt-epinet-toy",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_steps=10,
        remove_unused_columns=False,   # we pass dicts with exact keys; keep them
        report_to=[],                  # no wandb by default
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    # Train + evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # Predict on val set to show logits
    preds = trainer.predict(val_ds)
    print("Logits (val):", preds.predictions)

if __name__ == "__main__":
    main()
