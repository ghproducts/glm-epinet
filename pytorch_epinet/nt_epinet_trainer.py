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
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from datasets import load_dataset, ClassLabel

from peft import LoraConfig, TaskType, get_peft_model

# custom imports
from epinet import EpinetWrapper, EpinetConfig
from feature_fns import NT_feature_fn
from utils import ToySeqDataset, compute_metrics
from inference import predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
DATA_PATH = "pytorch_epinet/DATA/train.csv"
MAX_LEN = 512 # Max length of toneized sequences

# ---------------------------
# Trainer-compatible wrapper around EpinetWrapper
# ---------------------------

class HFEpinetSeqClassifier(nn.Module):
    def __init__(self, wrapper: EpinetWrapper, k_train=1, k_eval=8):
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
        logits = self.wrapper(batch, n_index_samples=K)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ---------------------------
# Build model  
# ---------------------------

def build_model_and_tokenizer(num_labels) -> (HFEpinetSeqClassifier, AutoTokenizer):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True, num_labels=num_labels
    )
    base.config.output_hidden_states = True

    # PEFT LoRA on backbone only (classifier & epinet train normally)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8, lora_alpha=16, lora_dropout=0.05,
        # match actual leaf Linear names; ESM-style usually has these:
        target_modules=["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
        bias="none",
    )
    base = get_peft_model(base, lora_cfg)

    # Epinet config (pooled hidden only)
    epi_cfg = EpinetConfig(
        num_classes=num_labels
   )

    wrapper = EpinetWrapper(base, NT_feature_fn, epi_cfg).to(DEVICE)
    model = HFEpinetSeqClassifier(wrapper, k_train=1, k_eval=8).to(DEVICE)
    return model, tok


# ---------------------------
# Main
# ---------------------------
def main():
    # build dataset
    ds = load_dataset("csv", data_files=DATA_PATH, split="train") 
    num_classes = len(set(ds['label']))  # assuming 'ID' is the label column 
    print(f"Number of classes: {num_classes}")
    ds = ds.cast_column("label", ClassLabel(num_classes=num_classes))
    ds = ds.remove_columns(["idx"])


    model, tok = build_model_and_tokenizer(num_classes)

    # Tokenize datasets
    def tokenize_fn(batch):
        return tok(
            batch["sequence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    ds_tok = ds.map(tokenize_fn, batched=True, desc="Tokenizing to fixed length")

    split = ds_tok.train_test_split(test_size=0.2, seed=42)           
    train, val = split["train"], split["test"]        

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


    # --- Warm-up forward (builds any lazy epinet cores), then re-freeze prior core just in case ---
    warm_loader = DataLoader(train, batch_size=8, shuffle=False)
    warm_batch = next(iter(warm_loader))
    with torch.no_grad():
        warm_inputs = {k: v.to(DEVICE) for k, v in warm_batch.items() if k != "labels"}
        _ = model(**warm_inputs)
    if hasattr(model.wrapper.epinet.prior_head, "core") and model.wrapper.epinet.prior_head.core is not None:
        model.wrapper.epinet.prior_head.core.requires_grad_(False)


    # --- Trainer ---
    args = TrainingArguments(
        output_dir="nt-epinet",
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
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    # Train + evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # Predict on val set to show logits
    # preds = trainer.predict(val_ds)
    # print("Logits (val):", preds.predictions)

    # Compute uncertainty
    K = getattr(model, "k_eval", 16)
    rows = predict(
        model=model,
        dataset=val,
        device=DEVICE,
        k_samples=K,
        batch_size=32,
        outfile="val_uncertainty.csv",     # or None
        save_logits_npz=None,              # or "val_logits_all.npz"
        use_amp=False,                     # True if you want faster inference on GPU
    )

    for j, r in list(enumerate(rows))[:10]:
        print(f"[val ex {j}] label={r['label']} pred={r['pred']} "
            f"conf={r['max_confidence']:.3f} U_tot={r['U_total']:.3f} "
            f"U_epi={r['U_epistemic']:.3f} U_ale={r['U_aleatoric']:.3f} "
            f"votes={r['vote_pct']:.3f}")

if __name__ == "__main__":
    main()
