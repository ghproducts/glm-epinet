# nt_epinet_trainer.py
# Trainer-compatible NT + Epinet with PEFT LoRA (base-only).
# - Base: NT transformer w/ automodel clsasification
# - Epinet: your EpinetWrapper from epinet.py (pooled hidden -> logits add-on)
# - Trainer: standard HF Trainer API

import os
import math
import random
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from datasets import load_dataset, ClassLabel # type: ignore

from peft import LoraConfig, TaskType, get_peft_model # type: ignore

# custom imports
from epinet import EpinetWrapper, EpinetConfig
from feature_fns import NT_feature_fn
from utils import compute_metrics, compute_uncertainty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
DATA_PATH = "DATA/gene_taxa/taxa_out.csv"
MAX_LEN = 512 # Max length of toneized sequences

# ---------------------------
# Trainer-compatible EpinetWrapper
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
def build_model_and_tokenizer(num_labels: int) -> (HFEpinetSeqClassifier, AutoTokenizer):
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

    # Epinet config 
    epi_cfg = EpinetConfig(
        num_classes=num_labels
   )

    wrapper = EpinetWrapper(base, NT_feature_fn, epi_cfg)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=8).to(DEVICE)

    # build epinet internals
    with torch.no_grad():
        dummy = tok("ACGT", truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
        _ = model(**dummy)  # builds epinet internals

    return model, tok


# ------------------------------
# load model from checkpoint for inference
# ------------------------------
def load_model_and_tokenizer(num_classes: int, checkpoint: str):
    # Rebuild architecture exactly like training and load weights
    model, tok = build_model_and_tokenizer(num_classes)

    # dummy init
    with torch.no_grad():
        dummy = tok("ACGT", truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
        _ = model(**dummy)  # builds epinet internals

    state_path = os.path.join(checkpoint, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    sd = load_file(state_path)
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()

    return model, tok



#-----------------------
# inference function
#-----------------------

def _collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

@torch.no_grad()
def predict(
    model,
    dataset,
    device: torch.device,
    k_samples: int = 16,
    batch_size: int = 32,
    outfile: Optional[str] = "val_uncertainty.csv",
    save_logits_npz: Optional[str] = None, 
    use_amp: bool = False,
) -> List[Dict[str, Any]]:
    """
    One base forward + K epinet head forwards per batch (return_all=True),
    computes uncertainty from the full stack, and optionally saves outputs.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    rows: List[Dict[str, Any]] = []
    all_logits = []
    all_labels = []

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.type == "cuda") else torch.cpu.amp.autocast(enabled=False)

    for batch in tqdm(loader, desc="Predicting", total=len(loader)):
        labels = batch.pop("labels").to(device)               # [B]
        inputs = {k: v.to(device) for k, v in batch.items()}

        with amp_ctx:
            # one base forward; K head forwards -> [K, B, C]
            logits_all = model.wrapper(inputs, n_index_samples=k_samples, return_all=True)

        unc = compute_uncertainty(logits_all)           

        B = labels.size(0)
        for i in range(B):
            rows.append({
                "label": int(labels[i]),
                "pred": int(unc["predicted_class"][i]),
                "max_confidence": float(unc["max_confidence"][i]),
                "U_total": float(unc["normalized_total_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "vote_pct": float(unc["vote_percentage"][i]),
            })

        if save_logits_npz is not None:
            all_logits.append(logits_all.cpu())               # [K,B,C]
            all_labels.append(labels.cpu())

    if outfile is not None:
        outdir = os.path.dirname(outfile)
        if outdir:  
            os.makedirs(outdir, exist_ok=True)
        pd.DataFrame(rows).to_csv(outfile, index=False)
        print(f"[uncertainty] wrote {outfile} with {len(rows)} rows")

    if save_logits_npz is not None:
        import numpy as np
        # stack batches into [K, N, C] and [N]
        logits = torch.cat(all_logits, dim=1).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        np.savez_compressed(save_logits_npz, logits_all=logits, labels=labels)
        print(f"[uncertainty] saved logits stack to {save_logits_npz} (shape {logits.shape})")

    return rows





# ---------------------------
# Main
# ---------------------------
# def main():
#     # build dataset
#     ds = load_dataset("csv", data_files=DATA_PATH, split="train") 
#     #ds = ds.remove_columns(["idx"])
#     num_classes = len(set(ds['labels']))
#     num_classes = 437 # temp for testing
#     print(f"Number of classes: {num_classes}")
#     ds = ds.cast_column("labels", ClassLabel(num_classes=num_classes))
# 
# 
#     model, tok = build_model_and_tokenizer(num_classes)
# 
#     # Tokenize datasets
#     def tokenize_fn(batch):
#         return tok(
#             batch["sequence"],
#             truncation=True,
#             padding="max_length",
#             max_length=MAX_LEN,
#         )
# 
#     ds_tok = ds.map(tokenize_fn, batched=True, desc="Tokenizing to fixed length")
# 
#     split = ds_tok.train_test_split(test_size=0.2, seed=42)           
#     train, val = split["train"], split["test"]        
# 
#     train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#     val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# 
# 
#     # --- Trainer ---
#     args = TrainingArguments(
#         output_dir="nt-epinet",
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=2e-4,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=2,
#         logging_steps=10,
#         remove_unused_columns=False,   # we pass dicts with exact keys; keep them
#         report_to=[],                  # no wandb by default
#     )
# 
#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=train,
#         eval_dataset=val,
#         tokenizer=tok,
#         compute_metrics=compute_metrics,
#     )
# 
#     # Train + evaluate
#     trainer.train()
#     metrics = trainer.evaluate()
#     print("Eval metrics:", metrics)
# 
#     # Predict on val set to show logits
#     # preds = trainer.predict(val_ds)
#     # print("Logits (val):", preds.predictions)
# 
#     # Compute uncertainty
#     K = getattr(model, "k_eval", 16)
#     rows = predict(
#         model=model,
#         dataset=val,
#         device=DEVICE,
#         k_samples=K,
#         batch_size=32,
#         outfile="val_uncertainty.csv",     # or None
#         save_logits_npz=None,              # or "val_logits_all.npz"
#         use_amp=False,                     # True if you want faster inference on GPU
#     )
# 
#     for j, r in list(enumerate(rows))[:10]:
#         print(f"[val ex {j}] label={r['label']} pred={r['pred']} "
#             f"conf={r['max_confidence']:.3f} U_tot={r['U_total']:.3f} "
#             f"U_epi={r['U_epistemic']:.3f} U_ale={r['U_aleatoric']:.3f} "
#             f"votes={r['vote_pct']:.3f}")
# 
# if __name__ == "__main__":
#     main()
