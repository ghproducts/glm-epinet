import os
import math
import random
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, TaskType, get_peft_model

# custom imports
from epinet import EpinetWrapper, EpinetConfig
from feature_fns import NT_feature_fn



# loading NT ennn
def load_model_and_tokenizer(num_classes: int, checkpoint: str, device: torch.device):
    # Rebuild architecture exactly like training and load weights
    model, tok = build_model_and_tokenizer(num_classes)

    # dummy init
    with torch.no_grad():
        dummy = tok("ACGT", truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        dummy = {k: v.to(device) for k, v in dummy.items()}
        _ = model(**dummy)  # builds epinet internals

    state_path = os.path.join(checkpoint, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    sd = load_file(state_path)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()



    return model, tok





# Metrics 
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
    


@torch.no_grad()
def compute_uncertainty(epi_out_logits: torch.Tensor):
    """
    Accepts [K, B, C] (preferred) or [B, C].
    Returns a dict of [B]-shaped tensors (predicted_class is long).
    """
    if epi_out_logits.dim() == 2:
        epi_out_logits = epi_out_logits.unsqueeze(0)   # [1, B, C]
    S, B, C = epi_out_logits.shape

    # per-sample probabilities
    per_logp = F.log_softmax(epi_out_logits, dim=-1)   # [S,B,C]
    per_p = per_logp.exp()                              # [S,B,C]

    mean_p = per_p.mean(dim=0)                         # [B,C]
    eps = 1e-9

    # Total (predictive) entropy
    pred_ent = -(mean_p * (mean_p.clamp_min(eps).log())).sum(dim=-1)   # [B]

    # Expected entropy over z (aleatoric)
    per_ent = -(per_p * per_logp).sum(dim=-1)                           # [S,B]
    exp_ent = per_ent.mean(dim=0)                                       # [B]

    # Epistemic = total - aleatoric
    epi = pred_ent - exp_ent                                            # [B]

    # Normalize by log(C)
    norm = torch.log(torch.tensor(float(C), device=epi_out_logits.device))
    out = {
        "predicted_class": mean_p.argmax(dim=-1),                       # [B], long
        "normalized_total_uncertainty": pred_ent / norm,                # [B]
        "normalized_epistemic_uncertainty": epi / norm,                 # [B]
        "normalized_aleatoric_uncertainty": exp_ent / norm,             # [B]
        "max_confidence": mean_p.max(dim=-1).values,                    # [B]
    }

    # Voting agreement
    per_preds = epi_out_logits.argmax(dim=-1)                           # [S,B]
    votes = (per_preds == out["predicted_class"].unsqueeze(0)).sum(dim=0)
    out["vote_percentage"] = votes.float() / float(S)                   # [B]

    # Also return mean probs if you want to store them
    out["mean_probs"] = mean_p                                          # [B,C]
    return out

