from transformers import AutoModelForSequenceClassification

# ---------------------------
# This file contains feature functions for EpinetWrapper
# ---------------------------

def NT_feature_fn(m: AutoModelForSequenceClassification, batch):
    inputs = batch.data if hasattr(batch, "data") else batch
    out = m(**inputs, output_hidden_states=True)
    mu = out.logits                                   # [B, C]
    last = out.hidden_states[-1]                      # [B, L, H]
    mask = inputs.get("attention_mask", None)
    if mask is None:
        pooled = last.mean(dim=1)                     # [B, H]
    else:
        msk = mask.unsqueeze(-1).to(last.dtype)       # [B, L, 1]
        pooled = (last * msk).sum(dim=1) / msk.sum(dim=1).clamp_min(1e-6)
    return mu, pooled
