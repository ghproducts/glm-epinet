# uncertainty_inference.py
from typing import Optional, List, Dict, Any
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils import compute_uncertainty  # you already have this

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
    save_logits_npz: Optional[str] = None,   # e.g., "val_logits_all.npz"
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

    for batch in loader:
        labels = batch.pop("labels").to(device)               # [B]
        inputs = {k: v.to(device) for k, v in batch.items()}

        with amp_ctx:
            # one base forward; K head forwards -> [K, B, C]
            logits_all = model.wrapper(inputs, n_index_samples=k_samples, return_all=True)

        unc = compute_uncertainty(logits_all)           # dict of [B]-tensors

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
