import argparse
import os
from typing import Optional

import torch
from safetensors.torch import load_file
import pandas as pd
from datasets import Dataset, load_dataset
from nt_epinet_trainer import build_model_and_tokenizer   # builds identical wrapper + tokenizer
from inference import predict                              # your predict() that computes uncertainty


def parse_args():
    p = argparse.ArgumentParser(description="NT+Epinet inference on a CSV of sequences.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a Trainer checkpoint dir containing pytorch_model.bin")
    p.add_argument("--input_csv", type=str, required=True,
                   help="CSV with at least a 'sequence' column. 'labels' optional (dummy will be added if missing).")
    p.add_argument("--num_classes", type=int, required=True,
                   help="Number of classes used during training (must match the trained head).")
    p.add_argument("--output_csv", type=str, default="inference_results.csv",
                   help="Where to save predictions + uncertainties (CSV). Use None to skip.")
    p.add_argument("--save_logits_npz", type=str, default=None,
                   help="Optional .npz path to save stacked logits [K,N,C] and labels [N].")
    p.add_argument("--k_samples", type=int, default=16,
                   help="Number of epinet index samples for uncertainty.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for inference.")
    p.add_argument("--max_len", type=int, default=512,
                   help="Tokenizer max sequence length (should match training).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cpu or cuda (default picks automatically).")
    p.add_argument("--use_amp", action="store_true",
                   help="Enable autocast(fp16) on CUDA during inference.")
    return p.parse_args()


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


def build_dataset(csv_path: str, tok, max_len: int) -> Dataset:
    ds = load_dataset("csv", data_files=csv_path, split="train")
    if "labels" not in ds.column_names:
        ds = ds.add_column("labels", [0] * len(ds))  # dummy

    def tokenize_fn(batch):
        return tok(
            batch["sequence"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    ds_tok = ds.map(tokenize_fn, batched=True, desc="Tokenizing to fixed length")
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds_tok

def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1) Build model + tokenizer and load checkpoint
    model, tok = load_model_and_tokenizer(args.num_classes, args.checkpoint, device)

    # 2) Prepare dataset
    ds_tok = build_dataset(args.input_csv, tok, args.max_len)

    # 3) Run inference with uncertainty
    outfile: Optional[str] = args.output_csv if args.output_csv.lower() != "none" else None
    rows = predict(
        model=model,
        dataset=ds_tok,
        device=device,
        k_samples=args.k_samples,
        batch_size=args.batch_size,
        outfile=outfile,
        save_logits_npz=args.save_logits_npz,
        use_amp=args.use_amp,
    )

    # 4) Small summary
    n = len(rows)
    preds = sum(int(r["pred"]) == int(r["label"]) for r in rows)
    print(f"[inference] completed on {n} samples.")
    if "labels" in ds_tok.column_names:
        acc = preds / max(1, n)
        print(f"[inference] (labels present) crude accuracy={acc:.3f}")
    if outfile is not None:
        print(f"[inference] wrote CSV to: {outfile}")
    if args.save_logits_npz is not None:
        print(f"[inference] saved logits stack to: {args.save_logits_npz}")


if __name__ == "__main__":
    main()