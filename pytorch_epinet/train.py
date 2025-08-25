# Builds and trains a NT+epinet model
# takes a single input training set and as many test sets as desired 

import os
import math
import random
import argparse
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from transformers import (
    TrainingArguments,
    Trainer,
)

from datasets import load_dataset, ClassLabel # type: ignore

# custom imports
from epinet import EpinetWrapper, EpinetConfig
from feature_fns import NT_feature_fn
from utils import compute_metrics, compute_uncertainty
from nt_epinet import build_model_and_tokenizer, predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    p = argparse.ArgumentParser(description="NT+Epinet training")
    p.add_argument("--input_csv", type=str, required=True,
                   help="CSV with a 'sequence' column and 'labels'.")
    p.add_argument("--output_dir", type=str, required=False, default = 'nt-epinet',
                   help = "output file location for predictions and checkpoint")
    p.add_argument("--num_classes", type=int, required=False,
                   help="Number of classes used during training.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for inference.")
    p.add_argument("--epochs", type = int, default = 1,
                   help="Number of epochs in training")
    p.add_argument("--max_len", type=int, default=512,
                   help="Tokenizer max sequence length")
    p.add_argument("--use_amp", action="store_true",
                   help="Enable autocast(fp16) on CUDA during inference.")
    return p.parse_args()

def main():
    args = parse_args()

    # build dataset
    ds = load_dataset("csv", data_files=args.input_csv, split="train") 

    if args.num_classes:
        num_classes = args.num_classes
    else:
        if 'labels' in ds.column_names:
            num_classes = len(set(ds['labels']))
        else:
            raise ValueError("'labels' column not found in dataset")
    print(f"Number of classes: {num_classes}")
    ds = ds.cast_column("labels", ClassLabel(num_classes=num_classes))

    model, tok = build_model_and_tokenizer(num_classes)

    # Tokenize datasets
    def tokenize_fn(batch):
        return tok(
            batch["sequence"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
        )

    ds_tok = ds.map(tokenize_fn, batched=True, desc="Tokenizing to fixed length")

    split = ds_tok.train_test_split(test_size=0.1, seed=42)           
    train, val = split["train"], split["test"]        

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    # --- Trainer ---
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        remove_unused_columns=False,   
        report_to=[],                  
    )

    trainer = Trainer(
        model=model,
        args=train_args,
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
        outfile="val_results.csv",     # or None
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