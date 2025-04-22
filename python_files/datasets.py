#standard
import dataclasses
import pickle
import argparse
import os
import re
from glob import glob
import numpy as np
import pandas as pd
from typing import Callable, NamedTuple
import time
import sys
from Bio import SeqIO
import random

#nn stuff
import optax
import chex
import haiku as hk
import jax
import jax.numpy as jnp

#NT
from nucleotide_transformer.pretrained import get_pretrained_model

#enn imports
import enn
from enn import losses
from enn import networks
from enn import supervised
from enn import base
from enn import data_noise
from enn import utils
from enn import losses
from enn.loggers import TerminalLogger
from enn.supervised import classification_data
from enn.supervised import regression_data
from enn.networks.bert import make_head_enn, AgentConfig

#custom
from utils import calc_metrics

import numpy as np
import jax.numpy as jnp
import os
import pandas as pd

class Dataset:
    """An iterator over a numpy array, revealing batch_size elements at a time."""

    def __init__(self, x, y, batch_size: int, tokenizer):
        if len(x) != len(y):
            raise ValueError("Input data x and labels y must have the same length.")

        self._x = np.array(x)
        self._y = np.array(y)
        self._batch_size = batch_size
        self._tokenizer = tokenizer

        self._tokenized = False  # Flag to track whether tokenization has been done
        self._tokens_ids = []
        self._tokens_str = []

        self._idx = 0
        self._length = len(self._x)

    def tokenize(self):
        self._tokens_ids = []
        self._tokens_str = []

        for seq in self._x:
            tokens, token_ids = self._tokenizer.tokenize(seq)
            pad_len = self._tokenizer.fixed_length - len(token_ids)
            if pad_len > 0:
                token_ids += [self._tokenizer.pad_token_id] * pad_len
                tokens += [self._tokenizer.pad_token] * pad_len
            elif pad_len < 0:
                token_ids = token_ids[:self._tokenizer.fixed_length]
                tokens = tokens[:self._tokenizer.fixed_length]

            self._tokens_str.append(tokens)
            self._tokens_ids.append(token_ids)

        self._tokenized = True

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if not self._tokenized:
            raise RuntimeError("You must call `.tokenize()` before using the Dataset.")

        if self._idx >= self._length:
            raise StopIteration

        start = self._idx
        end = min(self._idx + self._batch_size, self._length)

        batch_tokens = self._tokens_ids[start:end]
        batch_y = self._y[start:end]

        tokens = jnp.asarray(batch_tokens, dtype=jnp.int32)

        self._idx = end
        return tokens, batch_y


def make_dataset(input_path, tokenizer, batch_size):
    dataset = {}

    for split in ['train', 'test', 'dev']:
        file_name = f"{split}.csv"
        file_path = os.path.join(input_path, file_name)
        
        print("checking:", file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected file '{file_name}' not found in '{input_path}'.")

        data = pd.read_csv(file_path)
        if split == 'train':
            data = data.sample(frac=1, random_state=42)

        if data.isna().any().any():
            print(f"Found rows with missing values in {split}")
            data = data.dropna()
            print(f"Dropped NaN rows from {split}")

        sequences = data['sequence'].tolist()
        labels = data['label'].tolist()

        if split == 'train':
            num_classes = len(set(labels))

        dataset[split] = Dataset(sequences, labels, batch_size, tokenizer)
        dataset[split].tokenize()  # Explicit tokenization

    return dataset, num_classes



class Genome_Dataset(Dataset):
    """
    Dataset that utilizes full genomes rather than pre-defined chunks

    """

    def __init__(self, x, y, batch_size: int, tokenizer):
        super().__init__(x, y, batch_size, tokenizer) 

    def tokenize(self):
        # Tokenize the entire dataset at initialization
        # self._tokenized_data = tokenizer.batch_tokenize(self._x)
        self._tokens_ids = []
        self._tokens_str = []
        temp_labels = []

        for seq, label in zip(self._x, self._y):
            tokens, token_ids = self._tokenizer.tokenize(seq) # tokenize entire genome first
            val = len(tokens) // self._tokenizer.fixed_length # num of seqs in each genome

            tokens = np.array_split(tokens[:val*self._tokenizer.fixed_length], val)
            token_ids = np.array_split(token_ids[:val*self._tokenizer.fixed_length], val)
            
            self._tokens_str.extend(tokens)
            self._tokens_ids.extend(token_ids)
            temp_labels.extend([label]*val)
        
        self._y = temp_labels
        self._idx = 0  # Initialize the starting index
        self._length = len(self._y)

    def randomize(self):
        """
        Randomize the order of everything
        """
        combined = list(zip(self._y, self._tokens_str, self._tokens_ids))
        random.shuffle(combined)
        a_shuf, b_shuf, c_shuf = zip(*combined)
        self._y = list(a_shuf)
        self._tokens_str = list(b_shuf)
        self._tokens_ids = list(c_shuf)


def make_genome_dataset(input_path, tokenizer, batch_size):
    """
    Create dataset direcctly from labeled fasta files
    Files must be in the following folders:
    train - test - dev

    labels will be taken directly from the fasta IDs,
    so they should be in the following format:
    >1
    <SEQ>

    sequences are tokenized sequentially, if a sequence is 
    too short it is thrown out
    
    """
    dataset = {}

    for split in ['train', 'test', 'dev']:
        folder_path = os.path.join(input_path, split)
        
        genomes = []
        labels = []
        
        for entry in os.listdir(folder_path):
            if entry.endswith('.fasta'):
                file_path = os.path.join(folder_path, entry)
            else:
                continue
            fasta_sequences = SeqIO.parse(open(file_path),'fasta')
            
            for fasta in fasta_sequences:
                labels.append(fasta.id) 
                genomes.append(str(fasta.seq))

        # make database from genomes
        dataset[split] = Genome_Dataset(genomes, labels, batch_size, tokenizer)
        dataset[split].tokenize()  # Explicit tokenization
        if split == 'train':
            num_classes = len(set(labels))

    dataset['train'].randomize()

    return dataset, num_classes
    


