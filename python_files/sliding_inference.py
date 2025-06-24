from pathlib import Path
import pickle
import pandas as pd
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from config import TrainArgs
from datasets import Dataset
from nucleotide_transformer.pretrained import get_pretrained_model
from enn.networks.bert import make_head_enn, AgentConfig
from utils import calc_metrics
from Bio import SeqIO
from tqdm import tqdm



class SlidingGenomeDataset(Dataset):
    def __init__(self, fasta_folder, tokenizer, batch_size, window_size, step_size):
        self._tokenizer = tokenizer
        self._batch_size = batch_size

        self._window_size = window_size
        self._step_size = step_size

        self.sequences = []
        self.sequence_ids = []

        for fasta_path in Path(fasta_folder).glob("*.fasta"):
            sequences = []
            for record in SeqIO.parse(open(fasta_path), "fasta"):
                sequences.append((record.id, str(record.seq)))

            self.sequence_ids.append(fasta_path.stem)
            self.sequences.append(sequences)

        self._prepare_windows()
        self.num_batches = int(np.ceil(self._length / self._batch_size))

    def _prepare_windows(self):
        self._tokens_ids = []
        self._starts = []
        self._labels = []
        self._record_ids = []

        for seq_id, seq_list in zip(self.sequence_ids, self.sequences):
            for record_id, seq in seq_list:
                tokens, token_ids = self._tokenizer.tokenize(seq)

                # Flatten token_ids if nest ed
                if isinstance(token_ids[0], list):
                    token_ids = [item for sublist in token_ids for item in sublist]

                if len(token_ids) < self._window_size:
                    continue

                for start in range(0, len(token_ids) - self._window_size + 1, self._step_size):
                    window_ids = token_ids[start:start+self._window_size]
                    if len(window_ids) < self._window_size:
                        pad_len = self._window_size - len(window_ids)
                        window_ids += [self._tokenizer.pad_token_id] * pad_len

                    self._tokens_ids.append(window_ids)
                    self._starts.append(start)
                    self._labels.append(seq_id)
                    self._record_ids.append(record_id)

        self._tokens_ids = np.array(self._tokens_ids)
        self._starts = np.array(self._starts)
        self._labels = np.array(self._labels)
        self._record_ids = np.array(self._record_ids)
        self._length = len(self._labels)
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= self._length:
            raise StopIteration

        start = self._idx
        end = min(self._idx + self._batch_size, self._length)

        batch_tokens = self._tokens_ids[start:end]
        batch_labels = self._labels[start:end]
        batch_starts = self._starts[start:end]
        batch_record_ids = self._record_ids[start:end]

        tokens = jnp.asarray(batch_tokens, dtype=jnp.int32)

        self._idx = end
        return tokens, batch_labels, batch_starts, batch_record_ids

def sliding_inference(args: TrainArgs, step_size: int = 16):
    rng = hk.PRNGSequence(0)
    key = next(rng)

    m = args.model_args

    nt_params, forward_fn, tokenizer, _ = get_pretrained_model(
        model_name=m.model_name,
        embeddings_layers_to_save=(m.embeddings_layer,),
        max_positions=m.max_positions,
    )
    forward_fn = hk.transform(forward_fn)
    forward_apply = jax.jit(forward_fn.apply)

    dataset = SlidingGenomeDataset(
        fasta_folder=Path(args.input_path),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        window_size=m.max_positions,
        step_size=step_size
    )

    epinet_config = AgentConfig(index_dim=m.index_dim, prior_scale=m.prior_scale, hiddens=m.hiddens)
    epinet = make_head_enn(agent="epinet", num_classes=m.num_classes, agent_config=epinet_config)

    epinet_params_path = Path(args.params_path)
    with open(epinet_params_path, 'rb') as f:
        epinet_params = pickle.load(f)

    epinet_state = {}

    @jax.jit
    def epinet_forward(params, state, inputs, key):
        index = epinet.indexer(key)
        return epinet.apply(params, state, inputs, index)

    batch_fwd = jax.vmap(epinet_forward, in_axes=[None, None, None, 0])

    per_sequence_data = {}
    
    progress_bar = tqdm(dataset, total=dataset.num_batches, desc="Running Inference...")

    for batch_tokens, batch_labels, batch_starts, batch_record_ids in progress_bar:
        key = next(rng)
        #print(batch_tokens.shape)
        #print(batch_record_ids)
        embeddings = forward_apply(nt_params, key, batch_tokens)[f"embeddings_{m.embeddings_layer}"]
        pooled = jnp.mean(embeddings, axis=1)

        epinet_output, _ = batch_fwd(
            epinet_params, epinet_state, pooled,
            jax.random.split(key, args.epi_forwards)
        )
        #print(epinet_output.preds)
        metrics = calc_metrics(epinet_output.preds)

        for i, seq_id in enumerate(batch_labels):
            record = {
                "start_index": int(batch_starts[i]),
                "prediction": int(metrics["predicted_class"][i]),
                "total_uncertainty": float(metrics["normalized_total_uncertainty"][i]),
                "epistemic_uncertainty": float(metrics["normalized_epistemic_uncertainty"][i]),
                "aleatoric_uncertainty": float(metrics["normalized_aleatoric_uncertainty"][i]),
                "max_confidence": float(metrics["max_confidence"][i]),
                "vote_percentage": float(metrics["vote_percentage"][i]),
                "record_id": batch_record_ids[i],
            }

            if seq_id not in per_sequence_data:
                per_sequence_data[seq_id] = []

            per_sequence_data[seq_id].append(record)
    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for seq_id, records in per_sequence_data.items():
        df = pd.DataFrame(records)
        filename = f"{seq_id}_sliding_inference_{args.epi_forwards}.csv"
        df.to_csv(out_path / filename, index=False)
        print(f"Saved sliding inference results for {seq_id} to {out_path / filename}")

