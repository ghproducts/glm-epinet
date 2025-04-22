from pathlib import Path
import pickle
import pandas as pd
import jax
import jax.numpy as jnp
import haiku as hk

from config import TrainArgs
from datasets import make_genome_dataset
from nucleotide_transformer.pretrained import get_pretrained_model
from enn.networks.bert import make_head_enn, AgentConfig
from utils import calc_metrics

def inference(args: TrainArgs):
    rng = hk.PRNGSequence(0)
    key = next(rng)

    m = args.model_args

    # Load NT model
    nt_params, forward_fn, tokenizer, _ = get_pretrained_model(
        model_name=m.model_name,
        embeddings_layers_to_save=(m.embeddings_layer,),
        max_positions=m.max_positions,
    )
    forward_fn = jax.jit(forward_fn)

    # Load dataset
    dataset, num_classes = make_genome_dataset(Path(args.input_path), tokenizer, args.batch_size)
    test_dataset = dataset['test']

    if args.num_classes:
        num_classes = args.num_classes

    # Load epinet
    epinet_config = AgentConfig(index_dim=m.index_dim, prior_scale=m.prior_scale, hiddens=m.hiddens)
    epinet = make_head_enn(agent="epinet", num_classes=num_classes, agent_config=epinet_config)

    epinet_params_path = Path(args.params_path)
    if not epinet_params_path.exists():
        raise FileNotFoundError(f"Epinet params not found: {epinet_params_path}")
    with open(epinet_params_path, 'rb') as f:
        epinet_params = pickle.load(f)

    epinet_state = {}

    @jax.jit
    def epinet_forward(params, state, inputs, key):
        index = epinet.indexer(key)
        return epinet.apply(params, state, inputs, index)

    batch_fwd = jax.vmap(epinet_forward, in_axes=[None, None, None, 0])

    # Run inference
    all_metrics = {
        "label": [], "enn_predictions": [],
        "total_uncertainty": [], "epistemic_uncertainty": [],
        "aleatoric_uncertainty": [], "max_value": [], "vote_percents": []
    }

    for batch_tokens, batch_labels in test_dataset:
        key = next(rng)
        embeddings = forward_fn(nt_params, key, batch_tokens)[f"embeddings_{m.embeddings_layer}"]
        pooled = jnp.mean(embeddings, axis=1)

        epinet_output, _ = batch_fwd(
            epinet_params, epinet_state, pooled,
            jax.random.split(key, args.epi_forwards)
        )

        metrics = calc_metrics(epinet_output.preds)

        all_metrics["label"].append(batch_labels)
        all_metrics["enn_predictions"].append(metrics["predicted_class"])
        all_metrics["total_uncertainty"].append(metrics["normalized_total_uncertainty"])
        all_metrics["epistemic_uncertainty"].append(metrics["normalized_epistemic_uncertainty"])
        all_metrics["aleatoric_uncertainty"].append(metrics["normalized_aleatoric_uncertainty"])
        all_metrics["max_value"].append(metrics["max_confidence"])
        all_metrics["vote_percents"].append(metrics["vote_percentage"])

    results_df = pd.DataFrame({key: jnp.concatenate(val) for key, val in all_metrics.items()})

    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)
    file_name = f"inference_results_{args.out_name}_{args.epi_forwards}.csv"
    results_df.to_csv(out_path / file_name, index=False)
    print(f"Results saved to {out_path / file_name}")
