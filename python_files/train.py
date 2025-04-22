from pathlib import Path
import pickle as pkl
import time
import sys
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import chex

#NT
from nucleotide_transformer.pretrained import get_pretrained_model

#enn
from enn.networks.bert import make_head_enn, AgentConfig

# custom
from datasets import make_genome_dataset
from config import TrainArgs

def train(args: TrainArgs):
    rng = hk.PRNGSequence(0)
    key = next(rng)

    m = args.model_args

    nt_params, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=m.model_name,
        embeddings_layers_to_save=(m.embeddings_layer,),
        max_positions=m.max_positions,
    )

    dataset, num_classes = make_genome_dataset(Path(args.input_path), tokenizer, args.batch_size)
    m.num_classes = num_classes
    train_dataset = dataset['train']

    epinet_config = AgentConfig(index_dim=m.index_dim, prior_scale=m.prior_scale, hiddens=m.hiddens)
    epinet = make_head_enn(agent="epinet", num_classes=m.num_classes, agent_config=epinet_config)

    first_batch, _ = next(iter(train_dataset))
    pooled = jnp.mean(forward_fn(nt_params, key, first_batch)[f"embeddings_{m.embeddings_layer}"], axis=1)
    epinet_index = epinet.indexer(key)
    epinet_params, epinet_state = epinet.init(rng=key, x=pooled, z=epinet_index)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(epinet_params)

    def loss_fn(epinet_params, epinet_state, batch_input, batch_labels, key):
        # grab epi indices
        keys = jax.random.split(key, 5) # controls number of samples
        
        indices = jax.vmap(epinet.indexer)(keys)

        # forward the network multiple times for indices
        epinet_output, epinet_state = jax.vmap(
        lambda idx: epinet.apply(
            epinet_params, epinet_state, batch_input, idx)
        )(indices)
        
        # Step 3: Compute cross-entropy loss for each output
        def compute_loss(output_preds):
            logits = jax.nn.log_softmax(output_preds)  # Convert logits to log probabilities
            labels = jax.nn.one_hot(batch_labels, logits.shape[-1])  # Convert targets to one-hot encoding
            xent = -jnp.sum(labels * logits, axis=-1)  # Cross-entropy calculation
            return jnp.mean(xent)  # Mean loss for the batch
        
        losses = jax.vmap(compute_loss)(epinet_output.preds)
        
        # Step 4: Average the losses across indices
        mean_loss = jnp.mean(losses)

        return mean_loss, epinet_state
        
    def backprop(grads, opt_state, epinet_params):
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(epinet_params, updates)
        return new_opt_state, new_params

    @jax.jit
    def train_step(batch_tokens, batch_labels, NT_params, epinet_params, epinet_state, opt_state, key):
        # step 1 - get NT embeddings
        embeddings = forward_fn(NT_params, key, batch_tokens)
        pooled = jnp.mean(embeddings[f"embeddings_{embedding_layer}"], axis=1) # mean pooling - this can be switched to max for further testing
        
        # step 2 - pass embeddings through epinet at multiple indices + get loss                 
        (loss, new_epinet_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            epinet_params, epinet_state, pooled, batch_labels, key)
            
        
        # step 3 - backprop + param update
        new_opt_state, new_params = backprop(grads, opt_state, epinet_params)
            
        return new_params, new_epinet_state, new_opt_state, loss
              

    batch_no = 0
    for epoch in range(args.epochs):
        for batch_tokens, batch_labels in train_dataset:
            key = next(rng)
            epinet_params, epinet_state, opt_state, loss = train_step(
                batch_tokens, batch_labels, nt_params, epinet_params, epinet_state, opt_state, key)

            if batch_no % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_no}, Loss: {loss:.4f}")
            batch_no += 1

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_path) / 'epi_params_final.pkl', 'wb') as f:
        pickle.dump(epinet_params, f)
    print(f"Model saved to {Path(args.output_path) / 'epi_params_final.pkl'}")
