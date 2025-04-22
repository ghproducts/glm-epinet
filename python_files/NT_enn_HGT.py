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
import datasets


def train(args):
    '''
    runs the main training for nucleotide transformer + enn
    '''
    devices = jax.devices("gpu")
    
    # Initialize random key
    rng = hk.PRNGSequence(0)
    #random_key = jax.random.PRNGKey(0)
    
    # define NT model
    NT_name = "100M_multi_species_v2"
    embedding_layer = 21
    
    NT_params, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=NT_name,
        embeddings_layers_to_save=(embedding_layer,),
        max_positions=50,
    )
    
    print(tokenizer.__class__.__name__)
    print(tokenizer._k_mers)

    
    forward_fn = hk.transform(forward_fn)
    forward_fn = forward_fn.apply
    
    # grab datasets
    dataset, num_classes = datasets.make_genome_dataset(args.input_path, tokenizer, args.batch_size)
    print(f"running training with {num_classes} labels")
    train_dataset = dataset['train']
    dev_dataset = dataset['dev']
    
    # define epinet model
    epinet_config = AgentConfig(index_dim=100, prior_scale=1.0, hiddens=[100, 100])
    epinet = make_head_enn(agent="epinet", num_classes=num_classes, agent_config=epinet_config)
    
    # sample input for enn (run through NT)
    first_batch, first_labels = next(train_dataset)
    print(first_batch.shape)
    key = next(rng)
    embeddings = forward_fn(NT_params, key, first_batch)
    pooled = jnp.mean(embeddings[f"embeddings_{embedding_layer}"], axis=1)
    
    # init epinet
    epinet_index = epinet.indexer(key)
                                              
    epinet_params, epinet_state = epinet.init(rng = key, 
                                              x = pooled,
                                              z = epinet_index)

    # init optimizer
    optimizer = optax.adam(1e-3) 
    opt_state = optimizer.init(epinet_params)
    
    #exit() # temp just for testing
    
    # this func requires the epinet to be defined already before working - for inference
    def epinet_forward_fn(params: hk.Params,
                state: hk.State,
                inputs: chex.Array,
                key: chex.PRNGKey,):
      index = epinet.indexer(key)
      out, state = epinet.apply(params, state, inputs, index)
      return out, state

    # Batched forward at multiple random indices - for inference
    batch_fwd = jax.jit(jax.vmap(epinet_forward_fn, in_axes=[None, None, None, 0]))
    
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
              
    def evaluate(dataset, rng, NT_params, epinet_params, epinet_state):
        start = time.time()
        total_correct = 0
        total_samples = 0
        for dev_batch_tokens, dev_batch_labels in dataset:
            key = next(rng)
            embeddings = forward_fn(NT_params, key, dev_batch_tokens)
            pooled = jnp.mean(embeddings[f"embeddings_{embedding_layer}"], axis=1) # mean pooling - this can be switched to max for further testing
            #continue
            epinet_output, _ = batch_fwd(epinet_params, 
                                         epinet_state, 
                                         pooled,
                                         jax.random.split(key, 10)) # 10 forwards for batch
          
            mean_predictions = jnp.mean(epinet_output.preds, axis=0) 
            predicted_labels = jnp.argmax(mean_predictions, axis=-1)  # Get predicted class
            correct_predictions = jnp.sum(predicted_labels == dev_batch_labels)  # Compare to true labels
            
            # Accumulate correct predictions and total samples
            total_correct += correct_predictions
            total_samples += len(dev_batch_labels)
            #total_samples += 1
            
        dev_accuracy = total_correct / total_samples
        end = time.time()
        length = end - start
        print(f"dev set accuracy: {dev_accuracy:.4f}  eval time {length:.4f} ")
        return dev_accuracy
    
    # Iterate over a dataset - begin training
    batch_no = 0
    max_dev_acc = 0
    
    
    for i in range(args.epochs):
        for batch_tokens, batch_labels in train_dataset:    
            key = next(rng)
            # new way
            new_epinet_params, new_epinet_state, new_opt_state, loss = train_step(batch_tokens, 
                                                                                  batch_labels, 
                                                                                  NT_params, 
                                                                                  epinet_params, 
                                                                                  epinet_state, 
                                                                                  opt_state, 
                                                                                  key)
            opt_state = new_opt_state
            epinet_params = new_epinet_params
            epinet_state = new_epinet_state
            
            # temp end
            if batch_no%50==0:
                print(f"Epoch {i} batch number {batch_no} train loss: {loss:.4f}")
                
            # step 5 - evaluate periodically
            if batch_no%args.eval_steps==0:
                continue # just for now, I do not evaluation at the current time
                dev_acc = evaluate(dev_dataset, rng, NT_params, epinet_params, epinet_state)
                if max_dev_acc < dev_acc:
                    max_dev_acc = dev_acc
                    params_path = os.path.join(args.output_path, f'epi_params_{batch_no}.pkl')
                    with open(params_path, 'wb') as f:
                        pickle.dump(epinet_params, f)
                sys.stdout.flush()

            batch_no += 1
    
    # save params
    params_path = os.path.join(args.output_path, 'epi_params_final.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(epinet_params, f)

def inference(args):
    '''
    runs inference on input files
    '''    
    
    # Initialize random key
    rng = hk.PRNGSequence(0)
    
    # define NT model
    NT_name = "100M_multi_species_v2"
    embedding_layer = 21
    
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=NT_name,
        embeddings_layers_to_save=(embedding_layer,),
        max_positions=1001,
    )
    
    
    
    forward_fn = hk.transform(forward_fn)
    forward_fn = jax.jit(forward_fn.apply)
    
    # load data
    #file_path = os.path.join(args.input_path, "test.csv") 
    file_path = args.input_path
    print("checking: ", file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected file '{file_path}' not found'.")

    data = pd.read_csv(file_path)
    sequences = data['sequence'].tolist()
    labels = data['label'].tolist()
    num_classes = args.num_classes
    if num_classes == None:
        num_classes = len(set(labels))
        
    print(f"running inference with {num_classes} labels")
    dataset = Dataset(sequences, labels, args.batch_size, tokenizer)
    
    
    # define epinet model
    epinet_config = AgentConfig(index_dim=100, prior_scale=1.0, hiddens=[100, 100])
    epinet = make_head_enn(agent="epinet", num_classes=num_classes, agent_config=epinet_config)
    
    # init epinet
    epinet_state = {}
    with open(args.params_path, 'rb') as f:
        epinet_params = pickle.load(f)
        
    print(f'loaded epinet params from {args.params_path}')
    
    def epinet_forward_fn(params: hk.Params,
                state: hk.State,
                inputs: chex.Array,
                key: chex.PRNGKey,):
      index = epinet.indexer(key)
      out, state = epinet.apply(params, state, inputs, index)
      return out, state

    # Batched forward at multiple random indices - for inference
    batch_fwd = jax.jit(jax.vmap(epinet_forward_fn, in_axes=[None, None, None, 0]))
    
   
    labels = []
    predicted_classes = []
    total_uncertainty = [] 
    epistemic_uncertainty = []
    aleatoric_uncertainty = []
    max_confidence = []
    vote_percents = []
    
    for batch_tokens, batch_labels in dataset:
        key = next(rng)
        
        embeddings = forward_fn(parameters, key, batch_tokens)
        pooled = jnp.mean(embeddings[f"embeddings_{embedding_layer}"], axis=1) # mean pooling - this can be switched to max for further testing

        epinet_output, _ = batch_fwd(epinet_params, 
                                     epinet_state, 
                                     pooled,
                                     jax.random.split(key, args.epi_forwards)) # number of forwards for batch
        
        metrics = calc_metrics(epinet_output.preds)

        batch_predicted_classes = metrics["predicted_class"]
        batch_normalized_total_uncertainty = metrics["normalized_total_uncertainty"]
        batch_normalized_epistemic_uncertainty = metrics["normalized_epistemic_uncertainty"]
        batch_normalized_aleatoric_uncertainty = metrics["normalized_aleatoric_uncertainty"]
        batch_max_confidence = metrics["max_confidence"]
        batch_vote_percents = metrics["vote_percentage"]

        labels.append(batch_labels)
        predicted_classes.append(batch_predicted_classes)
        total_uncertainty.append(batch_normalized_total_uncertainty)
        epistemic_uncertainty.append(batch_normalized_epistemic_uncertainty)
        aleatoric_uncertainty.append(batch_normalized_aleatoric_uncertainty)
        max_confidence.append(batch_max_confidence)
        vote_percents.append(batch_vote_percents)

    #print(jnp.concatenate(labels).shape)
    #print(jnp.concatenate(labels).shape)
    #print(jnp.concatenate(final_variances).shape)
    #print(jnp.concatenate(class_variances).shape)
    #print(jnp.concatenate(entropies).shape)
    #print(jnp.concatenate(normalized_entropies).shape)
    #print(jnp.concatenate(normalized_variances).shape)
      
        
    # Save results to CSV
    results_df = pd.DataFrame({
        'label': jnp.concatenate(labels),
        'enn_predictions': jnp.concatenate(predicted_classes),
        'total_uncertainty': jnp.concatenate(total_uncertainty),
        'epistemic_uncertainty': jnp.concatenate(epistemic_uncertainty),
        'aleatoric_uncertainty': jnp.concatenate(aleatoric_uncertainty),
        'max_value': jnp.concatenate(max_confidence),
        'vote_percents': jnp.concatenate(vote_percents)
    })
    
    #files = os.listdir(args.output_path)
    #file_count = len(files)
    file_name = 'inference_results_' + args.out_name[:-4] + '_' + str(args.epi_forwards)
    
    results_df.to_csv(f'{args.output_path}/{file_name}.csv', index=False)
    print(f'Inference results saved to {args.output_path}/{file_name}.csv')
    sys.stdout.flush()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train epinet on embeddings and logits - nucleotide trasformer version.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input of the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the final output")
    parser.add_argument("--out_name", type=str, default = "filename",required=False, help="optinal output file name")
    parser.add_argument("--tester", type=int, default=0, help="Sets the program in testing mode - this is just to verify that it is working")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training or inference. Default is 1")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training iterations. Default is 1")
    parser.add_argument("--inference", type=int, default=0, help="Are you running inference. Default is 0")
    parser.add_argument("--eval_steps", type=int, default=100, help="how many steps to run before evaluating")
    parser.add_argument("--params_path", type=str, required=False, help="model path for epinet if inference is being performed")
    parser.add_argument("--epi_forwards", type=int, default=10, help="model path for epinet if inference is being performed")
    parser.add_argument("--num_classes", type=int, default=None, help="just in case you gotta do this")

    args = parser.parse_args()
    if args.tester:    
        print("testing nucleotide transformer...")
        test_NT()
        
        print("testing enn...")
        test_ENN()
    elif args.inference:
        print("running inference...")
        inference(args)
    else:
        print("begin training...")
        train(args)
    
