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

def test_NT():
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name="500M_human_ref",
        embeddings_layers_to_save=(20,),
        max_positions=32,
    )
    forward_fn = hk.transform(forward_fn)

    # Get data and tokenize it
    sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
    tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    # Infer
    outs = forward_fn.apply(parameters, random_key, tokens)

    # Get embeddings at layer 20
    print(outs["embeddings_20"].shape)

def test_ENN():
    @dataclasses.dataclass
    class Config:
      num_batch: int = 1_000
      index_dim: int = 10
      num_index_samples: int = 10
      seed: int = 0
      prior_scale: float = 5.
      learning_rate: float = 1e-3
      noise_std: float = 0.1

    FLAGS = Config()



    #@title Create the regression experiment

    # Generate dataset
    dataset = regression_data.make_dataset()

    # Logger
    logger = TerminalLogger('supervised_regression')

    # Create Ensemble ENN with a prior network 
    enn = networks.MLPEnsembleMatchedPrior(
        output_sizes=[50, 50, 1],
        dummy_input=next(dataset).x,
        num_ensemble=FLAGS.index_dim,
        prior_scale=FLAGS.prior_scale,
        seed=FLAGS.seed,
    )

    # L2 loss on perturbed outputs 
    noise_fn = data_noise.GaussianTargetNoise(enn, FLAGS.noise_std, FLAGS.seed)
    single_loss = losses.add_data_noise(losses.L2Loss(), noise_fn)
    loss_fn = losses.average_single_index_loss(single_loss, FLAGS.num_index_samples)
     
    # Optimizer
    optimizer = optax.adam(FLAGS.learning_rate)

    # Aggregating different components of the experiment
    experiment = supervised.Experiment(
        enn, loss_fn, optimizer, dataset, FLAGS.seed, logger=logger)
        
    seed = 0
    init_key, loss_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    initial_loss = experiment.loss(next(dataset), init_key)
    print("initial ENN loss: ", initial_loss)    

    experiment.train(FLAGS.num_batch)

    final_loss = experiment.loss(next(dataset), loss_key)

    print("final ENN loss: ", final_loss)
      
def print_params(params, prefix=""):
    """
    Recursively print the network parameters in the order they are defined in the forward pass.
    Parameters:
        params (dict or ndarray): The network parameters or a single parameter array.
        prefix (str): A string prefix for formatting the output (used for indentation).
    """
    if isinstance(params, dict):  # If the current item is a dictionary
        for module_name in params:  # Iterate over the keys in the natural order
            print(f"{prefix}{module_name}:")
            print_params(params[module_name], prefix=prefix + "  ")  # Recurse into sub-modules
    else:  # If the current item is an array (e.g., weights or biases)
        print(f"{prefix}Shape: {params.shape}")
 

def calc_metrics(epi_out_logits):
    """
    epi_out_logits: [num_samples, batch_size, num_classes]
    """
    num_samples, batch_size, num_classes = epi_out_logits.shape

    # Softmax each sample -> per-class probabilities [samples, batch, classes]
    per_class_probs = jax.nn.softmax(epi_out_logits, axis=-1)

    # Ensemble mean probability for each sample [batch, classes]
    mean_probs = jnp.mean(per_class_probs, axis=0)

    # Total uncertainty (predictive entropy)
    predictive_entropy = -jnp.sum(mean_probs * jnp.log(mean_probs + 1e-9), axis=-1)

    # Aleatoric uncertainty (expected entropy over z)
    expected_entropy = jnp.mean(
        -jnp.sum(per_class_probs * jnp.log(per_class_probs + 1e-9), axis=-1),
        axis=0
    )

    # Epistemic uncertainty = predictive entropy - expected entropy
    epistemic_uncertainty = predictive_entropy - expected_entropy

    # Normalize all uncertainties to [0, 1] by dividing by log(num_classes)
    norm_factor = jnp.log(num_classes)
    normalized_total_uncertainty = predictive_entropy / norm_factor
    normalized_aleatoric_uncertainty = expected_entropy / norm_factor
    normalized_epistemic_uncertainty = epistemic_uncertainty / norm_factor

    # Predicted class (argmax of mean prediction)
    predicted_class = jnp.argmax(mean_probs, axis=-1)

    # Confidence (max prob)
    max_confidence = jnp.max(mean_probs, axis=-1)

    # Voting agreement: how many z-sampled logits agree with final prediction
    per_sample_preds = jnp.argmax(epi_out_logits, axis=-1)  # [samples, batch]
    votes_for_prediction = jnp.sum(per_sample_preds == predicted_class, axis=0)
    vote_percentage = votes_for_prediction / num_samples

    return {
        "predicted_class": predicted_class,
        "normalized_total_uncertainty": normalized_total_uncertainty,
        "normalized_epistemic_uncertainty": normalized_epistemic_uncertainty,
        "normalized_aleatoric_uncertainty": normalized_aleatoric_uncertainty,
        "max_confidence": max_confidence,
        "vote_percentage": vote_percentage,
    }

