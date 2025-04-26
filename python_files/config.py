# config.py
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelArgs:
    # NT model args
    model_name: str = "100M_multi_species_v2"
    max_positions: int = 33
    embeddings_layer: int = 21

    # Epinet model args
    index_dim: int = 100
    prior_scale: float = 1.0
    hiddens: List[int] = (100, 100)
    num_classes: Optional[int] = None

@dataclass
class TrainArgs:
    input_path: str = ""
    output_path: str = ""
    batch_size: int = 1
    epochs: int = 1
    out_name: str = "output"
    params_path: Optional[str] = None
    epi_forwards: int = 10
    num_classes: Optional[int] = None
    model_args: ModelArgs = ModelArgs()
