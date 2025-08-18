# glm-epinet

This is a repo for using epistemic neural networks with genomic language models


The current implementation relies on two primary libraries: the Nucleotide Transformer, and the 
epinstemic neural network epinet implementation. Both are built on jax-haiku which also must be 
installed

So for this repo to be run, both need to be installed. I recommend installing them first, then 
installing the other dependencies from the .requirements, as there are a lot of slightly older
libraries that are used that may be updated to incompatible versions with installation of the 
epinet in particular.


Included is also functionality to implement the basic epinet module through pytorch, and utilities
to implement this with genomics language modeling. 
