#!/bin/bash

source ~/envs/py390-ENN/bin/activate
cd ~/Projects/glm-epinet/python_files

python3 main.py \
    --input_path ~/Projects/glm-epinet/DATA/HGT \
    --output_path ~/Projects/glm-epinet/models \
    --epochs 1 \
    --batch_size 1 \
    --mode train \
    --out_name training_run

