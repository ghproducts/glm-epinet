#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0

source ~/py390-ENN/bin/activate
cd /scratch/home/glh52/glm-epinet/python_files

python3 main.py \
    --input_path /scratch/home/glh52/glm-epinet/DATA/basic_sampled \
    --output_path /scratch/home/glh52/glm-epinet/models/basic_sampled \
    --epochs 1 \
    --batch_size 16 \
    --mode train \
    --out_name basic_sampled_database

