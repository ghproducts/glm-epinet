#!/bin/bash

#export CUDA_VISIBLE_DEVICES=7

source ~/py390-ENN/bin/activate
cd /scratch/home/glh52/glm-epinet/python_files

for forwards in 1 2 4 8 10 20 40
do
	python3 main.py \
	    --input_path /scratch/home/glh52/glm-epinet/DATA/HGT/high_HGT_genomes \
	    --output_path /scratch/home/glh52/glm-epinet/inference_small \
	    --batch_size 4 \
	    --mode sliding_inference \
	    --out_name small_test \
	    --step_size 16 \
	    --epi_forwards $forwards \
	    --params_path /scratch/home/glh52/glm-epinet/models/epi_params_final.pkl \
	    --load_args /scratch/home/glh52/glm-epinet/models/training_run_args.pkl
done

