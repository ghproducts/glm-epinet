#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0

source ~/py390-ENN/bin/activate
cd /scratch/home/glh52/glm-epinet/python_files

for file in up_0_75 up_1_0 up_0_25 #test hgt high_hgt withheld_gene withheld_taxa withheld_genome
do	
	for forwards in 50 100 #1 2 4 8 10 25 50 100
	do
		python3 main.py \
		    --input_path /scratch/home/glh52/glm-epinet/DATA/darkmatter_test/$file.csv \
		    --output_path /scratch/home/glh52/glm-epinet/darkmatter \
		    --batch_size 8 \
		    --mode inference \
		    --out_name $file \
		    --step_size 16 \
		    --epi_forwards $forwards \
		    --params_path /scratch/home/glh52/glm-epinet/models/basic_sampled/epi_params_final.pkl \
		    --load_args /scratch/home/glh52/glm-epinet/models/basic_sampled/basic_sampled_database_args.pkl
	done
done
