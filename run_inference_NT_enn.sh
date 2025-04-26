#!/bin/bash
#SBATCH --account=rosenmriprj
#SBATCH --output=error_out/inference_NT-%j.out
#SBATCH --error=error_out/inference_NT-%j.err
#SBATCH --mem=6GB
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1


module load python/gcc/3.9
module load cuda11.2/blas cuda11.2/fft cuda11.2/nsight cuda11.2/profiler cuda11.2/toolkit cudnn8.0-cuda11.2 cudnn8.1-cuda11.2 cutensor-cuda11.2
source /ifs/groups/rosenMRIGrp/glh52/venvs/py390-ENN2/bin/activate

#pip freeze > packages.txt

#working jax versioning stuff - keep for posterity
#python3 -m pip install jax==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
#python3 -m pip install jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#python3 -m pip install chex==0.1.6
#python3 -m pip install distrax==0.1.2
#python3 -m pip install rlax==0.1.4
#python3 -m pip install tensorflow==2.8.0
#python3 -m pip install tensorflow_probability==0.15.0
#pip install optax==0.1.3 --no-deps


cd /ifs/groups/rosenMRIGrp/glh52/epinet_v2/python_files/
for files in witheld.csv test_base.csv partial.csv #witheld_sampled.csv
do
	for forwards in 1 2 4 8 10 20 40
	do
		python3 NT_enn_pbsim_2.py \
			--input_path /ifs/groups/rosenMRIGrp/glh52/epinet_v2/DATA/TAXA/PBsim_long/full/$files \
			--output_path /ifs/groups/rosenMRIGrp/glh52/epinet_v2/output/TAXA-PBsim_long/full_5/inference_new \
			--params_path /ifs/groups/rosenMRIGrp/glh52/epinet_v2/output/TAXA-PBsim_long/full_5/epi_params_final.pkl \
			--batch_size 8 \
			--inference 1 \
			--num_classes 1344 \
			--epi_forwards $forwards \
			--out_name $files
	done
done
