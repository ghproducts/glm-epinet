#!/bin/bash
#SBATCH --account=rosenmriprj
#SBATCH --output=error_out/run_train_NT-%j.out
#SBATCH --error=error_out/run_train_NT-%j.err
#SBATCH --mem=28GB
#SBATCH --time 36:00:00
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

cd /ifs/groups/rosenMRIGrp/glh52/epinet_v2/python_files/

python3 -m pip install biopython

python3 update/NT_enn_HGT_test.py \
	--input_path /ifs/groups/rosenMRIGrp/glh52/epinet_v2/DATA/HGT \
	--output_path /ifs/groups/rosenMRIGrp/glh52/epinet_v2/output/HGT \
	--epochs 1 \
	--batch_size 64 \
	--eval_steps 10000000000000 \
