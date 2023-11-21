#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=eecs595f23_class
#SBATCH -t 0-8:00

# set up job
# module load python cuda
module load python3.10-anaconda/2023.03
pushd /home/chriskok/595_project
source venv/bin/activate
# source activate 595
pip install git+https://github.com/google-research/bleurt.git
pip install transformers[torch]
pip install -r requirements.txt

# run job
python finetune_trainer.py