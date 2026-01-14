#!/bin/bash
#SBATCH --job-name=BEvoEval       # Job name
#SBATCH --output=nas_evo_%j.out         # Output file (%j expands to jobID)
#SBATCH --error=nas_evo_%j.err          # Error file (%j expands to jobID)
#SBATCH --mem=64G                   # Memory requirement
#SBATCH --time=12:00:00             # Time limit hrs:min:sec
#SBATCH --gres=gpu:1

source /home/$USER/miniconda3/etc/profile.d/conda.sh

# Activate your virtual environment if you're using one
conda activate ShalloBrainNAS

# Run the script
python3 evaluation.py --num_runs 50 --architecture_name IT_3_new --arch_config_file architectures/IT_3.json --roi V2