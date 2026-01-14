#!/bin/bash
#SBATCH --job-name=BEvoEval       # Job name
#SBATCH --output=logs/nas_evo_%j.out         # Output file (%j expands to jobID)
#SBATCH --error=logs/nas_evo_%j.err          # Error file (%j expands to jobID)
#SBATCH --gres=gpu:1

source /home/lkuhn/BrainEvo/.venv/bin/activate

# Run the script
python3 train_imagenet.py --arch_config_file architectures/CorNET.json --data_path /projects/datasets_ML/imagenet/ --batch_size 128 --epochs 200
