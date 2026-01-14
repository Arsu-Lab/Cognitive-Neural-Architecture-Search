#!/bin/bash
#SBATCH --job-name=BEvoEval       # Job name
#SBATCH --output=eval_%j.out         # Output file (%j expands to jobID)
#SBATCH --error=eval_%j.err          # Error file (%j expands to jobID)
#SBATCH --gres=gpu:1

source .venv/bin/activate

# Run the script
python eval.py --arch_name IT --num_runs 20 --model_weights best_acc1_IT.pth