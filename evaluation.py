from net2brain.utils.download_datasets import DatasetNSD_872
from utils.feature_extraction import FeatureExtractor as FeatureExtractor_1

from utils.ridge_regression import Ridge_Encoding, RidgeCV_Encoding
from net2brain.evaluations.plotting import Plotting
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
import io # Import io for BytesIO
from PIL import Image # Import PIL.Image
from evolution.architecture import Architecture

from net2brain.rdm_creation import RDMCreator
from net2brain.feature_extraction import FeatureExtractor

import argparse
import wandb
import matplotlib.pyplot as plt
import json # Added import

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", type=str, default="V2")
    parser.add_argument("--architecture_name", type=str, default="V2")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of runs to aggregate results over")
    parser.add_argument("--arch_config_file", type=str, default=None, help="Path to a JSON file defining the architecture layers.")
    parser.add_argument("--model_weights", type=str, default=None, help="File path to model weights as .pth")
    args = parser.parse_args()
    
    if args.roi == "V2":
        roi_path = "./V2combined"
    elif args.roi == "V4":
        roi_path = "./V4combined"
    elif args.roi == "IT":
        roi_path = "./ITcombined"
    else:
        raise ValueError(f"Invalid ROI: {args.roi}")

    paths_NSD_872 = DatasetNSD_872().load_dataset()
    stimuli_path = paths_NSD_872["NSD_872_images"]

    wandb.init(project="shallow-brain-evo-nas-nsd-results", name=args.architecture_name + "_" + args.roi)

    if args.arch_config_file:
        try:
            with open(args.arch_config_file, 'r') as f:
                layers = json.load(f)
                print(layers)
            print(f"Loaded architecture from {args.arch_config_file}")
        except FileNotFoundError:
            print(f"Error: Architecture file {args.arch_config_file} not found. Using default architecture.")
            raise FileNotFoundError(f"Architecture file {args.arch_config_file} not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.arch_config_file}. Using default architecture.")
            raise json.JSONDecodeError(f"Could not decode JSON from {args.arch_config_file}.", doc="", pos=0)
    else:
        print("No architecture file provided. Using default hardcoded architecture.")
        raise ValueError("No architecture file provided. Please provide a valid JSON file.")

    # Create the architecture
    architecture = Architecture(layers=layers)

    # Get the first available GPU ID
    if torch.cuda.is_available():
        gpu_id = 0  # Since we requested 1 GPU in SLURM, use index 0
    else:
        gpu_id = None
        print("No GPU available, using CPU")

    # Build the model (assuming you have a device defined)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model = architecture.build_model(device)

    if args.model_weights is not None:
        try:
            model.load_state_dict(torch.load(args.model_weights), strict=False) 
        except:
            print(f"Error: Could not load model weight file {args.model_weights}")

    all_layer_metrics_across_runs = []

    for run_idx in range(args.num_runs):
        print(f"Starting run {run_idx + 1}/{args.num_runs}")
        model = architecture.build_model(device)
        model.apply(architecture.init_weights)

        fx = FeatureExtractor_1(model=model, device=device, pretrained=True)
        features_1 = fx.extract(data_path=stimuli_path, save_path=f'features_run_{run_idx}_{args.architecture_name}_{args.roi}_{np.random.randint(0, 1000000)}')

        results_dataframe_cv = RidgeCV_Encoding(
            features=features_1,
            roi_path=roi_path,
            model_name="a",
            n_folds=3,
            trn_tst_split=0.8,
            n_components=100,       
            batch_size=64,
            return_correlations=False,
            save_path="results",
            alpha=1.0,
        )

        layerr = results_dataframe_cv[['Layer', 'R']]
        layerr = layerr.groupby('Layer').agg(['mean', 'std'])
        layerr.columns = ['_'.join(col).strip() for col in layerr.columns.values]

        all_layer_metrics_across_runs.append(layerr)

    if not all_layer_metrics_across_runs:
        print("No layer metrics were collected across runs.")
    else:
        combined_metrics_df = pd.concat(all_layer_metrics_across_runs)

        final_aggregated_stats = combined_metrics_df.groupby(level="Layer").agg(
            mean_R_mean=('R_mean', 'mean'),
            std_R_mean=('R_mean', 'std'),
            mean_R_std=('R_std', 'mean'),
            std_R_std=('R_std', 'std')
        )

        # Convert Layer index to numeric and sort for correct ascending order
        final_aggregated_stats.index = pd.to_numeric(final_aggregated_stats.index)
        final_aggregated_stats = final_aggregated_stats.sort_index()

        print("\nFinal Aggregated Statistics over", args.num_runs, "runs:")
        print(final_aggregated_stats)

        # Log the aggregated statistics as a W&B Table
        wandb_summary_table = wandb.Table(dataframe=final_aggregated_stats.reset_index())
        wandb.log({"final_summary_layer_stats": wandb_summary_table})
        print("Logged final_summary_layer_stats to W&B.")

        # --- Create and Log Box Plot with SEM ---
        if args.num_runs > 1: # Box plot and SEM make sense if there's more than 1 run
            # Prepare data for boxplot
            # Ensure combined_metrics_df.index are strings '0', '1', etc. before this step.
            # Group R_mean values by layer, ensuring numeric sorting of layers
            grouped_r_means_by_layer = combined_metrics_df.groupby(lambda x: int(x))['R_mean']
            sorted_layers_numeric = sorted(grouped_r_means_by_layer.groups.keys())
            
            boxplot_data = [grouped_r_means_by_layer.get_group(layer).tolist() for layer in sorted_layers_numeric]
            layer_labels_for_plot = [str(l) for l in sorted_layers_numeric]

            # Data for SEM error bars (from final_aggregated_stats, which is numerically indexed and sorted)
            means_r_mean_for_ebar = final_aggregated_stats['mean_R_mean'].values
            std_r_mean_for_ebar = final_aggregated_stats['std_R_mean'].values
            sem_r_mean = std_r_mean_for_ebar / np.sqrt(args.num_runs)
            
            x_positions = np.arange(1, len(sorted_layers_numeric) + 1) # Box positions are 1, 2, ...

            fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
            
            # Create the boxplot
            bp = ax.boxplot(boxplot_data, positions=x_positions, widths=0.6, patch_artist=True,
                            medianprops={'color': 'black'}, showfliers=True)
            
            # Style the boxes (optional)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')

            # Overlay SEM error bars
            # The error bars are centered on the mean of R_mean for each layer
            ax.errorbar(x_positions, means_r_mean_for_ebar, yerr=sem_r_mean,
                        fmt='o', color='red', ecolor='red', elinewidth=1.5, capsize=3,
                        label='Mean R_mean with SEM', markersize=5, zorder=3) # zorder to bring to front

            ax.set_xticks(x_positions)
            ax.set_xticklabels(layer_labels_for_plot)
            ax.set_xlabel("Layer")
            ax.set_ylabel("R_mean")
            ax.set_title(f"Distribution of R_mean per Layer across {args.num_runs} runs (with SEM of R_mean)")
            ax.legend()
            plt.tight_layout()

            # Save plot to a BytesIO buffer and log to W&B
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            pil_image = Image.open(img_buffer) # Create PIL image from buffer
            wandb.log({"R_mean_distribution_boxplot": wandb.Image(pil_image)})
            img_buffer.close() # Close the buffer once the PIL image is created
            plt.close(fig) # Close the figure to free memory
            print("Logged R_mean_distribution_boxplot to W&B.")
        elif args.num_runs == 1:
            print("Skipping box plot generation as num_runs is 1. Box plot requires multiple runs.")
        # --- End of Box Plot ---

if __name__ == "__main__":
    main()