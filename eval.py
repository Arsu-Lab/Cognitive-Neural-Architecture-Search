
import argparse
from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import shutil

import uuid

from evolution.architecture import Architecture
from net2brain.utils.download_datasets import DatasetNSD_872
from utils.feature_extraction import FeatureExtractor as FeatureExtractor_1
from architectures import get_model_by_archname

from net2brain.rdm_creation import RDMCreator
from net2brain.evaluations.rsa import RSA
from net2brain.feature_extraction import FeatureExtractor

from utils.ridge_regression import Ridge_Encoding, RidgeCV_Encoding


def main():
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_name", type=str, default="V2")
    parser.add_argument("--model_weights", type=str, default="best_acc1_V2.pth")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of runs to aggregate results over")
    args = parser.parse_args()

    paths_NSD_872 = DatasetNSD_872().load_dataset()
    stimuli_path = paths_NSD_872["NSD_872_images"]

    model, init_weights = get_model_by_archname(args.arch_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arch_name == "CorNetS":
        layer_names = [
            "V1",
            "V2",
            "V4",
            "IT",
        ]
    elif args.arch_name == "AlexNet":
        layer_names = [
            "features.0",
            "features.3",
            "features.6",
            "features.8",
            "features.10"
        ]
    elif args.arch_name == "VGG16":
        layer_names = [
            f"features.{i}" for i in range(20,31)
        ]
    else:
        layer_names = [
            f"{name}" for name, layer in model.named_children()
            #if not isinstance(layer, nn.ReLU)
        ]

    print(f"Layers that will be observed")
    for name in layer_names:
        print(name)

    layerrs = []
    """
    for roi_path in ["V2combined", "V4combined", "ITcombined"]:
        print(f"Currently running random model on ROI: {roi_path}")
        for _ in tqdm(range(args.num_runs)):
            model.apply(init_weights)

            fx = FeatureExtractor_1(model=model, device='cuda', pretrained=True)
            features_1 = fx.extract(data_path=stimuli_path, save_path=f'{np.random.randint(0, 100000)}', layers_to_extract=layer_names)

            results_dataframe_cv = RidgeCV_Encoding(
                features=features_1,
                roi_path=roi_path,
                model_name="a",
                n_folds=3,
                trn_tst_split=0.8,
                n_components=100,       
                batch_size=512,
                return_correlations=False,
                save_path="results",
                alpha=1.0
            )

            layerrs.append(results_dataframe_cv[['Layer', 'R']])
            
        meaned_layerrs = [layerrs[i].groupby('Layer').agg(['mean', 'std']).to_numpy() for i in range(len(layerrs))]

        with open(f"fin_results/{args.arch_name}_{roi_path}.npy", 'wb') as f:
            np.save(f, np.array(meaned_layerrs))
            print(f"Saved fin_results/{args.arch_name}_{roi_path}.npy")

    """
    for roi_path in ["V2combined", "V4combined", "ITcombined"]:
        print(f"Currently running trained model on ROI: {roi_path}")
        for _ in tqdm(range(1)):

            checkpoint = torch.load(args.model_weights)
            if args.arch_name == "V2" or args.arch_name == "V4" or args.arch_name == "IT":
               
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    # Remap keys like '0.0.weight' -> '0.weight'
                    new_k = k[2:]
                    new_state_dict[new_k] = v
                
                checkpoint = new_state_dict

            model.load_state_dict(checkpoint, strict=False)

            fx = FeatureExtractor_1(model=model, device='cuda', pretrained=True)
            features_1 = fx.extract(data_path=stimuli_path, save_path=f'{np.random.randint(0, 100000)}', layers_to_extract=layer_names)

            results_dataframe_cv = RidgeCV_Encoding(
                features=features_1,
                roi_path=roi_path,
                model_name="a",
                n_folds=3,
                trn_tst_split=0.8,
                n_components=100,       
                batch_size=512,
                return_correlations=False,
                save_path="results",
                alpha=1.0
            )

            layerrs.append(results_dataframe_cv[['Layer', 'R']])
            
        meaned_layerrs = [layerrs[i].groupby('Layer').agg(['mean', 'std']).to_numpy() for i in range(len(layerrs))]

        with open(f"fin_results/{args.arch_name}_trained_{roi_path}.npy", 'wb') as f:
            np.save(f, np.array(meaned_layerrs))
            print(f"Saved fin_results/{args.arch_name}_trained_{roi_path}.npy")

    rsa_dataframes = []
    for i in range(args.num_runs):
        folder_name = "temporary/" + str(uuid.uuid4())
        name = args.arch_name + "_" + str(i)

        model.apply(init_weights)

        creator = RDMCreator(verbose=True, device='cuda')
        fx = FeatureExtractor(model=model, device='cuda', pretrained=True)

        save_path_features = f"{folder_name}/{name}_Extract"
        features_1 = fx.extract(data_path=stimuli_path, save_path=save_path_features, layers_to_extract=layer_names, consolidate_per_layer=False)
        save_path_rdms = f"{folder_name}/{name}_RDMs"
        _ = creator.create_rdms(feature_path=save_path_features, save_path=save_path_rdms, save_format='npz')

        rsa_evaluation = RSA(save_path_rdms, "./NSD Dataset/NSD_872_RDMs/prf-visualrois/combined", model_name=name)
        results_df = rsa_evaluation.evaluate()
        rsa_dataframes.append(results_df)

    # Calculate mean across all dataframes
    # Calculate mean across all dataframes, grouping only by Model
    mean_df = pd.concat(rsa_dataframes).groupby(['Layer', 'ROI']).agg({
        'R2': lambda x: x.mean(),
        '%R2': lambda x: x.mean(),
        'Significance': lambda x: x.mean(),
        'SEM': lambda x: x.mean(),
        'LNC': lambda x: x.mean(),
        'UNC': lambda x: x.mean()
    }).reset_index()

    mean_df.to_csv(f"fin_results/{args.arch_name}_RSA.csv")
   

    """
    rsa_dataframes = []
    for i in range(1):
        folder_name = "temporary/" + str(uuid.uuid4())
        name = args.arch_name + "_" + str(i)

        checkpoint = torch.load(args.model_weights)
        if args.arch_name == "V2" or args.arch_name == "V4" or args.arch_name == "IT":
            
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # Remap keys like '0.0.weight' -> '0.weight'
                new_k = k[2:]
                new_state_dict[new_k] = v
            
            checkpoint = new_state_dict

        model.load_state_dict(checkpoint, strict=False)

        creator = RDMCreator(verbose=True, device='cuda')
        fx = FeatureExtractor(model=model, device='cuda', pretrained=True)

        save_path_features = f"{folder_name}/{name}_Extract"
        features_1 = fx.extract(data_path=stimuli_path, save_path=save_path_features, layers_to_extract=layer_names, consolidate_per_layer=False)
        save_path_rdms = f"{folder_name}/{name}_RDMs"
        _ = creator.create_rdms(feature_path=save_path_features, save_path=save_path_rdms, save_format='npz')

        rsa_evaluation = RSA(save_path_rdms, "./NSD Dataset/NSD_872_RDMs/prf-visualrois/combined", model_name=name)
        results_df = rsa_evaluation.evaluate()
        rsa_dataframes.append(results_df)


    # Calculate mean across all dataframes
    # Calculate mean across all dataframes, grouping only by Model
    mean_df = pd.concat(rsa_dataframes).groupby(['Layer', 'ROI']).agg({
        'R2': lambda x: x.mean(),
        '%R2': lambda x: x.mean(),
        'Significance': lambda x: x.mean(),
        'SEM': lambda x: x.mean(),
        'LNC': lambda x: x.mean(),
        'UNC': lambda x: x.mean()
    }).reset_index()

    mean_df.to_csv(f"fin_results/{args.arch_name}_trained_RSA.csv")
    """
if __name__ == "__main__":
  main()