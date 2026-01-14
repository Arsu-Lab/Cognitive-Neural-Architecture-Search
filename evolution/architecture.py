from config import LAYER_TYPES, NON_LINEARITIES, PARAM_RANGES
import random
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import gc
from utils.feature_extraction import FeatureExtractor
from utils.ridge_regression import RidgeCV_Encoding
from multiprocessing import Pool
import functools


class Architecture:
    def __init__(
        self,
        layers=None,
        roi_path=None,
        tmp_dir=None,
        enforce_increasing_channels=True,
        searchable_padding=False,
    ):
        self.enforce_increasing_channels = enforce_increasing_channels
        self.searchable_padding = searchable_padding
        if layers is None:
            self.layers = []
            # Force first layer to be conv with large kernel size
            first_layer = {
                "type": "conv",
                "out_channels": random.choice(PARAM_RANGES["conv"]["out_channels"]),
                "kernel_size": random.choice([5, 7, 9, 11]),
                "stride": random.choice([2, 3, 4]),
                "padding": 0,
            }
            self.layers.append(first_layer)

            # Add up to 9 more random layers
            num_additional_layers = random.randint(0, 9)
            fc_count = 0
            prev_channels = first_layer["out_channels"]  # Track previous channels

            for _ in range(num_additional_layers):
                prev_layer_type = self.layers[-1]["type"]
                valid_types = [
                    lt
                    for lt in LAYER_TYPES
                    if self.is_valid_action(
                        lt, len(self.layers), prev_layer_type, fc_count
                    )
                ]

                if not valid_types:
                    break

                new_layer_type = random.choice(valid_types)
                new_layer = self.random_layer_of_type(new_layer_type, prev_channels)
                self.layers.append(new_layer)

                if new_layer_type == "fc":
                    fc_count += 1
                elif new_layer_type == "conv":
                    prev_channels = new_layer["out_channels"]  # Update prev_channels
        else:
            self.layers = layers

        self.fitness = None
        self.roi_path = roi_path
        self.tmp_dir = tmp_dir
        self.age = 0  # Track age of architecture
        self.ensure_constraints()

    def is_valid_action(self, layer_type, current_depth, prev_layer_type, fc_count):
        if current_depth >= 20:  # Max depth constraint
            return False
        if prev_layer_type == "pool" and layer_type == "pool":
            return False
        if prev_layer_type == "fc" and layer_type in ["conv", "pool"]:
            return False
        if layer_type == "fc":
            if fc_count >= 3:  # Max 3 FC layers
                return False
            if current_depth < 2:  # Need at least 2 conv/pool layers before FC
                return False
        return True

    def mutate(self):
        mutation_type = random.choice(["add", "modify", "remove"])

        if mutation_type == "add" and len(self.layers) < 20:
            fc_count = sum(1 for layer in self.layers if layer["type"] == "fc")
            prev_layer_type = self.layers[-1]["type"]
            valid_types = [
                lt
                for lt in LAYER_TYPES
                if self.is_valid_action(lt, len(self.layers), prev_layer_type, fc_count)
            ]

            if valid_types:
                new_layer_type = random.choice(valid_types)
                # Get previous channels if adding a conv layer
                prev_channels = None
                if new_layer_type == "conv":
                    prev_channels = next(
                        (
                            layer["out_channels"]
                            for layer in reversed(self.layers)
                            if layer["type"] == "conv"
                        ),
                        None,
                    )
                new_layer = self.random_layer_of_type(new_layer_type, prev_channels)
                self.layers.append(new_layer)

        elif mutation_type == "modify" and len(self.layers) > 1:
            idx = random.randint(1, len(self.layers) - 1)
            layer = self.layers[idx]

            if random.random() < 0.3:  # 30% chance to change layer type
                current_type = layer["type"]
                # Get valid layer types for this position
                fc_count = sum(1 for l in self.layers if l["type"] == "fc")
                prev_layer_type = self.layers[idx - 1]["type"] if idx > 0 else None
                valid_types = [
                    lt
                    for lt in LAYER_TYPES
                    if self.is_valid_action(lt, idx, prev_layer_type, fc_count)
                ]

                # Remove current type from options
                if current_type in valid_types:
                    valid_types.remove(current_type)

                if valid_types:  # If we have valid alternatives
                    new_type = random.choice(valid_types)
                    # Replace layer with new random layer of chosen type
                    self.layers[idx] = self.random_layer_of_type(new_type)
                    return

            # If modifying a conv layer's channels, respect the constraints
            if layer["type"] == "conv" and self.enforce_increasing_channels:
                prev_channels = next(
                    (
                        l["out_channels"]
                        for l in reversed(self.layers[:idx])
                        if l["type"] == "conv"
                    ),
                    0,
                )
                next_channels = next(
                    (
                        l["out_channels"]
                        for l in self.layers[idx + 1 :]
                        if l["type"] == "conv"
                    ),
                    float("inf"),
                )
                valid_channels = [
                    c
                    for c in PARAM_RANGES["conv"]["out_channels"]
                    if c >= prev_channels and c <= next_channels
                ]
                if valid_channels:
                    layer["out_channels"] = random.choice(valid_channels)
            else:
                param = random.choice(list(PARAM_RANGES[layer["type"]].keys()))
                layer[param] = random.choice(PARAM_RANGES[layer["type"]][param])

        elif (
            mutation_type == "remove" and len(self.layers) > 1
        ):  # Don't remove first layer
            removable = [
                i
                for i in range(1, len(self.layers))
                if self.layers[i]["type"] != "fc"
                or sum(1 for l in self.layers if l["type"] == "fc") > 1
            ]
            if removable:
                idx = random.choice(removable)
                del self.layers[idx]

        self.ensure_constraints()

    def crossover(self, other):
        # Implement crossover by swapping a subset of layers
        child_layers = copy.deepcopy(self.layers)
        if other.layers:
            crossover_point = random.randint(0, len(child_layers))
            child_layers = child_layers[:crossover_point] + copy.deepcopy(
                other.layers[crossover_point:]
            )
        return Architecture(child_layers)

    def random_layer(self):
        layer_type = random.choice(LAYER_TYPES)
        return self.random_layer_of_type(layer_type)

    def ensure_constraints(self):
        if len(self.layers) == 0:
            return

        new_layers = [self.layers[0]]  # Keep first conv layer
        current_size = 224  # Input size
        prev_channels = self.layers[0][
            "out_channels"
        ]  # Track previous conv layer's channels

        # Calculate size after first conv layer
        first_layer = self.layers[0]
        current_size = (
            (current_size - first_layer["kernel_size"] + 2 * first_layer["padding"])
            // first_layer["stride"]
        ) + 1

        for layer in self.layers[1:]:
            if current_size < 4:  # Minimum size threshold
                break

            # Skip if this would create consecutive pool layers
            if layer["type"] == "pool" and new_layers[-1]["type"] == "pool":
                continue

            if layer["type"] == "conv":
                # Enforce monotonically increasing channels if flag is set
                if (
                    self.enforce_increasing_channels
                    and layer["out_channels"] < prev_channels
                ):
                    layer["out_channels"] = prev_channels

                layer["stride"] = 1  # All subsequent conv layers have stride 1
                if not self.searchable_padding:
                    layer["padding"] = (
                        layer["kernel_size"] // 2
                    )  # Use 'same' padding only if not searchable
                current_size = (
                    (current_size - layer["kernel_size"] + 2 * layer["padding"])
                    // layer["stride"]
                ) + 1
                new_layers.append(layer)
                prev_channels = layer["out_channels"]  # Update previous channels
            elif layer["type"] == "pool":
                current_size = current_size // 2
                new_layers.append(layer)
            elif layer["type"] == "fc":
                new_layers.append(layer)

        self.layers = new_layers

    def random_layer_of_type(self, layer_type, prev_channels=None):
        layer = {"type": layer_type}
        if layer_type == "conv":
            if self.enforce_increasing_channels and prev_channels is not None:
                # Filter channel choices to only those >= prev_channels
                valid_channels = [
                    c
                    for c in PARAM_RANGES["conv"]["out_channels"]
                    if c >= prev_channels
                ]
                layer["out_channels"] = random.choice(valid_channels)
            else:
                layer["out_channels"] = random.choice(
                    PARAM_RANGES["conv"]["out_channels"]
                )
            layer["kernel_size"] = random.choice(PARAM_RANGES["conv"]["kernel_size"])
            layer["stride"] = random.choice(PARAM_RANGES["conv"]["stride"])
            layer["padding"] = random.choice(PARAM_RANGES["conv"]["padding"])
        elif layer_type == "pool":
            layer["kernel_size"] = random.choice(PARAM_RANGES["pool"]["kernel_size"])
            layer["stride"] = random.choice(PARAM_RANGES["pool"]["stride"])
        elif layer_type == "fc":
            layer["out_features"] = random.choice(PARAM_RANGES["fc"]["out_features"])
        return layer

    def build_model(self, device):
        layers = []
        in_channels = 3  # RGB input
        current_size = 224  # Input size
        first_fc = True  # Flag to track first FC layer

        for i, layer in enumerate(self.layers):
            layer_type = layer["type"]
            if layer_type == "conv":
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                )
                layers.append(conv)
                layers.append(nn.BatchNorm2d(layer["out_channels"]))
                layers.append(NON_LINEARITIES["Conv2D"]())
                in_channels = layer["out_channels"]
                # Update spatial size
                current_size = self.compute_conv_output_size(
                    current_size,
                    layer["kernel_size"],
                    layer["stride"],
                    layer["padding"],
                )
            elif layer_type == "pool":
                pool = nn.MaxPool2d(
                    kernel_size=layer["kernel_size"], stride=layer["stride"]
                )
                layers.append(pool)
                # Update spatial size
                current_size = self.compute_conv_output_size(
                    current_size, layer["kernel_size"], layer["stride"], 0
                )
            elif layer_type == "fc":
                if first_fc:
                    # Only flatten before first FC layer
                    layers.append(nn.Flatten())
                    in_features = in_channels * current_size * current_size
                    first_fc = False
                else:
                    # For subsequent FC layers, use previous layer's out_features
                    in_features = prev_out_features

                linear = nn.Linear(
                    in_features=in_features, out_features=layer["out_features"]
                )
                layers.append(linear)
                layers.append(NON_LINEARITIES["Linear"]())
                prev_out_features = layer["out_features"]  # Store for next FC layer
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        model = nn.Sequential(*layers)
        model = model.to(device)
        return model

    @staticmethod
    def compute_conv_output_size(size, kernel_size, stride, padding):
        return int((size - kernel_size + 2 * padding) / stride) + 1

    def evaluate_single_seed(
        self, seed, stimuli_path, device, evaluate_all_layers=False
    ):
        """Helper method to evaluate a single seed"""
        try:
            model = self.build_model(device)
            model.apply(self.init_weights)

            # Get layers to extract, filtering out non-linearity layers
            computational_layers = []
            for i, layer in enumerate(model):
                # Check if layer is a computational layer (Conv2d, MaxPool2d, Linear)
                if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Linear)):
                    computational_layers.append(str(i))

            if evaluate_all_layers:
                layers_to_extract = computational_layers
            else:
                # Get the last computational layer
                layers_to_extract = [computational_layers[-1]]

            # Extract features
            fx = FeatureExtractor(model=model, device=device)
            features = fx.extract(
                data_path=stimuli_path, layers_to_extract=layers_to_extract
            )

            torch.cuda.empty_cache()
            del fx
            gc.collect()

            results_dataframe = RidgeCV_Encoding(
                features=features,
                roi_path=self.roi_path,
                model_name=f"temp_model_{int(time.time() * 1000)}",
                n_folds=3,
                trn_tst_split=0.8,
                n_components=100,
                batch_size=64,
                return_correlations=False,
                save_path=os.path.join(
                    self.tmp_dir, f"results_{int(time.time() * 1000)}"
                ),
                alpha=1.0,
            )

            if evaluate_all_layers:
                # Get mean R score for each layer
                layer_scores = {}
                for layer_name in layers_to_extract:
                    layer_df = results_dataframe[results_dataframe.Layer == layer_name]
                    layer_scores[layer_name] = layer_df.R.mean()
                return layer_scores
            else:
                # Return just the last layer score
                last_layer_df = results_dataframe[
                    results_dataframe.Layer == layers_to_extract[0]
                ]
                return last_layer_df.R.mean()

        except Exception as e:
            print(f"Error during evaluation (seed {seed}): {e}")
            return None
        finally:
            if "results_dataframe" in locals():
                del results_dataframe
            gc.collect()
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evaluate_model(
        self,
        stimuli_path,
        device,
        num_seeds=1,
        evaluate_all_layers=False,
        num_workers=1,
    ):
        try:
            # Test if model can be built
            model = self.build_model(device)
            del model
        except Exception as e:
            print(f"Error building model: {e}")
            return 0

        # Create partial function with fixed arguments
        evaluate_fn = functools.partial(
            self.evaluate_single_seed,
            stimuli_path=stimuli_path,
            device=device,
            evaluate_all_layers=evaluate_all_layers,
        )

        # Use multiprocessing if num_workers > 1, otherwise use sequential processing
        if num_workers > 1:
            with Pool(num_workers) as pool:
                results = pool.map(evaluate_fn, range(num_seeds))
        else:
            results = [evaluate_fn(seed) for seed in range(num_seeds)]

        if not evaluate_all_layers:
            valid_rewards = [r for r in results if r is not None]
            return np.mean(valid_rewards) if valid_rewards else float("-inf")

        # Process layer-wise results
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return float("-inf")

        # Calculate average performance for each layer across seeds
        layer_averages = {}
        for layer in valid_results[0].keys():
            scores = [result[layer] for result in valid_results]
            layer_averages[layer] = np.mean(scores)

        return layer_averages

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def to_dict(self):
        """Convert architecture to dictionary format for storage"""
        return {
            "layers": self.layers,  # No need to convert to string, JSON handles lists/dicts
            "fitness": self.fitness if self.fitness is not None else float("-inf"),
            "age": self.age,
        }

    @classmethod
    def from_dict(
        cls,
        data,
        roi_path=None,
        tmp_dir=None,
        searchable_padding=False,
        enforce_increasing_channels=True,
    ):
        """Reconstruct architecture from dictionary format"""
        # Create new architecture instance
        arch = cls(
            roi_path=roi_path,
            tmp_dir=tmp_dir,
            searchable_padding=searchable_padding,
            enforce_increasing_channels=enforce_increasing_channels,
        )

        # Set attributes (no need for eval since JSON preserves data structures)
        arch.layers = data["layers"]
        arch.fitness = float(data["fitness"]) if data["fitness"] != "None" else None
        arch.age = int(data["age"])

        return arch
