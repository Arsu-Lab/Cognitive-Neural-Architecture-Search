import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import numpy as np
import os
import time
import shutil
import gc
import json

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from evolution.architecture import Architecture
import wandb
import argparse

class EvolutionStrategy:
    def __init__(self, population_size=25, generations=100, mutation_rate=0.25, crossover_rate=0.5, 
                 roi_path=None, stimuli_path=None, tmp_dir='./tmp_evo', device=None, gpu_id=None, 
                 wandb_run=None, searchable_padding=False, enforce_increasing_channels=True):
        self.population_size = population_size 
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.run = wandb_run  # Store the wandb run instance
        
        # Create tmp directory with proper permissions
        try:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, mode=0o777, exist_ok=True)
        except Exception as e:
            # If default path fails, try user's home directory
            self.tmp_dir = os.path.join(os.path.expanduser('~'), 'tmp_evo')
            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir, mode=0o777, exist_ok=True)
            print(f"Using alternative tmp directory: {self.tmp_dir}")
        else:
            self.tmp_dir = tmp_dir
        
        # Updated device selection logic
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
        elif device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Using device: {self.device}")
        
        self.population = [Architecture(roi_path=roi_path, tmp_dir=tmp_dir, searchable_padding=searchable_padding, enforce_increasing_channels=enforce_increasing_channels) 
                          for _ in range(population_size)]
        self.roi_path = roi_path
        self.stimuli_path = stimuli_path
        
        # Set up console
        self.console = Console()

        self.architecture_cache = {}

    def evolve(self):
        for generation in range(self.generations):
            gen_text = Text(f"Generation {generation+1}", style="bold cyan")
            self.console.print(gen_text)
            
            # Increment age of all architectures
            for arch in self.population:
                arch.age += 1
            
            # Evaluate fitness
            for i, architecture in enumerate(self.population):
                if architecture.fitness is None:
                    try:
                        architecture.fitness = self.evaluate(architecture)
                    except Exception as e:
                        self.console.print(f"[red]Error evaluating architecture {i}: {e}[/red]")
                        architecture.fitness = float('-inf')
                        continue

            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)
            
            # Keep top 50%
            top_half = self.population[:self.population_size//2]
            
            # Create new children to replace bottom 50%
            new_children = []
            for _ in range(self.population_size//2):
                parent1, parent2 = random.sample(top_half, 2)
                
                if random.random() < self.crossover_rate:
                    child = parent1.crossover(parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                if random.random() < self.mutation_rate:
                    child.mutate()
                
                child.fitness = None
                child.age = 0
                child.roi_path = self.roi_path
                child.tmp_dir = self.tmp_dir
                
                new_children.append(child)
            
            # Combine populations
            self.population = top_half + new_children
            
            # Log stats - only consider valid fitness values
            valid_fitness = [arch.fitness for arch in self.population if arch.fitness is not None]
            if valid_fitness:  # Only log if we have valid fitness values
                best_fitness = max(valid_fitness)
                avg_age = sum(arch.age for arch in self.population) / len(self.population)
                
                self.run.log({
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "average_age": avg_age,
                    "max_age": max(arch.age for arch in self.population)
                })

            self.save_generation(generation)

    def evaluate(self, architecture):
        # Create a hash of the architecture's layers
        arch_key = str(architecture.layers)
        original_num_layers = len(architecture.layers)
        
        # Check cache first
        if arch_key in self.architecture_cache:
            fitness = self.architecture_cache[arch_key]
            self.console.print(Panel(f"[bold yellow]Cached Fitness: {fitness:.4f}[/bold yellow]", expand=False))
        else:
            # If not in cache, evaluate and store
            model_summary = self.get_model_summary(architecture)
            self.console.print(Panel(model_summary, title="[bold green]Evaluating Model[/bold green]", expand=False))
            
            # Build model to get mapping between layer indices
            model = architecture.build_model(self.device)
            
            # Create mapping between feature extraction indices and architecture indices
            layer_mapping = {}
            arch_idx = 0
            for i, layer in enumerate(model):
                if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Linear)):
                    layer_mapping[str(i)] = arch_idx
                    arch_idx += 1
            
            # Get scores for all layers
            layer_scores = architecture.evaluate_model(
                self.stimuli_path, 
                self.device, 
                num_seeds=3,
                evaluate_all_layers=True
            )
            
            if isinstance(layer_scores, dict):
                # Create performance summary
                performance_lines = ["[bold blue]Layer Performance:[/bold blue]"]
                
                for layer_idx in sorted(layer_scores.keys(), key=int):
                    score = layer_scores[layer_idx]
                    layer_type = type(model[int(layer_idx)]).__name__
                    arch_idx = layer_mapping[layer_idx]
                    performance_lines.append(f"Architecture Layer {arch_idx + 1} ({layer_type}): {score:.4f}")
                
                # Find best performing layer
                best_layer = max(layer_scores.items(), key=lambda x: x[1])
                best_layer_idx = int(best_layer[0])
                
                # Map the feature extraction index to architecture index
                best_arch_idx = layer_mapping[str(best_layer_idx)]
                
                # Add truncation information if applicable
                if best_arch_idx < len(architecture.layers) - 1:
                    architecture.layers = architecture.layers[:best_arch_idx + 1]
                    performance_lines.append("")
                    performance_lines.append("[bold red]Network Truncated![/bold red]")
                    performance_lines.append(f"Original layers: {original_num_layers}")
                    performance_lines.append(f"Truncated to architecture layer: {best_arch_idx + 1}")
                    performance_lines.append(f"Layer type: {type(model[best_layer_idx]).__name__}")
                
                del model
                
                fitness = best_layer[1]
                performance_lines.append("")
                performance_lines.append(f"[bold green]Best Fitness: {fitness:.4f}[/bold green]")
                
                # Print performance summary
                self.console.print(Panel("\n".join(performance_lines), 
                                       title="[bold yellow]Layer-wise Performance[/bold yellow]",
                                       expand=False))
            else:
                fitness = layer_scores
                self.console.print(Panel(f"[bold yellow]Mean Fitness: {fitness:.4f}[/bold yellow]", expand=False))

            self.architecture_cache[arch_key] = fitness
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log individual reward to wandb
        if fitness > 0:
            self.run.log({
                "rewards": fitness
            })
        
        return fitness

    def get_model_summary(self, architecture):
        summary = []
        for i, layer in enumerate(architecture.layers):
            layer_str = f"Layer {i+1}: {layer['type']}"
            if layer['type'] == 'conv':
                layer_str += f" (out_channels={layer['out_channels']}, kernel_size={layer['kernel_size']}, stride={layer['stride']}, padding={layer['padding']})"
            elif layer['type'] == 'pool':
                layer_str += f" (kernel_size={layer['kernel_size']}, stride={layer['stride']})"
            elif layer['type'] == 'fc':
                layer_str += f" (out_features={layer['out_features']})"
            summary.append(layer_str)
        return "\n".join(summary)

    def save_generation(self, generation_number):
        """Save current population to JSON"""
        # Convert population to list of dictionaries
        population_data = {
            'generation': generation_number,
            'architectures': [
                {
                    'individual': i,
                    'architecture': arch.to_dict()
                }
                for i, arch in enumerate(self.population)
            ]
        }
        
        # Create directory if it doesn't exist
        save_dir = os.path.join(self.tmp_dir, 'architecture_history')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save to JSON
        json_path = os.path.join(save_dir, f'generation_{generation_number}.json')
        with open(json_path, 'w') as f:
            json.dump(population_data, f, indent=2)
        
        # If using wandb, log the JSON file
        if self.run:
            self.run.save(json_path)

    @classmethod
    def load_generation(cls, json_path, roi_path=None, tmp_dir=None, searchable_padding=False, 
                       enforce_increasing_channels=True):
        """Load population from JSON file"""
        # Read JSON file
        with open(json_path, 'r') as f:
            population_data = json.load(f)
        
        # Create architectures from data
        architectures = []
        for arch_data in population_data['architectures']:
            arch = Architecture.from_dict(
                arch_data['architecture'],
                roi_path=roi_path,
                tmp_dir=tmp_dir,
                searchable_padding=searchable_padding,
                enforce_increasing_channels=enforce_increasing_channels
            )
            architectures.append(arch)
        
        return architectures