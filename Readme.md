## Evolutionary Neural Architecture Search for Brain-inspired Neural Networks

This code is a work in progress. It is a wrapper around the net2brain library to perform evolutionary neural architecture search.

### Arguments
- `--wandb_key`: Weights & Biases API key
- `--population_size`: Size of the population for evolution (default: 25)
- `--generations`: Number of generations to run (default: 400)
- `--mutation_rate`: Rate of mutation during evolution (default: 0.5)
- `--crossover_rate`: Rate of crossover during evolution (default: 0.5)
- `--tmp_dir`: Directory for temporary evolution files (default: './tmp_evo')
- `--gpu_id`: Specify GPU ID (0-3)
- `--run_name`: Name for this wandb run. If not provided, wandb will auto-generate one
- `--searchable_padding`: Whether to search for padding (default: False)
- `--enforce_increasing_channels`: Whether to enforce increasing channels (default: True)