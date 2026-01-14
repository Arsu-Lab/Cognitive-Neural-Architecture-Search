import argparse
import os
import shutil
import wandb
from rich.panel import Panel
from evolution.evolution import EvolutionStrategy
from net2brain.utils.download_datasets import DatasetNSD_872

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_key', type=str, help='Weights & Biases API key')
    parser.add_argument('--population_size', type=int, default=25)
    parser.add_argument('--generations', type=int, default=400)
    parser.add_argument('--mutation_rate', type=float, default=0.5)
    parser.add_argument('--crossover_rate', type=float, default=0.5)
    parser.add_argument('--tmp_dir', type=str, default='./tmp_evo',
                       help='Directory for temporary evolution files')
    parser.add_argument('--gpu_id', type=int, default=None, help="Specify GPU ID (0-3)")
    parser.add_argument('--run_name', type=str, default=None,
                       help='Name for this wandb run. If not provided, wandb will auto-generate one.')
    parser.add_argument('--searchable_padding', type=bool, default=False, help="Whether to search for padding")
    parser.add_argument('--enforce_increasing_channels', type=bool, default=True, help="Whether to enforce increasing channels")
    args = parser.parse_args()

    # Set GPU if specified
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")

    # Set up wandb if key provided
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key

    # Initialize wandb run
    wandb_run = wandb.init(
        project="shallow-brain-evo-nas-test",
        name=args.run_name,
        config={
            "population_size": args.population_size,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "crossover_rate": args.crossover_rate,
            "searchable_padding": args.searchable_padding,
            "enforce_increasing_channels": args.enforce_increasing_channels,
        }
    )

    # Create tmp directory before initializing EvolutionStrategy
    try:
        if os.path.exists(args.tmp_dir):
            shutil.rmtree(args.tmp_dir)
        os.makedirs(args.tmp_dir, mode=0o777, exist_ok=True)
    except Exception:
        args.tmp_dir = os.path.join(os.path.expanduser('~'), 'tmp_evo')
        if not os.path.exists(args.tmp_dir):
            os.makedirs(args.tmp_dir, mode=0o777, exist_ok=True)
        print(f"Using alternative tmp directory: {args.tmp_dir}")

    paths_NSD_872 = DatasetNSD_872().load_dataset()
    stimuli_path = paths_NSD_872["NSD_872_images"]
    roi_path = "./V2Combined"

    evo = EvolutionStrategy(
        roi_path=roi_path,
        stimuli_path=stimuli_path,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        tmp_dir=args.tmp_dir,
        gpu_id=args.gpu_id,
        wandb_run=wandb_run,
        searchable_padding=args.searchable_padding,
        enforce_increasing_channels=args.enforce_increasing_channels
    )

    evo.evolve()

    # After evolution, print the best architecture
    best_arch = max(evo.population, key=lambda x: x.fitness)
    evo.console.print(Panel.fit(
        evo.get_model_summary(best_arch),
        title="[bold magenta]Best Architecture[/bold magenta]",
        subtitle=f"[bold cyan]Fitness: {best_arch.fitness:.4f}[/bold cyan]"
    ))

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
