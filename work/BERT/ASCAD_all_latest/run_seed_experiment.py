import argparse
import subprocess
import yaml
from pathlib import Path
import os
import wandb
from datetime import datetime
import pandas as pd

def run_experiment(args):
    """Run the complete experiment: training and evaluation"""
    
    # Create experiment directory
    experiment_dir = Path(args.output_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb for the overall experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="bert-sca-seed-experiment",
        name=f"seed_experiment_{timestamp}",
        config={
            "dataset_path": args.dataset_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "downsample": args.downsample,
            "num_traces": args.num_traces,
            "seeds": args.seeds,
            "model_type": "bert-base-uncased",
            "embedding": "hex",
            "trace_net": "fc",
            "fusion": "concat"
        }
    )
    
    # Run training
    print("\n=== Starting Training Phase ===")
    train_cmd = [
        "python", "train_ascad_bert_seed_search.py",
        "--dataset_path", args.dataset_path,
        "--output_dir", str(experiment_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--downsample", str(args.downsample),
        "--num_traces", str(args.num_traces),
        "--seeds"
    ] + [str(seed) for seed in args.seeds]
    
    subprocess.run(train_cmd, check=True)
    
    # Find the latest experiment directory
    experiment_dirs = sorted(experiment_dir.glob("RandomSeedExperiment_*"))
    if not experiment_dirs:
        raise RuntimeError("No experiment directory found!")
    latest_experiment = experiment_dirs[-1]
    
    # Load experiment summary
    with open(latest_experiment / "summary.yaml", 'r') as f:
        summary = yaml.safe_load(f)
    
    # Run evaluation for each seed
    print("\n=== Starting Evaluation Phase ===")
    results_table = []
    
    for result in summary['results']:
        seed = result['seed']
        model_path = result['model_paths']['best']
        
        print(f"\nEvaluating model for seed {seed}")
        eval_cmd = [
            "python", "test_ascad_bert_improved.py",
            "--dataset_path", args.dataset_path,
            "--model_path", model_path,
            "--num_traces", str(args.num_traces),
            "--batch_size", str(args.batch_size),
            "--key_byte", "0"  # Assuming we're evaluating key byte 0
        ]
        
        subprocess.run(eval_cmd, check=True)
        
        # Add results to wandb table
        results_table.append({
            "seed": seed,
            "best_val_acc": result['best_val_acc'],
            "final_rank": result['final_rank'],
            "key_found": result['key_found'],
            "model_path": model_path
        })
    
    # Log results to wandb
    wandb.log({
        "results_table": wandb.Table(dataframe=pd.DataFrame(results_table)),
        "best_seed": min(results_table, key=lambda x: x['final_rank'])['seed'],
        "best_final_rank": min(results_table, key=lambda x: x['final_rank'])['final_rank']
    })
    
    print("\n=== Experiment Complete ===")
    print(f"Results saved in: {latest_experiment}")
    
    # Print summary of results
    print("\nSummary of Results:")
    print("=" * 50)
    for result in results_table:
        print(f"\nSeed {result['seed']}:")
        print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
        print(f"Final key rank: {result['final_rank']}")
        print(f"Key found: {result['key_found']}")
    
    # Find best seed
    best_result = min(results_table, key=lambda x: x['final_rank'])
    print("\nBest performing seed:", best_result['seed'])
    print(f"Final key rank: {best_result['final_rank']}")
    print(f"Key found: {best_result['key_found']}")
    
    # Save final summary to wandb
    wandb.log({
        "experiment_summary": wandb.Table(dataframe=pd.DataFrame([{
            "best_seed": best_result['seed'],
            "best_final_rank": best_result['final_rank'],
            "best_val_acc": best_result['best_val_acc'],
            "key_found": best_result['key_found']
        }]))
    })
    
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Run complete random seed experiment")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the ASCAD dataset")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--downsample", type=int, default=1000, help="Number of training samples to use")
    parser.add_argument("--num_traces", type=int, default=1000, help="Number of traces to use for evaluation")
    parser.add_argument("--seeds", type=int, nargs='+', default=[423, 222, 399], help="Random seeds to try")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main() 