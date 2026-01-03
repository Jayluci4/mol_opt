"""
Meta-training script for ES seed selection.

Loads collected experience and trains the seed scorer using ES updates.
"""

import os
import sys
import argparse
import glob
import torch
import numpy as np
from typing import List, Dict

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main.es_meta.meta_learner import ESMetaSeedLearner, Experience
from main.es_meta.seed_scorer import SeedScorer
from main.es_meta.task_embeddings import get_task_embedding


def load_experience(experience_dir: str) -> List[Experience]:
    """
    Load all experience files from a directory.

    Args:
        experience_dir: Directory containing experience .pt files

    Returns:
        List of Experience objects
    """
    experiences = []

    pattern = os.path.join(experience_dir, '*_experience.pt')
    exp_files = glob.glob(pattern)

    print(f"Found {len(exp_files)} experience files in {experience_dir}")

    for exp_file in sorted(exp_files):
        try:
            data = torch.load(exp_file)
            exp = Experience(
                oracle_name=data['oracle_name'],
                seed_smiles=data['seed_smiles'],
                final_scores=data['final_scores'],
                seed_embeddings=data['seed_embeddings']
            )
            experiences.append(exp)
            print(f"  Loaded: {os.path.basename(exp_file)} "
                  f"(oracle={exp.oracle_name}, n_seeds={len(exp.seed_smiles)})")
        except Exception as e:
            print(f"  Error loading {exp_file}: {e}")
            continue

    return experiences


def meta_train(experiences: List[Experience],
               n_epochs: int = 100,
               population_size: int = 20,
               noise_std: float = 0.02,
               elite_fraction: float = 0.2,
               lr: float = 0.1,
               output_path: str = 'meta_scorer.pt') -> ESMetaSeedLearner:
    """
    Train the meta-learner on collected experience.

    Args:
        experiences: List of Experience objects
        n_epochs: Number of meta-update epochs
        population_size: ES population size
        noise_std: Noise standard deviation for perturbation
        elite_fraction: Fraction of population to select as elites
        lr: Learning rate for meta-update
        output_path: Path to save trained meta-learner

    Returns:
        Trained ESMetaSeedLearner
    """
    print(f"\n{'='*60}")
    print("Starting meta-training")
    print(f"{'='*60}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Population size: {population_size}")
    print(f"  Noise std: {noise_std}")
    print(f"  Elite fraction: {elite_fraction}")
    print(f"  Learning rate: {lr}")
    print(f"  Total experience samples: {len(experiences)}")

    # Initialize meta-learner
    learner = ESMetaSeedLearner(
        population_size=population_size,
        noise_std=noise_std,
        elite_fraction=elite_fraction,
        lr=lr
    )

    # Load experiences into buffer
    for exp in experiences:
        learner.experience_buffer.append(exp)

    # Run meta-training
    print(f"\nRunning {n_epochs} meta-update epochs...")

    fitness_history = learner.meta_train(
        n_epochs=n_epochs,
        task_embedding_fn=get_task_embedding,
        verbose=True
    )

    # Save trained meta-learner
    learner.save(output_path)

    # Print summary statistics
    print(f"\n{'='*60}")
    print("Meta-training complete")
    print(f"{'='*60}")
    print(f"  Final best fitness: {fitness_history[-1]:.4f}")
    print(f"  Improvement: {fitness_history[-1] - fitness_history[0]:.4f}")
    print(f"  Saved to: {output_path}")

    return learner


def evaluate_on_held_out(learner: ESMetaSeedLearner,
                         held_out_dir: str = None) -> Dict:
    """
    Evaluate meta-learner on held-out experience (if available).

    Args:
        learner: Trained meta-learner
        held_out_dir: Directory with held-out experience

    Returns:
        Evaluation metrics
    """
    if held_out_dir is None or not os.path.exists(held_out_dir):
        return {}

    held_out = load_experience(held_out_dir)
    if not held_out:
        return {}

    print(f"\nEvaluating on {len(held_out)} held-out experiences...")

    # Evaluate prediction error
    total_error = 0.0
    learner.scorer.eval()

    with torch.no_grad():
        for exp in held_out:
            task_emb = get_task_embedding(exp.oracle_name)
            predicted = learner.score_molecules(exp.seed_embeddings, task_emb)

            actual = torch.tensor(exp.final_scores, dtype=torch.float32)
            if actual.max() > 0:
                actual = actual / actual.max()

            error = ((predicted - actual) ** 2).mean().item()
            total_error += error

            print(f"  {exp.oracle_name}: MSE = {error:.4f}")

    avg_error = total_error / len(held_out)
    print(f"  Average MSE: {avg_error:.4f}")

    return {'held_out_mse': avg_error}


def main():
    parser = argparse.ArgumentParser(description='Meta-train ES seed selector')
    parser.add_argument('--experience_dir', type=str,
                        default='main/es_meta/data/experience',
                        help='Directory containing experience files')
    parser.add_argument('--output', type=str, default='main/es_meta/data/meta_scorer.pt',
                        help='Path to save trained meta-learner')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of meta-update epochs')
    parser.add_argument('--population_size', type=int, default=20,
                        help='ES population size')
    parser.add_argument('--noise_std', type=float, default=0.02,
                        help='Noise standard deviation')
    parser.add_argument('--elite_fraction', type=float, default=0.2,
                        help='Fraction of population to select as elites')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for meta-update')
    parser.add_argument('--held_out_dir', type=str, default=None,
                        help='Directory with held-out experience for evaluation')

    args = parser.parse_args()

    # Load experience
    experiences = load_experience(args.experience_dir)

    if not experiences:
        print("No experience found! Run collect_experience.py first.")
        sys.exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Train
    learner = meta_train(
        experiences=experiences,
        n_epochs=args.epochs,
        population_size=args.population_size,
        noise_std=args.noise_std,
        elite_fraction=args.elite_fraction,
        lr=args.lr,
        output_path=args.output
    )

    # Evaluate on held-out
    if args.held_out_dir:
        evaluate_on_held_out(learner, args.held_out_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
