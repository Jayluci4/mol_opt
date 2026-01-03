"""
Experience collection for ES meta-learning.

Runs graph_ga-style optimization on training oracles to collect
(seed_molecules, final_scores) pairs for meta-training.
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tdc import Oracle
from main.es_meta.run import ES_Meta_Collector
from main.es_meta.task_embeddings import TRAINING_ORACLES


def collect_for_oracle(oracle_name: str, args):
    """
    Run experience collection for a single oracle.

    Args:
        oracle_name: Name of the oracle
        args: Command-line arguments
    """
    print(f"\n{'='*60}")
    print(f"Collecting experience for: {oracle_name}")
    print(f"{'='*60}")

    # Set up oracle
    oracle = Oracle(name=oracle_name)

    # Create collector (uses random selection, collects experience)
    collector = ES_Meta_Collector(args=args)

    # Load default config
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'hparams_default.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run optimization and collect experience
    collector.optimize(oracle=oracle, config=config, seed=args.seed)

    print(f"Done collecting for {oracle_name}")


def main():
    parser = argparse.ArgumentParser(description='Collect experience for ES meta-learning')
    parser.add_argument('--oracles', nargs='+', default=None,
                        help='Oracles to collect experience for (default: all training oracles)')
    parser.add_argument('--experience_dir', type=str, default='main/es_meta/data/experience',
                        help='Directory to save experience')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--max_oracle_calls', type=int, default=10000, help='Max oracle calls per task')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--collect_experience', default=True)

    args = parser.parse_args()
    args.experience_dir = args.experience_dir
    args.method = 'es_meta_collector'

    # Set output_dir if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), 'data', 'results')
    os.makedirs(args.output_dir, exist_ok=True)

    # Use all training oracles if not specified
    oracles = args.oracles if args.oracles else TRAINING_ORACLES

    print(f"Will collect experience for {len(oracles)} oracles:")
    for i, o in enumerate(oracles, 1):
        print(f"  {i}. {o}")

    # Collect experience for each oracle
    for oracle_name in oracles:
        try:
            collect_for_oracle(oracle_name, args)
        except Exception as e:
            print(f"Error collecting for {oracle_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # List collected experience files
    exp_dir = args.experience_dir
    if os.path.exists(exp_dir):
        exp_files = [f for f in os.listdir(exp_dir) if f.endswith('_experience.pt')]
        print(f"\nCollected {len(exp_files)} experience files in {exp_dir}")
        for f in exp_files:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
