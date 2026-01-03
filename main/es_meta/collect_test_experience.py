"""
Collect experience from test oracles for improved meta-training.
"""

import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TEST_ORACLES = [
    'Sitagliptin_MPO',
    'Zaleplon_MPO',
    'Deco_Hop',
    'Fexofenadine_MPO',
    'Osimertinib_MPO',
]


def collect_oracle_experience(oracle_name: str, max_calls: int = 5000):
    """Collect experience by running graph_ga on an oracle."""
    script = f'''
import sys
import os
import numpy as np
import yaml
import torch

sys.path.insert(0, '/home/jayantlohia16/mol_opt')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tdc import Oracle

config_path = 'main/es_meta/hparams_default.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

class Args:
    method = 'graph_ga'
    smi_file = None
    n_jobs = -1
    output_dir = 'main/es_meta/data/experience'
    patience = 5
    max_oracle_calls = {max_calls}
    freq_log = 500
    log_results = False
    pool = None

os.makedirs(Args.output_dir, exist_ok=True)
np.random.seed(42)

oracle = Oracle(name='{oracle_name}')

from main.graph_ga.run import GB_GA_Optimizer
from main.es_meta.mol_embeddings import EmbeddingCache

optimizer = GB_GA_Optimizer(args=Args())

# Get initial seeds before optimization
initial_seeds = list(np.random.choice(optimizer.all_smiles, config["population_size"], replace=False))

# Run optimization
optimizer.optimize(oracle=oracle, config=config, seed=42)
optimizer.sort_buffer()

# Get final scores
top_items = list(optimizer.mol_buffer.items())[:100]
final_scores = [item[1][0] for item in top_items]
final_top10_avg = np.mean(final_scores[:10])

# Get seed embeddings
cache = EmbeddingCache(optimizer.all_smiles, cache_path='main/es_meta/data/zinc_embeddings.pt')
seed_embeddings = cache.get_embeddings(initial_seeds)

# Save experience
exp_path = f'main/es_meta/data/experience/{oracle_name.lower()}_experience.pt'
torch.save({{
    'oracle_name': '{oracle_name}',
    'seed_smiles': initial_seeds,
    'seed_embeddings': seed_embeddings,
    'final_scores': final_scores,
    'final_top10_avg': final_top10_avg,
}}, exp_path)

print(f"Saved experience for {oracle_name}: top10_avg={{final_top10_avg:.4f}}")
'''

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=1200,
        cwd='/home/jayantlohia16/mol_opt'
    )

    if result.returncode == 0:
        print(f"Success: {oracle_name}")
        for line in result.stdout.strip().split('\n')[-5:]:
            print(f"  {line}")
    else:
        print(f"Error: {oracle_name}")
        print(result.stderr[-500:] if result.stderr else "No stderr")

    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracles', nargs='+', default=TEST_ORACLES)
    parser.add_argument('--max_calls', type=int, default=5000)
    args = parser.parse_args()

    print(f"Collecting experience for {len(args.oracles)} oracles...")

    for oracle in args.oracles:
        print(f"\n{'='*60}")
        print(f"Processing: {oracle}")
        print('='*60)
        collect_oracle_experience(oracle, args.max_calls)

    print("\nDone!")


if __name__ == '__main__':
    main()
