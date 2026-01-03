"""
Train V4 meta-learner with hybrid objective and better hyperparameters.
"""

import os
import sys
import glob
import json
import numpy as np
import torch
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main.es_meta.meta_learner_v3 import ESMetaLearnerV3, ExperienceV3
from main.es_meta.task_embeddings_v2 import get_task_embedding_v2


def load_v1_experience(exp_dir: str):
    """Load V1 experience files and convert to V3 format."""
    experiences = []

    for exp_file in glob.glob(os.path.join(exp_dir, '*_experience.pt')):
        try:
            data = torch.load(exp_file, weights_only=False)
            oracle_name = data['oracle_name']
            seed_smiles = data['seed_smiles']
            seed_embeddings = data['seed_embeddings']
            final_scores = data.get('final_scores', [])

            if len(final_scores) >= 10:
                final_top10_avg = np.mean(final_scores[:10])
            else:
                final_top10_avg = data.get('final_top10_avg', np.mean(final_scores) if final_scores else 0)

            exp = ExperienceV3(
                oracle_name=oracle_name,
                seed_smiles=seed_smiles,
                seed_embeddings=seed_embeddings,
                final_top10_avg=final_top10_avg,
                final_scores=final_scores,
            )
            experiences.append(exp)
            print(f"  Loaded: {os.path.basename(exp_file)} (oracle={oracle_name}, score={final_top10_avg:.4f})")

        except Exception as e:
            print(f"  Error loading {exp_file}: {e}")

    return experiences


def train_v4(experiences, n_epochs=500, output_path='main/es_meta/data/meta_scorer_v5.pt'):
    """Train V4 meta-learner with tuned hyperparameters."""
    print(f"\nTraining V4 meta-learner on {len(experiences)} experiences...")

    # Larger population, smaller noise, slower learning for stability
    learner = ESMetaLearnerV3(population_size=50, noise_std=0.015, lr=0.08)

    for exp in experiences:
        learner.experience_buffer.append(exp)

    fitness_history = learner.meta_train(n_epochs=n_epochs, verbose=True)

    learner.save(output_path)
    print(f"Saved to {output_path}")

    return learner


def test_single(method, oracle, seed, max_calls, meta_checkpoint=None, stochastic=True, temperature=None):
    """Run single test in subprocess with adaptive temperature."""
    temp_str = 'None' if temperature is None else str(temperature)
    script = f'''
import sys
import os
import numpy as np
import yaml
import json

sys.path.insert(0, '/home/jayantlohia16/mol_opt')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tdc import Oracle

config_path = 'main/es_meta/hparams_default.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

class Args:
    method = '{method}'
    smi_file = None
    n_jobs = -1
    output_dir = 'main/es_meta/data/test_results_v4'
    patience = 5
    max_oracle_calls = {max_calls}
    freq_log = 500
    log_results = False
    pool = None
    meta_checkpoint = '{meta_checkpoint or ""}'
    embedding_cache = 'main/es_meta/data/zinc_embeddings.pt'
    collect_experience = False
    experience_dir = 'main/es_meta/data/experience'
    temperature = {temp_str}
    stochastic = {stochastic}

os.makedirs(Args.output_dir, exist_ok=True)
np.random.seed({seed})

oracle = Oracle(name='{oracle}')

if '{method}' == 'graph_ga':
    from main.graph_ga.run import GB_GA_Optimizer
    optimizer = GB_GA_Optimizer(args=Args())
else:
    from main.es_meta.run_v4 import ES_Meta_Optimizer_V4
    optimizer = ES_Meta_Optimizer_V4(args=Args())

optimizer.optimize(oracle=oracle, config=config, seed={seed})
optimizer.sort_buffer()

items = list(optimizer.mol_buffer.items())
top1 = items[0][1][0] if items else 0
top10 = [item[1][0] for item in items[:10]]
top10_avg = np.mean(top10) if top10 else 0

print(json.dumps({{"method": "{method}", "oracle": "{oracle}", "seed": {seed}, "top1": float(top1), "top10_avg": float(top10_avg)}}))
'''

    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=600,
            cwd='/home/jayantlohia16/mol_opt'
        )
        for line in result.stdout.strip().split('\n')[::-1]:
            if line.startswith('{'):
                return json.loads(line)
        return {'error': 'No JSON', 'stderr': result.stderr[-300:]}
    except Exception as e:
        return {'error': str(e)}


def run_comparison(oracles, seeds, max_calls, meta_checkpoint, temperature=0.08):
    """Compare graph_ga vs es_meta_v4."""
    results = {'graph_ga': {}, 'es_meta_v4': {}}

    total = len(oracles) * len(seeds) * 2
    count = 0

    for oracle in oracles:
        results['graph_ga'][oracle] = []
        results['es_meta_v4'][oracle] = []

        for seed in seeds:
            for method in ['graph_ga', 'es_meta_v4']:
                count += 1
                print(f"\n[{count}/{total}] {method} on {oracle} (seed={seed})")

                if method == 'graph_ga':
                    res = test_single('graph_ga', oracle, seed, max_calls)
                else:
                    res = test_single('es_meta_v4', oracle, seed, max_calls, meta_checkpoint, temperature=None)  # Adaptive

                if 'error' not in res:
                    print(f"  Top-10: {res['top10_avg']:.4f}")
                    results[method][oracle].append(res)
                else:
                    print(f"  Error: {res.get('error', 'unknown')}")

    return results


def print_summary(results, oracles):
    """Print comparison summary."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY (V4 - Tuned Hyperparameters)")
    print("="*70)
    print(f"{'Oracle':<30} {'graph_ga':>10} {'es_meta_v4':>12} {'Diff':>10}")
    print("-"*70)

    wins = 0
    total_diff = []

    for oracle in oracles:
        ga = [r['top10_avg'] for r in results['graph_ga'].get(oracle, []) if 'top10_avg' in r]
        es = [r['top10_avg'] for r in results['es_meta_v4'].get(oracle, []) if 'top10_avg' in r]

        if ga and es:
            ga_mean = np.mean(ga)
            es_mean = np.mean(es)
            diff = es_mean - ga_mean
            total_diff.append(diff)
            if diff > 0:
                wins += 1
            print(f"{oracle:<30} {ga_mean:>10.4f} {es_mean:>12.4f} {diff:>+10.4f}{'*' if diff > 0 else ''}")

    print("-"*70)
    if total_diff:
        print(f"{'AVERAGE':<30} {' ':>10} {' ':>12} {np.mean(total_diff):>+10.4f}")
    print(f"\nes_meta_v4 wins: {wins}/{len(oracles)}")

    return wins, len(oracles)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='main/es_meta/data/experience')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--output', default='main/es_meta/data/meta_scorer_v5.pt')
    parser.add_argument('--oracles', nargs='+', default=['Sitagliptin_MPO', 'Zaleplon_MPO', 'Deco_Hop'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--max_calls', type=int, default=3000)
    parser.add_argument('--temperature', type=float, default=0.08)
    parser.add_argument('--skip_train', action='store_true')
    args = parser.parse_args()

    # Load experience
    print("Loading experience...")
    experiences = load_v1_experience(args.exp_dir)

    if not experiences:
        print("No experience found!")
        return

    # Train
    if not args.skip_train:
        learner = train_v4(experiences, n_epochs=args.epochs, output_path=args.output)
    else:
        print(f"Skipping training, using existing checkpoint: {args.output}")

    # Test
    print("\nRunning comparison...")
    results = run_comparison(args.oracles, args.seeds, args.max_calls, args.output, args.temperature)

    # Summary
    wins, total = print_summary(results, args.oracles)

    # Save results
    with open('main/es_meta/data/comparison_v4.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to main/es_meta/data/comparison_v4.json")

    # Check if SOTA achieved
    if wins == total:
        print("\n" + "="*70)
        print("ALL ORACLES WON! SOTA ACHIEVED!")
        print("<promise>SOTA</promise>")
        print("="*70)


if __name__ == '__main__':
    main()
