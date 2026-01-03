"""
Full benchmark: es_meta vs graph_ga on all test oracles.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
from tdc import Oracle

from main.es_meta.task_embeddings import TEST_ORACLES


def run_single(method, oracle_name, config, seed, max_calls, meta_checkpoint=None):
    """Run a single optimization and return results."""
    import importlib
    import gc

    # Force reimport to clear any cached state
    import main.graph_ga.run
    import main.es_meta.run
    importlib.reload(main.graph_ga.run)
    importlib.reload(main.es_meta.run)

    from main.graph_ga.run import GB_GA_Optimizer
    from main.es_meta.run import ES_Meta_Optimizer

    # Clear memory
    gc.collect()

    class Args:
        smi_file = None
        n_jobs = -1
        output_dir = 'main/es_meta/data/benchmark_results'
        patience = 5
        max_oracle_calls = max_calls
        freq_log = 500
        log_results = False
        pool = None

    Args.method = method
    os.makedirs(Args.output_dir, exist_ok=True)

    if method == 'es_meta':
        Args.meta_checkpoint = meta_checkpoint
        Args.embedding_cache = 'main/es_meta/data/zinc_embeddings.pt'
        Args.collect_experience = False
        Args.experience_dir = 'main/es_meta/data/experience'

    np.random.seed(seed)
    oracle = Oracle(name=oracle_name)

    if method == 'graph_ga':
        optimizer = GB_GA_Optimizer(args=Args())
    else:
        optimizer = ES_Meta_Optimizer(args=Args())

    optimizer.optimize(oracle=oracle, config=config, seed=seed)
    optimizer.sort_buffer()

    # Get results
    items = list(optimizer.mol_buffer.items())
    top1 = items[0][1][0] if items else 0
    top10 = [item[1][0] for item in items[:10]]
    top10_avg = np.mean(top10) if top10 else 0

    return {
        'top1': float(top1),
        'top10_avg': float(top10_avg),
        'n_calls': len(optimizer.oracle)
    }


def run_benchmark(oracles, seeds, max_calls, meta_checkpoint):
    """Run full benchmark."""

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'hparams_default.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = {
        'timestamp': datetime.now().isoformat(),
        'max_calls': max_calls,
        'seeds': seeds,
        'oracles': oracles,
        'graph_ga': {},
        'es_meta': {}
    }

    total_runs = len(oracles) * len(seeds) * 2
    run_count = 0

    for oracle_name in oracles:
        results['graph_ga'][oracle_name] = []
        results['es_meta'][oracle_name] = []

        for seed in seeds:
            # Run graph_ga
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] graph_ga on {oracle_name} (seed={seed})")
            try:
                res = run_single('graph_ga', oracle_name, config, seed, max_calls)
                results['graph_ga'][oracle_name].append(res)
                print(f"  Top-1: {res['top1']:.4f}, Top-10: {res['top10_avg']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                results['graph_ga'][oracle_name].append({'error': str(e)})

            # Run es_meta
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] es_meta on {oracle_name} (seed={seed})")
            try:
                res = run_single('es_meta', oracle_name, config, seed, max_calls, meta_checkpoint)
                results['es_meta'][oracle_name].append(res)
                print(f"  Top-1: {res['top1']:.4f}, Top-10: {res['top10_avg']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                results['es_meta'][oracle_name].append({'error': str(e)})

    return results


def summarize_results(results):
    """Print summary table."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(f"{'Oracle':<25} {'graph_ga':>12} {'es_meta':>12} {'Diff':>10} {'%':>8}")
    print("-"*70)

    total_ga, total_es = [], []

    for oracle in results['oracles']:
        ga_scores = [r['top10_avg'] for r in results['graph_ga'].get(oracle, []) if 'top10_avg' in r]
        es_scores = [r['top10_avg'] for r in results['es_meta'].get(oracle, []) if 'top10_avg' in r]

        if ga_scores and es_scores:
            ga_mean = np.mean(ga_scores)
            es_mean = np.mean(es_scores)
            diff = es_mean - ga_mean
            pct = (diff / ga_mean * 100) if ga_mean > 0 else 0

            total_ga.append(ga_mean)
            total_es.append(es_mean)

            winner = "**" if diff > 0 else ""
            print(f"{oracle:<25} {ga_mean:>12.4f} {es_mean:>12.4f} {diff:>+10.4f} {pct:>+7.1f}%{winner}")
        else:
            print(f"{oracle:<25} {'error':>12} {'error':>12}")

    print("-"*70)
    if total_ga and total_es:
        ga_avg = np.mean(total_ga)
        es_avg = np.mean(total_es)
        diff_avg = es_avg - ga_avg
        pct_avg = (diff_avg / ga_avg * 100) if ga_avg > 0 else 0
        print(f"{'AVERAGE':<25} {ga_avg:>12.4f} {es_avg:>12.4f} {diff_avg:>+10.4f} {pct_avg:>+7.1f}%")
    print("="*70)

    # Win rate
    wins = sum(1 for g, e in zip(total_ga, total_es) if e > g)
    print(f"\nes_meta wins: {wins}/{len(total_ga)} oracles")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracles', nargs='+', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--max_calls', type=int, default=5000)
    parser.add_argument('--meta_checkpoint', type=str, default='main/es_meta/data/meta_scorer_v2.pt')
    parser.add_argument('--output', type=str, default='main/es_meta/data/benchmark_results.json')
    args = parser.parse_args()

    oracles = args.oracles if args.oracles else TEST_ORACLES

    print("="*70)
    print("ES META BENCHMARK")
    print("="*70)
    print(f"Oracles: {oracles}")
    print(f"Seeds: {args.seeds}")
    print(f"Max calls: {args.max_calls}")
    print(f"Meta checkpoint: {args.meta_checkpoint}")
    print("="*70)

    results = run_benchmark(oracles, args.seeds, args.max_calls, args.meta_checkpoint)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary
    summarize_results(results)
