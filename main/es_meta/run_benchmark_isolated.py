"""
Full benchmark using isolated subprocess runs to avoid caching.
"""
import os
import sys
import json
import subprocess
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main.es_meta.task_embeddings import TEST_ORACLES


def run_isolated(method, oracle, seed, max_calls, meta_checkpoint):
    """Run single optimization in isolated subprocess."""
    cmd = [
        sys.executable, 'main/es_meta/run_single_oracle.py',
        '--method', method,
        '--oracle', oracle,
        '--seed', str(seed),
        '--max_calls', str(max_calls),
        '--meta_checkpoint', meta_checkpoint
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
            cwd='/home/jayantlohia16/mol_opt'
        )
        # Find JSON in output (last line)
        for line in result.stdout.strip().split('\n')[::-1]:
            if line.startswith('{'):
                return json.loads(line)
        return {'error': 'No JSON output', 'stderr': result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {'error': 'Timeout'}
    except Exception as e:
        return {'error': str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracles', nargs='+', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--max_calls', type=int, default=5000)
    parser.add_argument('--meta_checkpoint', type=str, default='main/es_meta/data/meta_scorer_v2.pt')
    parser.add_argument('--output', type=str, default='main/es_meta/data/benchmark_v2.json')
    args = parser.parse_args()

    oracles = args.oracles if args.oracles else TEST_ORACLES

    print("="*70)
    print("ES META BENCHMARK (Isolated Runs)")
    print("="*70)
    print(f"Oracles: {len(oracles)}")
    print(f"Seeds: {args.seeds}")
    print(f"Max calls: {args.max_calls}")
    print("="*70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'graph_ga': {},
        'es_meta': {}
    }

    total = len(oracles) * len(args.seeds) * 2
    count = 0

    for oracle in oracles:
        results['graph_ga'][oracle] = []
        results['es_meta'][oracle] = []

        for seed in args.seeds:
            for method in ['graph_ga', 'es_meta']:
                count += 1
                print(f"\n[{count}/{total}] {method} on {oracle} (seed={seed})")

                res = run_isolated(method, oracle, seed, args.max_calls, args.meta_checkpoint)

                if 'error' not in res:
                    print(f"  Top-1: {res['top1']:.4f}, Top-10: {res['top10_avg']:.4f}")
                    results[method][oracle].append(res)
                else:
                    print(f"  Error: {res['error']}")
                    results[method][oracle].append(res)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Oracle':<30} {'graph_ga':>10} {'es_meta':>10} {'Diff':>10}")
    print("-"*70)

    wins = 0
    for oracle in oracles:
        ga = [r['top10_avg'] for r in results['graph_ga'][oracle] if 'top10_avg' in r]
        es = [r['top10_avg'] for r in results['es_meta'][oracle] if 'top10_avg' in r]

        if ga and es:
            ga_mean = np.mean(ga)
            es_mean = np.mean(es)
            diff = es_mean - ga_mean
            if diff > 0:
                wins += 1
            print(f"{oracle:<30} {ga_mean:>10.4f} {es_mean:>10.4f} {diff:>+10.4f}{'*' if diff > 0 else ''}")

    print("-"*70)
    print(f"es_meta wins: {wins}/{len(oracles)}")


if __name__ == '__main__':
    main()
