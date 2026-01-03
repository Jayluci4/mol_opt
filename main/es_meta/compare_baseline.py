"""
Quick comparison: es_meta (trained) vs graph_ga (random seeds).

Runs both on the same oracle with limited budget to compare sample efficiency.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tdc import Oracle


def run_comparison(oracle_name: str, max_calls: int = 1000, seed: int = 42):
    """
    Compare es_meta vs graph_ga on the same oracle.
    """
    import yaml

    print(f"\n{'='*60}")
    print(f"Comparing on: {oracle_name} (max {max_calls} calls)")
    print(f"{'='*60}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'hparams_default.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = {}

    # Test 1: graph_ga (random seeds)
    print("\n--- Running graph_ga (random seeds) ---")
    try:
        from main.graph_ga.run import GB_GA_Optimizer

        class Args:
            method = 'graph_ga'
            smi_file = None
            n_jobs = -1
            output_dir = 'main/es_meta/data/results'
            patience = 3
            max_oracle_calls = max_calls
            freq_log = 100
            log_results = False
            pool = None

        os.makedirs(Args.output_dir, exist_ok=True)
        np.random.seed(seed)

        oracle = Oracle(name=oracle_name)
        optimizer = GB_GA_Optimizer(args=Args())
        optimizer.optimize(oracle=oracle, config=config, seed=seed)

        # Get results
        optimizer.sort_buffer()
        top10 = [item[1][0] for item in list(optimizer.mol_buffer.items())[:10]]
        results['graph_ga'] = {
            'top1': top10[0] if top10 else 0,
            'top10_avg': np.mean(top10) if top10 else 0,
            'n_calls': len(optimizer.oracle)
        }
        print(f"  Top-1: {results['graph_ga']['top1']:.4f}")
        print(f"  Top-10 avg: {results['graph_ga']['top10_avg']:.4f}")
        print(f"  Oracle calls: {results['graph_ga']['n_calls']}")

    except Exception as e:
        print(f"  Error: {e}")
        results['graph_ga'] = None

    # Test 2: es_meta (trained scorer)
    print("\n--- Running es_meta (trained scorer) ---")
    try:
        from main.es_meta.run import ES_Meta_Optimizer

        class Args2:
            method = 'es_meta'
            smi_file = None
            n_jobs = -1
            output_dir = 'main/es_meta/data/results'
            patience = 3
            max_oracle_calls = max_calls
            freq_log = 100
            log_results = False
            meta_checkpoint = 'main/es_meta/data/meta_scorer.pt'
            embedding_cache = None
            collect_experience = False
            experience_dir = 'main/es_meta/data/experience'
            pool = None

        np.random.seed(seed)

        oracle2 = Oracle(name=oracle_name)
        optimizer2 = ES_Meta_Optimizer(args=Args2())
        optimizer2.optimize(oracle=oracle2, config=config, seed=seed)

        # Get results
        optimizer2.sort_buffer()
        top10 = [item[1][0] for item in list(optimizer2.mol_buffer.items())[:10]]
        results['es_meta'] = {
            'top1': top10[0] if top10 else 0,
            'top10_avg': np.mean(top10) if top10 else 0,
            'n_calls': len(optimizer2.oracle)
        }
        print(f"  Top-1: {results['es_meta']['top1']:.4f}")
        print(f"  Top-10 avg: {results['es_meta']['top10_avg']:.4f}")
        print(f"  Oracle calls: {results['es_meta']['n_calls']}")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results['es_meta'] = None

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    if results.get('graph_ga') and results.get('es_meta'):
        diff = results['es_meta']['top10_avg'] - results['graph_ga']['top10_avg']
        print(f"  graph_ga top-10: {results['graph_ga']['top10_avg']:.4f}")
        print(f"  es_meta top-10:  {results['es_meta']['top10_avg']:.4f}")
        print(f"  Difference: {diff:+.4f} ({'better' if diff > 0 else 'worse'})")
    else:
        print("  Comparison incomplete due to errors")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', type=str, default='QED')
    parser.add_argument('--max_calls', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_comparison(args.oracle, args.max_calls, args.seed)
