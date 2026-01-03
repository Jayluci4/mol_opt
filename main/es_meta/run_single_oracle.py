"""
Run single oracle comparison - isolated execution to avoid caching issues.
"""
import os
import sys
import argparse
import numpy as np
import yaml
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tdc import Oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['graph_ga', 'es_meta'])
    parser.add_argument('--oracle', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_calls', type=int, default=5000)
    parser.add_argument('--meta_checkpoint', type=str, default='main/es_meta/data/meta_scorer_v2.pt')
    args = parser.parse_args()

    # Load config
    config_path = 'main/es_meta/hparams_default.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    class OptArgs:
        method = args.method
        smi_file = None
        n_jobs = -1
        output_dir = 'main/es_meta/data/benchmark_results'
        patience = 5
        max_oracle_calls = args.max_calls
        freq_log = 500
        log_results = False
        pool = None
        meta_checkpoint = args.meta_checkpoint
        embedding_cache = 'main/es_meta/data/zinc_embeddings.pt'
        collect_experience = False
        experience_dir = 'main/es_meta/data/experience'

    os.makedirs(OptArgs.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    oracle = Oracle(name=args.oracle)

    if args.method == 'graph_ga':
        from main.graph_ga.run import GB_GA_Optimizer
        optimizer = GB_GA_Optimizer(args=OptArgs())
    else:
        from main.es_meta.run import ES_Meta_Optimizer
        optimizer = ES_Meta_Optimizer(args=OptArgs())

    optimizer.optimize(oracle=oracle, config=config, seed=args.seed)
    optimizer.sort_buffer()

    items = list(optimizer.mol_buffer.items())
    top1 = items[0][1][0] if items else 0
    top10 = [item[1][0] for item in items[:10]]
    top10_avg = np.mean(top10) if top10 else 0

    result = {
        'method': args.method,
        'oracle': args.oracle,
        'seed': args.seed,
        'top1': float(top1),
        'top10_avg': float(top10_avg),
        'n_calls': len(optimizer.oracle)
    }

    print(json.dumps(result))


if __name__ == '__main__':
    main()
