"""
ES Meta Optimizer V3 - with contrastive-trained scorer.
"""

from __future__ import print_function

import os
import random
from typing import List

import joblib
import numpy as np
import torch
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

import main.graph_ga.crossover as co
import main.graph_ga.mutate as mu
from main.optimizer import BaseOptimizer

try:
    from .meta_learner_v3 import ESMetaLearnerV3
    from .task_embeddings_v2 import get_task_embedding_v2
    from .mol_embeddings import EmbeddingCache
except ImportError:
    from meta_learner_v3 import ESMetaLearnerV3
    from task_embeddings_v2 import get_task_embedding_v2
    from mol_embeddings import EmbeddingCache


MINIMUM = 1e-10


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


class ES_Meta_Optimizer_V3(BaseOptimizer):
    """
    ES Meta Optimizer V3 with contrastive-trained scorer.
    """

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "es_meta_v3"

        self.meta_checkpoint = getattr(args, 'meta_checkpoint', None)
        self.embedding_cache_path = getattr(args, 'embedding_cache', None)
        self.collect_experience = getattr(args, 'collect_experience', False)
        self.experience_dir = getattr(args, 'experience_dir', 'data/experience')
        self.temperature = getattr(args, 'temperature', 0.1)
        self.stochastic = getattr(args, 'stochastic', True)

        # Initialize meta-learner
        self.meta_learner = ESMetaLearnerV3()

        # Load checkpoint if provided
        if self.meta_checkpoint and os.path.exists(self.meta_checkpoint):
            self.meta_learner.load(self.meta_checkpoint)
            print(f"Loaded meta-learner V3 from {self.meta_checkpoint}")

        self._embedding_cache = None

    @property
    def embedding_cache(self):
        if self._embedding_cache is None:
            cache_path = self.embedding_cache_path or 'main/es_meta/data/zinc_embeddings.pt'
            print(f"Initializing embedding cache...")
            self._embedding_cache = EmbeddingCache(
                self.all_smiles,
                cache_path=cache_path,
                radius=2,
                n_bits=1024
            )
        return self._embedding_cache

    def select_seeds(self, oracle_name: str, population_size: int) -> List[str]:
        """Select seeds using meta-learned scorer with stochastic sampling."""
        if not self.meta_learner.is_trained:
            print("Meta-learner not trained, using random selection")
            return list(np.random.choice(self.all_smiles, population_size, replace=False))

        all_embeddings = self.embedding_cache.get_all_embeddings()

        top_indices = self.meta_learner.select_seeds(
            all_embeddings, oracle_name, k=population_size,
            temperature=self.temperature, stochastic=self.stochastic
        )

        seeds = [self.all_smiles[i] for i in top_indices.tolist()]
        print(f"Selected {len(seeds)} seeds using meta-learned scorer V3 (T={self.temperature})")
        return seeds

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        oracle_name = oracle.name if hasattr(oracle, 'name') else str(oracle)

        pool = joblib.Parallel(n_jobs=self.n_jobs)

        # Select seeds
        if self.smi_file is not None:
            starting_population = self.all_smiles[:config["population_size"]]
        elif self.meta_learner.is_trained:
            starting_population = self.select_seeds(oracle_name, config["population_size"])
        else:
            starting_population = list(np.random.choice(
                self.all_smiles, config["population_size"], replace=False
            ))

        initial_seeds = starting_population.copy()

        # Initialize population
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_mol = [m for m in population_mol if m is not None]
        population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

        patience = 0

        # Main loop
        while True:
            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
            offspring_mol = pool(delayed(reproduce)(mating_pool, config["mutation_rate"])
                                 for _ in range(config["offspring_size"]))

            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            if len(self.oracle) > 100:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('Convergence criteria met, stopping...')
                        break
                else:
                    patience = 0

            if self.finish:
                break

        # Collect experience
        if self.collect_experience:
            self._save_experience(oracle_name, initial_seeds)

    def _save_experience(self, oracle_name: str, initial_seeds: List[str]):
        """Save experience for meta-training."""
        self.sort_buffer()
        top_items = list(self.mol_buffer.items())[:100]
        final_scores = [item[1][0] for item in top_items]
        final_top10_avg = np.mean(final_scores[:10])

        seed_embeddings = self.embedding_cache.get_embeddings(initial_seeds)

        self.meta_learner.collect_experience(
            oracle_name=oracle_name,
            seed_smiles=initial_seeds,
            seed_embeddings=seed_embeddings,
            final_top10_avg=final_top10_avg,
            final_scores=final_scores,
        )

        os.makedirs(self.experience_dir, exist_ok=True)
        exp_path = os.path.join(self.experience_dir, f'{oracle_name.lower()}_exp_v3.pt')
        torch.save({
            'oracle_name': oracle_name,
            'seed_smiles': initial_seeds,
            'seed_embeddings': seed_embeddings,
            'final_scores': final_scores,
            'final_top10_avg': final_top10_avg,
        }, exp_path)
        print(f"Saved experience to {exp_path}")


if __name__ == '__main__':
    print("ES Meta Optimizer V3 loaded")
