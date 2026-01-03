"""
ES Meta-Learning Optimizer for PMO benchmark.

Extends the graph_ga approach with meta-learned seed selection.
Instead of random seed selection from ZINC, uses a learned scorer
that predicts which molecules will work well for each task.
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
from main.es_meta.meta_learner import ESMetaSeedLearner
from main.es_meta.seed_scorer import SeedScorer
from main.es_meta.task_embeddings import get_task_embedding
from main.es_meta.mol_embeddings import get_zinc_embeddings, EmbeddingCache


MINIMUM = 1e-10


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights.
    """
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """Crossover and mutate to produce offspring."""
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


class ES_Meta_Optimizer(BaseOptimizer):
    """
    ES Meta-Learning Optimizer.

    Uses meta-learned seed selection instead of random selection,
    then runs the same genetic algorithm as graph_ga.
    """

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "es_meta"

        # Meta-learner configuration
        self.meta_checkpoint = getattr(args, 'meta_checkpoint', None)
        self.embedding_cache_path = getattr(args, 'embedding_cache', None)
        self.collect_experience = getattr(args, 'collect_experience', False)
        self.experience_dir = getattr(args, 'experience_dir', 'data/experience')

        # Initialize meta-learner
        self.meta_learner = ESMetaSeedLearner(
            mol_dim=1024,
            task_dim=9,
            population_size=10,
            noise_std=0.02,
            elite_fraction=0.2,
        )

        # Load meta-checkpoint if provided
        if self.meta_checkpoint and os.path.exists(self.meta_checkpoint):
            self.meta_learner.load(self.meta_checkpoint)
            print(f"Loaded meta-learner from {self.meta_checkpoint}")

        # Embedding cache (computed lazily)
        self._embedding_cache = None

    @property
    def embedding_cache(self):
        """Lazy initialization of embedding cache."""
        if self._embedding_cache is None:
            cache_path = self.embedding_cache_path or 'main/es_meta/data/zinc_embeddings.pt'
            print(f"Initializing embedding cache (this may take a while)...")
            self._embedding_cache = EmbeddingCache(
                self.all_smiles,
                cache_path=cache_path,
                radius=2,
                n_bits=1024
            )
        return self._embedding_cache

    def select_seeds(self, oracle_name: str, population_size: int) -> List[str]:
        """
        Select seed molecules using meta-learned scorer.

        Args:
            oracle_name: Name of the oracle/task
            population_size: Number of seeds to select

        Returns:
            List of SMILES strings for starting population
        """
        if not self.meta_learner.is_trained:
            # Fall back to random selection if not trained
            print("Meta-learner not trained, using random selection")
            return list(np.random.choice(self.all_smiles, population_size, replace=False))

        # Get task embedding
        task_emb = get_task_embedding(oracle_name)

        # Get all molecule embeddings
        all_embeddings = self.embedding_cache.get_all_embeddings()

        # Score and select top-k
        top_indices = self.meta_learner.select_top_k(
            all_embeddings, task_emb, k=population_size
        )

        # Get corresponding SMILES
        seeds = [self.all_smiles[i] for i in top_indices.tolist()]
        print(f"Selected {len(seeds)} seeds using meta-learned scorer")
        return seeds

    def _optimize(self, oracle, config):
        """
        Main optimization loop.

        Same as graph_ga but with meta-learned seed selection.
        """
        self.oracle.assign_evaluator(oracle)
        oracle_name = oracle.name if hasattr(oracle, 'name') else str(oracle)

        pool = joblib.Parallel(n_jobs=self.n_jobs)

        # KEY DIFFERENCE: Use meta-learned selection instead of random
        if self.smi_file is not None:
            # Exploitation run (use provided SMILES file)
            starting_population = self.all_smiles[:config["population_size"]]
        elif self.meta_learner.is_trained:
            # Meta-learned selection
            starting_population = self.select_seeds(oracle_name, config["population_size"])
        else:
            # Random selection (fallback)
            starting_population = list(np.random.choice(
                self.all_smiles, config["population_size"], replace=False
            ))

        # Store starting population for experience collection
        initial_seeds = starting_population.copy()

        # Initialize population
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_mol = [m for m in population_mol if m is not None]
        population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

        patience = 0

        # Main optimization loop (same as graph_ga)
        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            # Generate offspring
            mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
            offspring_mol = pool(delayed(reproduce)(mating_pool, config["mutation_rate"])
                                 for _ in range(config["offspring_size"]))

            # Add offspring to population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # Score and select
            old_scores = population_scores
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # Early stopping check
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
                old_score = new_score

            if self.finish:
                break

        # Collect experience for meta-training
        if self.collect_experience:
            self._save_experience(oracle_name, initial_seeds)

    def _save_experience(self, oracle_name: str, initial_seeds: List[str]):
        """
        Save experience for meta-training.

        Stores the seeds used and the final optimization scores.
        """
        # Get final top scores
        self.sort_buffer()
        top_items = list(self.mol_buffer.items())[:100]
        final_scores = [item[1][0] for item in top_items]

        # Get seed embeddings
        seed_embeddings = self.embedding_cache.get_embeddings(initial_seeds)

        # Collect experience
        self.meta_learner.collect_experience(
            oracle_name=oracle_name,
            seed_smiles=initial_seeds,
            final_scores=final_scores[:len(initial_seeds)],
            seed_embeddings=seed_embeddings
        )

        # Save experience to disk
        os.makedirs(self.experience_dir, exist_ok=True)
        exp_path = os.path.join(self.experience_dir, f'{oracle_name}_experience.pt')
        torch.save({
            'oracle_name': oracle_name,
            'seed_smiles': initial_seeds,
            'final_scores': final_scores,
            'seed_embeddings': seed_embeddings,
        }, exp_path)
        print(f"Saved experience to {exp_path}")


class ES_Meta_Collector(ES_Meta_Optimizer):
    """
    Experience collector variant.

    Runs random selection (like graph_ga) but collects experience
    for later meta-training.
    """

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "es_meta_collector"
        self.collect_experience = True
        # Force random selection during collection
        self.meta_learner.is_trained = False


if __name__ == '__main__':
    # Quick test
    print("ES Meta Optimizer module loaded successfully")
    print(f"Available classes: ES_Meta_Optimizer, ES_Meta_Collector")
