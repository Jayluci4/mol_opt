"""
Improved ES Meta-Learner with:
1. Oracle-specific task embeddings
2. Stochastic seed sampling
3. Better training objective
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Callable

try:
    from .seed_scorer_v2 import SeedScorerV2
    from .task_embeddings_v2 import get_task_embedding_v2, get_embedding_dim
except ImportError:
    from seed_scorer_v2 import SeedScorerV2
    from task_embeddings_v2 import get_task_embedding_v2, get_embedding_dim


class ExperienceV2:
    """Experience with oracle-specific embeddings."""

    def __init__(self, oracle_name: str, seed_smiles: List[str],
                 seed_embeddings: torch.Tensor, final_top10_avg: float):
        self.oracle_name = oracle_name
        self.seed_smiles = seed_smiles
        self.seed_embeddings = seed_embeddings
        self.final_top10_avg = final_top10_avg  # Single scalar target

    def to_dict(self):
        return {
            'oracle_name': self.oracle_name,
            'seed_smiles': self.seed_smiles,
            'seed_embeddings': self.seed_embeddings,
            'final_top10_avg': self.final_top10_avg,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            oracle_name=d['oracle_name'],
            seed_smiles=d['seed_smiles'],
            seed_embeddings=d['seed_embeddings'],
            final_top10_avg=d['final_top10_avg'],
        )


class ESMetaLearnerV2:
    """
    Improved ES meta-learner.

    Key improvements:
    1. Uses oracle-specific embeddings (target FP + category)
    2. Stochastic seed sampling for exploration
    3. Direct optimization of final score
    """

    def __init__(self, scorer: Optional[SeedScorerV2] = None,
                 population_size: int = 20, noise_std: float = 0.02,
                 elite_fraction: float = 0.2, lr: float = 0.1,
                 device: str = 'cpu'):

        if scorer is None:
            scorer = SeedScorerV2(mol_dim=1024, task_dim=get_embedding_dim())

        self.scorer = scorer.to(device)
        self.device = device
        self.population_size = population_size
        self.noise_std = noise_std
        self.n_elites = max(1, int(population_size * elite_fraction))
        self.lr = lr

        self.experience_buffer: List[ExperienceV2] = []
        self.is_trained = False
        self.training_history = []

    def perturb_scorer(self) -> SeedScorerV2:
        """Create perturbed copy."""
        perturbed = copy.deepcopy(self.scorer)
        with torch.no_grad():
            for param in perturbed.parameters():
                noise = torch.randn_like(param) * self.noise_std
                param.add_(noise)
        return perturbed.to(self.device)

    def collect_experience(self, oracle_name: str, seed_smiles: List[str],
                           seed_embeddings: torch.Tensor, final_top10_avg: float):
        """Store experience."""
        exp = ExperienceV2(
            oracle_name=oracle_name,
            seed_smiles=seed_smiles,
            seed_embeddings=seed_embeddings.to(self.device),
            final_top10_avg=final_top10_avg,
        )
        self.experience_buffer.append(exp)

    def evaluate_scorer(self, scorer: SeedScorerV2) -> float:
        """
        Evaluate scorer on collected experience.

        Objective: predicted seed quality should correlate with final score.
        """
        if not self.experience_buffer:
            return 0.0

        scorer.eval()
        total_correlation = 0.0

        with torch.no_grad():
            for exp in self.experience_buffer:
                task_emb = get_task_embedding_v2(exp.oracle_name).to(self.device)
                predicted = scorer(exp.seed_embeddings, task_emb)

                # Mean predicted score should correlate with final performance
                pred_mean = predicted.mean().item()
                target = exp.final_top10_avg

                # Simple correlation proxy: minimize squared difference
                # (Higher final score should have higher predicted mean)
                correlation = -((pred_mean - target) ** 2)
                total_correlation += correlation

        return total_correlation / len(self.experience_buffer)

    def meta_update(self) -> float:
        """Perform one ES meta-update."""
        if not self.experience_buffer:
            return 0.0

        # Create population
        population = [self.scorer] + [self.perturb_scorer()
                                       for _ in range(self.population_size - 1)]

        # Evaluate
        fitness_scores = [self.evaluate_scorer(s) for s in population]

        # Select elites
        elite_indices = np.argsort(fitness_scores)[-self.n_elites:]
        elite_scorers = [population[i] for i in elite_indices]
        best_fitness = fitness_scores[elite_indices[-1]]

        # Update toward elite mean
        with torch.no_grad():
            for name, param in self.scorer.named_parameters():
                elite_params = [dict(e.named_parameters())[name].data for e in elite_scorers]
                elite_mean = torch.stack(elite_params).mean(dim=0)
                param.lerp_(elite_mean, self.lr)

        self.is_trained = True
        self.training_history.append({
            'best_fitness': best_fitness,
            'mean_fitness': np.mean(fitness_scores),
        })

        return best_fitness

    def meta_train(self, n_epochs: int = 100, verbose: bool = True) -> List[float]:
        """Run meta-training."""
        fitness_history = []
        for epoch in range(n_epochs):
            fitness = self.meta_update()
            fitness_history.append(fitness)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: fitness={fitness:.4f}")
        return fitness_history

    def select_seeds(self, mol_embeddings: torch.Tensor, oracle_name: str,
                     k: int, temperature: float = 0.1, stochastic: bool = True) -> torch.Tensor:
        """
        Select k seed molecules for an oracle.

        Args:
            mol_embeddings: All available molecule embeddings
            oracle_name: Name of the oracle
            k: Number of seeds to select
            temperature: Sampling temperature (lower = more deterministic)
            stochastic: Whether to use stochastic sampling

        Returns:
            Indices of selected molecules
        """
        self.scorer.eval()
        task_emb = get_task_embedding_v2(oracle_name).to(self.device)
        mol_emb = mol_embeddings.to(self.device)

        with torch.no_grad():
            indices = self.scorer.sample_top_k(mol_emb, task_emb, k,
                                                temperature=temperature,
                                                stochastic=stochastic)
        return indices.cpu()

    def save(self, path: str):
        """Save meta-learner."""
        state = {
            'scorer_state': self.scorer.state_dict(),
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'experience_buffer': [exp.to_dict() for exp in self.experience_buffer],
            'config': {
                'population_size': self.population_size,
                'noise_std': self.noise_std,
                'n_elites': self.n_elites,
                'lr': self.lr,
            }
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(state, path)

    def load(self, path: str):
        """Load meta-learner."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.scorer.load_state_dict(state['scorer_state'])
        self.is_trained = state['is_trained']
        self.training_history = state['training_history']
        self.experience_buffer = [ExperienceV2.from_dict(d) for d in state['experience_buffer']]


if __name__ == '__main__':
    print("Testing ESMetaLearnerV2:")

    learner = ESMetaLearnerV2(population_size=5)
    print(f"  Scorer params: {learner.scorer.num_params()}")

    # Simulate experience
    for oracle in ['QED', 'Sitagliptin_MPO', 'Zaleplon_MPO']:
        seed_emb = torch.randn(120, 1024)
        score = np.random.uniform(0.3, 0.9)
        learner.collect_experience(oracle, [f'smi_{i}' for i in range(120)], seed_emb, score)

    print(f"  Experience: {len(learner.experience_buffer)}")

    # Meta-update
    fitness = learner.meta_update()
    print(f"  Meta-update fitness: {fitness:.4f}")

    # Test selection
    test_emb = torch.randn(1000, 1024)
    idx1 = learner.select_seeds(test_emb, 'Sitagliptin_MPO', k=10, stochastic=True)
    idx2 = learner.select_seeds(test_emb, 'Sitagliptin_MPO', k=10, stochastic=True)
    idx3 = learner.select_seeds(test_emb, 'Zaleplon_MPO', k=10, stochastic=True)

    print(f"  Sitagliptin seeds (run 1): {idx1.tolist()}")
    print(f"  Sitagliptin seeds (run 2): {idx2.tolist()}")
    print(f"  Zaleplon seeds: {idx3.tolist()}")
    print(f"  Different seeds for same oracle: {not torch.equal(idx1, idx2)}")
    print(f"  Different seeds for diff oracle: {not torch.equal(idx1, idx3)}")
