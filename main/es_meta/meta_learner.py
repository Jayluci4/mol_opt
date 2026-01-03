"""
ES Meta-Learner for seed selection.

Uses Evolution Strategies to meta-learn a seed scorer that predicts
which molecules will perform well as starting populations for different tasks.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple

try:
    from .seed_scorer import SeedScorer
    from .task_embeddings import get_task_embedding
except ImportError:
    from seed_scorer import SeedScorer
    from task_embeddings import get_task_embedding


class Experience:
    """Stores experience from a single optimization run."""

    def __init__(self, oracle_name: str, seed_smiles: List[str],
                 final_scores: List[float], seed_embeddings: torch.Tensor):
        """
        Args:
            oracle_name: Name of the oracle/task
            seed_smiles: SMILES of seed molecules used
            final_scores: Final optimization scores achieved
            seed_embeddings: Precomputed embeddings of seeds
        """
        self.oracle_name = oracle_name
        self.seed_smiles = seed_smiles
        self.final_scores = final_scores
        self.seed_embeddings = seed_embeddings

    def to_dict(self) -> dict:
        return {
            'oracle_name': self.oracle_name,
            'seed_smiles': self.seed_smiles,
            'final_scores': self.final_scores,
            'seed_embeddings': self.seed_embeddings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Experience':
        return cls(
            oracle_name=d['oracle_name'],
            seed_smiles=d['seed_smiles'],
            final_scores=d['final_scores'],
            seed_embeddings=d['seed_embeddings'],
        )


class ESMetaSeedLearner:
    """
    Evolution Strategies-based meta-learner for seed selection.

    Meta-learns a scorer that predicts which molecules will work well
    as starting populations for different optimization tasks.
    """

    def __init__(self, scorer: Optional[SeedScorer] = None,
                 mol_dim: int = 1024, task_dim: int = 9,
                 population_size: int = 10, noise_std: float = 0.02,
                 elite_fraction: float = 0.2, lr: float = 0.1,
                 device: str = 'cpu'):
        """
        Initialize the ES meta-learner.

        Args:
            scorer: Pre-initialized SeedScorer (optional)
            mol_dim: Molecular embedding dimension
            task_dim: Task embedding dimension
            population_size: ES population size
            noise_std: Standard deviation of Gaussian noise for perturbation
            elite_fraction: Fraction of population to select as elites
            lr: Learning rate for meta-update (interpolation toward elites)
            device: Device to use for computation
        """
        if scorer is None:
            scorer = SeedScorer(mol_dim=mol_dim, task_dim=task_dim)

        self.scorer = scorer.to(device)
        self.device = device
        self.population_size = population_size
        self.noise_std = noise_std
        self.n_elites = max(1, int(population_size * elite_fraction))
        self.lr = lr

        # Experience buffer
        self.experience_buffer: List[Experience] = []

        # Training state
        self.is_trained = False
        self.training_history = []

    def perturb_scorer(self) -> SeedScorer:
        """
        Create a perturbed copy of the scorer.

        Returns:
            New scorer with Gaussian noise added to parameters
        """
        perturbed = copy.deepcopy(self.scorer)
        with torch.no_grad():
            for param in perturbed.parameters():
                noise = torch.randn_like(param) * self.noise_std
                param.add_(noise)
        return perturbed.to(self.device)

    def collect_experience(self, oracle_name: str, seed_smiles: List[str],
                           final_scores: List[float], seed_embeddings: torch.Tensor):
        """
        Store experience from an optimization run.

        Args:
            oracle_name: Name of the oracle/task
            seed_smiles: SMILES of seed molecules
            final_scores: Final scores achieved (best scores from optimization)
            seed_embeddings: Precomputed embeddings of seeds
        """
        exp = Experience(
            oracle_name=oracle_name,
            seed_smiles=seed_smiles,
            final_scores=final_scores,
            seed_embeddings=seed_embeddings.to(self.device)
        )
        self.experience_buffer.append(exp)

    def evaluate_scorer(self, scorer: SeedScorer,
                        task_embedding_fn: Callable) -> float:
        """
        Evaluate a scorer's prediction accuracy on collected experience.

        Args:
            scorer: Scorer to evaluate
            task_embedding_fn: Function to get task embedding from oracle name

        Returns:
            Negative MSE (higher is better)
        """
        if not self.experience_buffer:
            return 0.0

        scorer.eval()
        total_error = 0.0

        with torch.no_grad():
            for exp in self.experience_buffer:
                task_emb = task_embedding_fn(exp.oracle_name).to(self.device)
                predicted = scorer(exp.seed_embeddings, task_emb)

                # Normalize actual scores to [0, 1] for comparison
                actual = torch.tensor(exp.final_scores,
                                       dtype=torch.float32, device=self.device)
                if actual.max() > 0:
                    actual = actual / actual.max()

                # Handle size mismatch: use mean of final scores as target
                # Higher final scores = better seeds, so predicted should correlate
                # with the overall optimization success
                target_score = actual.mean()

                # MSE: want predicted scores to be high when optimization succeeded
                # Use the mean final score as proxy for seed quality
                error = ((predicted.mean() - target_score) ** 2).item()
                total_error += error

        # Return negative error (higher is better for selection)
        return -total_error / len(self.experience_buffer)

    def meta_update(self, task_embedding_fn: Optional[Callable] = None) -> float:
        """
        Perform one ES meta-update step.

        Algorithm:
        1. Create population of perturbed scorers
        2. Evaluate each on collected experience
        3. Select top elites
        4. Update main scorer toward elite mean

        Args:
            task_embedding_fn: Function to get task embedding (default: get_task_embedding)

        Returns:
            Best fitness score from this update
        """
        if task_embedding_fn is None:
            task_embedding_fn = get_task_embedding

        if not self.experience_buffer:
            print("Warning: No experience to learn from")
            return 0.0

        # Create population
        population = [self.scorer] + [self.perturb_scorer()
                                       for _ in range(self.population_size - 1)]

        # Evaluate each candidate
        fitness_scores = []
        for scorer in population:
            fitness = self.evaluate_scorer(scorer, task_embedding_fn)
            fitness_scores.append(fitness)

        # Select elites (highest fitness = lowest error)
        elite_indices = np.argsort(fitness_scores)[-self.n_elites:]
        elite_scorers = [population[i] for i in elite_indices]
        best_fitness = fitness_scores[elite_indices[-1]]

        # Update main scorer toward elite mean
        with torch.no_grad():
            for name, param in self.scorer.named_parameters():
                elite_params = [dict(e.named_parameters())[name].data
                                for e in elite_scorers]
                elite_mean = torch.stack(elite_params).mean(dim=0)
                # Interpolate toward elite mean
                param.lerp_(elite_mean, self.lr)

        self.is_trained = True
        self.training_history.append({
            'best_fitness': best_fitness,
            'mean_fitness': np.mean(fitness_scores),
            'n_experience': len(self.experience_buffer)
        })

        return best_fitness

    def meta_train(self, n_epochs: int = 50,
                   task_embedding_fn: Optional[Callable] = None,
                   verbose: bool = True) -> List[float]:
        """
        Run full meta-training loop.

        Args:
            n_epochs: Number of meta-update epochs
            task_embedding_fn: Function to get task embedding
            verbose: Whether to print progress

        Returns:
            List of best fitness values per epoch
        """
        fitness_history = []

        for epoch in range(n_epochs):
            fitness = self.meta_update(task_embedding_fn)
            fitness_history.append(fitness)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: "
                      f"best_fitness={fitness:.4f}, "
                      f"n_experience={len(self.experience_buffer)}")

        return fitness_history

    def score_molecules(self, mol_embeddings: torch.Tensor,
                        task_embedding: torch.Tensor) -> torch.Tensor:
        """
        Score all molecules for a given task.

        Args:
            mol_embeddings: Molecular embeddings (N, mol_dim)
            task_embedding: Task embedding (task_dim,)

        Returns:
            Scores for all molecules (N,)
        """
        self.scorer.eval()
        with torch.no_grad():
            mol_emb = mol_embeddings.to(self.device)
            task_emb = task_embedding.to(self.device)
            return self.scorer(mol_emb, task_emb).cpu()

    def select_top_k(self, mol_embeddings: torch.Tensor,
                     task_embedding: torch.Tensor, k: int) -> torch.Tensor:
        """
        Select indices of top-k scoring molecules.

        Args:
            mol_embeddings: All molecular embeddings
            task_embedding: Task embedding
            k: Number of molecules to select

        Returns:
            Indices of top-k molecules
        """
        scores = self.score_molecules(mol_embeddings, task_embedding)
        return torch.topk(scores, k).indices

    def save(self, path: str):
        """Save meta-learner state to file."""
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
        print(f"Saved meta-learner to {path}")

    def load(self, path: str):
        """Load meta-learner state from file."""
        state = torch.load(path, map_location=self.device)
        self.scorer.load_state_dict(state['scorer_state'])
        self.is_trained = state['is_trained']
        self.training_history = state['training_history']
        self.experience_buffer = [Experience.from_dict(d)
                                  for d in state['experience_buffer']]

        config = state.get('config', {})
        self.population_size = config.get('population_size', self.population_size)
        self.noise_std = config.get('noise_std', self.noise_std)
        self.n_elites = config.get('n_elites', self.n_elites)
        self.lr = config.get('lr', self.lr)

        print(f"Loaded meta-learner from {path} "
              f"(trained={self.is_trained}, n_exp={len(self.experience_buffer)})")

    def get_statistics(self) -> dict:
        """Get current training statistics."""
        return {
            'is_trained': self.is_trained,
            'n_experience': len(self.experience_buffer),
            'training_history': self.training_history,
            'scorer_params': self.scorer.num_params(),
        }


if __name__ == '__main__':
    # Test the meta-learner
    print("Testing ESMetaSeedLearner:")

    learner = ESMetaSeedLearner(population_size=5, noise_std=0.02)
    print(f"  Scorer params: {learner.scorer.num_params()}")

    # Simulate some experience
    for i, oracle in enumerate(['QED', 'GSK3B', 'JNK3']):
        n_seeds = 10
        seed_emb = torch.randn(n_seeds, 1024)
        scores = np.random.rand(n_seeds).tolist()
        learner.collect_experience(oracle, [f'SMILES_{j}' for j in range(n_seeds)],
                                   scores, seed_emb)

    print(f"  Experience buffer size: {len(learner.experience_buffer)}")

    # Test meta-update
    fitness = learner.meta_update()
    print(f"  Meta-update fitness: {fitness:.4f}")

    # Test scoring
    test_emb = torch.randn(100, 1024)
    task_emb = get_task_embedding('QED')
    scores = learner.score_molecules(test_emb, task_emb)
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test selection
    top_k = learner.select_top_k(test_emb, task_emb, k=10)
    print(f"  Top-10 indices: {top_k.tolist()}")

    print("\n  Stats:", learner.get_statistics())
