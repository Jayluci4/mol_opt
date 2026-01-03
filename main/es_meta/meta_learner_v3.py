"""
ES Meta-Learner V3 with contrastive learning objective.

Key insight: Learn which seeds led to BETTER outcomes by comparing experiences.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    from .seed_scorer_v2 import SeedScorerV2
    from .task_embeddings_v2 import get_task_embedding_v2, get_embedding_dim
except ImportError:
    from seed_scorer_v2 import SeedScorerV2
    from task_embeddings_v2 import get_task_embedding_v2, get_embedding_dim


class ExperienceV3:
    """Experience with relative ranking information."""

    def __init__(self, oracle_name: str, seed_smiles: List[str],
                 seed_embeddings: torch.Tensor, final_top10_avg: float,
                 final_scores: Optional[List[float]] = None):
        self.oracle_name = oracle_name
        self.seed_smiles = seed_smiles
        self.seed_embeddings = seed_embeddings
        self.final_top10_avg = final_top10_avg
        self.final_scores = final_scores or []

    def to_dict(self):
        return {
            'oracle_name': self.oracle_name,
            'seed_smiles': self.seed_smiles,
            'seed_embeddings': self.seed_embeddings,
            'final_top10_avg': self.final_top10_avg,
            'final_scores': self.final_scores,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            oracle_name=d['oracle_name'],
            seed_smiles=d['seed_smiles'],
            seed_embeddings=d['seed_embeddings'],
            final_top10_avg=d['final_top10_avg'],
            final_scores=d.get('final_scores', []),
        )


class ESMetaLearnerV3:
    """
    ES meta-learner with contrastive learning.

    Training objective: For pairs of experiences on the same oracle category,
    predict which seed set led to higher final score.
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

        self.experience_buffer: List[ExperienceV3] = []
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
                           seed_embeddings: torch.Tensor, final_top10_avg: float,
                           final_scores: Optional[List[float]] = None):
        """Store experience."""
        exp = ExperienceV3(
            oracle_name=oracle_name,
            seed_smiles=seed_smiles,
            seed_embeddings=seed_embeddings.to(self.device),
            final_top10_avg=final_top10_avg,
            final_scores=final_scores or [],
        )
        self.experience_buffer.append(exp)

    def evaluate_scorer_contrastive(self, scorer: SeedScorerV2) -> float:
        """
        Contrastive evaluation: for pairs of experiences,
        check if scorer correctly ranks the better one higher.
        """
        if len(self.experience_buffer) < 2:
            return 0.0

        scorer.eval()
        total_correct = 0
        total_pairs = 0

        with torch.no_grad():
            for i, exp1 in enumerate(self.experience_buffer):
                for exp2 in self.experience_buffer[i+1:]:
                    # Get predictions for each experience
                    task_emb1 = get_task_embedding_v2(exp1.oracle_name).to(self.device)
                    task_emb2 = get_task_embedding_v2(exp2.oracle_name).to(self.device)

                    pred1 = scorer(exp1.seed_embeddings, task_emb1).mean().item()
                    pred2 = scorer(exp2.seed_embeddings, task_emb2).mean().item()

                    actual1 = exp1.final_top10_avg
                    actual2 = exp2.final_top10_avg

                    # Check if relative ordering is correct
                    pred_order = pred1 > pred2
                    actual_order = actual1 > actual2

                    if pred_order == actual_order:
                        total_correct += 1
                    total_pairs += 1

        return total_correct / total_pairs if total_pairs > 0 else 0.0

    def evaluate_scorer_regression(self, scorer: SeedScorerV2) -> float:
        """
        Regression evaluation: minimize MSE between predicted and actual.
        """
        if not self.experience_buffer:
            return 0.0

        scorer.eval()
        total_error = 0.0

        with torch.no_grad():
            for exp in self.experience_buffer:
                task_emb = get_task_embedding_v2(exp.oracle_name).to(self.device)
                predicted = scorer(exp.seed_embeddings, task_emb).mean().item()
                target = exp.final_top10_avg
                total_error += (predicted - target) ** 2

        return -total_error / len(self.experience_buffer)

    def evaluate_scorer(self, scorer: SeedScorerV2) -> float:
        """Combined objective: contrastive + regression."""
        contrastive = self.evaluate_scorer_contrastive(scorer)
        regression = self.evaluate_scorer_regression(scorer)

        # Combine: contrastive matters more for ranking, regression for calibration
        return 0.7 * contrastive + 0.3 * (1 + regression)  # Scale regression to ~[0,1]

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
            'contrastive': self.evaluate_scorer_contrastive(self.scorer),
        })

        return best_fitness

    def meta_train(self, n_epochs: int = 100, verbose: bool = True) -> List[float]:
        """Run meta-training."""
        fitness_history = []
        for epoch in range(n_epochs):
            fitness = self.meta_update()
            fitness_history.append(fitness)
            if verbose and (epoch + 1) % 10 == 0:
                contrastive = self.training_history[-1]['contrastive']
                print(f"Epoch {epoch + 1}/{n_epochs}: fitness={fitness:.4f}, contrastive_acc={contrastive:.2%}")
        return fitness_history

    def select_seeds(self, mol_embeddings: torch.Tensor, oracle_name: str,
                     k: int, temperature: float = 0.1, stochastic: bool = True) -> torch.Tensor:
        """Select k seed molecules for an oracle."""
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
        self.experience_buffer = [ExperienceV3.from_dict(d) for d in state['experience_buffer']]


if __name__ == '__main__':
    print("Testing ESMetaLearnerV3:")

    learner = ESMetaLearnerV3(population_size=10)
    print(f"  Scorer params: {learner.scorer.num_params()}")

    # Simulate experience with varying scores
    test_cases = [
        ('QED', 0.94),
        ('Sitagliptin_MPO', 0.45),
        ('Zaleplon_MPO', 0.52),
        ('GSK3B', 0.91),
        ('DRD2', 0.88),
    ]

    for oracle, score in test_cases:
        seed_emb = torch.randn(120, 1024)
        learner.collect_experience(oracle, [f'smi_{i}' for i in range(120)], seed_emb, score)

    print(f"  Experience: {len(learner.experience_buffer)}")

    # Meta-train
    for epoch in range(30):
        fitness = learner.meta_update()
        if (epoch + 1) % 10 == 0:
            contrastive = learner.training_history[-1]['contrastive']
            print(f"  Epoch {epoch+1}: fitness={fitness:.4f}, contrastive={contrastive:.2%}")

    # Test selection
    test_emb = torch.randn(1000, 1024)
    idx1 = learner.select_seeds(test_emb, 'Sitagliptin_MPO', k=10, stochastic=True)
    idx2 = learner.select_seeds(test_emb, 'Sitagliptin_MPO', k=10, stochastic=True)
    idx3 = learner.select_seeds(test_emb, 'Zaleplon_MPO', k=10, stochastic=True)

    print(f"\n  Sitagliptin seeds (run 1): {idx1.tolist()}")
    print(f"  Sitagliptin seeds (run 2): {idx2.tolist()}")
    print(f"  Different for same oracle: {not torch.equal(idx1, idx2)}")
    print(f"  Different for diff oracle: {not torch.equal(idx1, idx3)}")
