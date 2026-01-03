"""
Improved Seed Scorer with oracle-specific embeddings.

Key changes:
1. Larger task embedding (521 dim with target FP)
2. Deeper network
3. Stochastic sampling option
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeedScorerV2(nn.Module):
    """
    Improved seed scorer with oracle-specific task embeddings.
    """

    def __init__(self, mol_dim: int = 1024, task_dim: int = 521,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.mol_dim = mol_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim

        # Separate encoders for molecule and task
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined scorer
        combined_dim = hidden_dim + hidden_dim // 2
        self.scorer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, mol_emb: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Score molecules for a task.

        Args:
            mol_emb: (N, mol_dim)
            task_emb: (task_dim,) or (N, task_dim)

        Returns:
            Scores (N,) - unbounded (use sigmoid for [0,1])
        """
        # Encode molecules
        mol_h = self.mol_encoder(mol_emb)  # (N, hidden)

        # Encode task
        if task_emb.dim() == 1:
            task_emb = task_emb.unsqueeze(0).expand(mol_emb.size(0), -1)
        task_h = self.task_encoder(task_emb)  # (N, hidden//2)

        # Combine and score
        combined = torch.cat([mol_h, task_h], dim=-1)
        scores = self.scorer(combined).squeeze(-1)

        return scores

    def score_with_temperature(self, mol_emb: torch.Tensor, task_emb: torch.Tensor,
                                temperature: float = 1.0) -> torch.Tensor:
        """Score with temperature for softer/harder selection."""
        scores = self.forward(mol_emb, task_emb)
        return scores / temperature

    def sample_top_k(self, mol_emb: torch.Tensor, task_emb: torch.Tensor,
                     k: int, temperature: float = 0.1, stochastic: bool = True) -> torch.Tensor:
        """
        Sample k molecules using temperature-scaled softmax.

        Args:
            mol_emb: All molecule embeddings
            task_emb: Task embedding
            k: Number to select
            temperature: Lower = more deterministic
            stochastic: If True, sample; if False, take top-k

        Returns:
            Indices of selected molecules
        """
        scores = self.forward(mol_emb, task_emb)

        if not stochastic:
            return torch.topk(scores, k).indices

        # Temperature-scaled softmax sampling
        probs = F.softmax(scores / temperature, dim=0)

        # Sample without replacement
        indices = torch.multinomial(probs, k, replacement=False)
        return indices

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/jayantlohia16/mol_opt')

    from main.es_meta.task_embeddings_v2 import get_task_embedding_v2, get_embedding_dim

    print("Testing SeedScorerV2:")
    scorer = SeedScorerV2(mol_dim=1024, task_dim=get_embedding_dim())
    print(f"  Parameters: {scorer.num_params()}")

    # Test forward
    mol_emb = torch.randn(100, 1024)
    task_emb = get_task_embedding_v2('Sitagliptin_MPO')

    scores = scorer(mol_emb, task_emb)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Scores range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test stochastic sampling
    idx1 = scorer.sample_top_k(mol_emb, task_emb, k=10, stochastic=True)
    idx2 = scorer.sample_top_k(mol_emb, task_emb, k=10, stochastic=True)
    print(f"  Stochastic sample 1: {idx1.tolist()}")
    print(f"  Stochastic sample 2: {idx2.tolist()}")
    print(f"  Different samples: {not torch.equal(idx1, idx2)}")
