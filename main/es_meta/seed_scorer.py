"""
Seed Scorer Network for meta-learned seed selection.

Predicts expected fitness for (molecule, task) pairs to select
optimal starting populations for molecular optimization.
"""

import torch
import torch.nn as nn


class SeedScorer(nn.Module):
    """
    Neural network that predicts expected fitness for molecules on a given task.

    Takes molecular embeddings and task embeddings as input,
    outputs predicted fitness score in [0, 1].
    """

    def __init__(self, mol_dim: int = 1024, task_dim: int = 9,
                 hidden_dim: int = 128, dropout: float = 0.1):
        """
        Initialize the seed scorer network.

        Args:
            mol_dim: Dimension of molecular embedding (default 1024 for Morgan FP)
            task_dim: Dimension of task embedding (default 9 for PMO categories)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.mol_dim = mol_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim

        # Main network
        self.net = nn.Sequential(
            nn.Linear(mol_dim + task_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, mol_emb: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict fitness scores for molecules on a task.

        Args:
            mol_emb: Molecular embeddings of shape (N, mol_dim)
            task_emb: Task embedding of shape (task_dim,) or (N, task_dim)

        Returns:
            Predicted fitness scores of shape (N,)
        """
        # Handle task embedding broadcasting
        if task_emb.dim() == 1:
            task_emb = task_emb.unsqueeze(0).expand(mol_emb.size(0), -1)

        # Concatenate inputs
        x = torch.cat([mol_emb, task_emb], dim=-1)

        # Forward pass
        return self.net(x).squeeze(-1)

    def score_molecules(self, mol_embeddings: torch.Tensor,
                        task_embedding: torch.Tensor) -> torch.Tensor:
        """
        Score all molecules for a given task.

        Args:
            mol_embeddings: All molecular embeddings (N, mol_dim)
            task_embedding: Task embedding (task_dim,)

        Returns:
            Scores for all molecules (N,)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(mol_embeddings, task_embedding)

    def get_top_k_indices(self, mol_embeddings: torch.Tensor,
                          task_embedding: torch.Tensor, k: int) -> torch.Tensor:
        """
        Get indices of top-k scoring molecules for a task.

        Args:
            mol_embeddings: All molecular embeddings
            task_embedding: Task embedding
            k: Number of top molecules to select

        Returns:
            Indices of top-k molecules
        """
        scores = self.score_molecules(mol_embeddings, task_embedding)
        return torch.topk(scores, k).indices

    def get_flat_params(self) -> torch.Tensor:
        """Get all parameters as a flat tensor."""
        return torch.cat([p.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params: torch.Tensor):
        """Set parameters from a flat tensor."""
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[idx:idx + numel].view(p.shape))
            idx += numel

    def num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class SeedScorerWithContext(SeedScorer):
    """
    Extended seed scorer that also considers molecular context features.

    Adds additional features like molecular weight, number of rings, etc.
    """

    def __init__(self, mol_dim: int = 1024, task_dim: int = 9,
                 context_dim: int = 10, hidden_dim: int = 128, dropout: float = 0.1):
        # Don't call parent __init__ to avoid double initialization
        nn.Module.__init__(self)

        self.mol_dim = mol_dim
        self.task_dim = task_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        total_input_dim = mol_dim + task_dim + context_dim

        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def forward(self, mol_emb: torch.Tensor, task_emb: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional context features.

        Args:
            mol_emb: Molecular embeddings (N, mol_dim)
            task_emb: Task embedding (task_dim,) or (N, task_dim)
            context: Optional context features (N, context_dim)

        Returns:
            Predicted scores (N,)
        """
        if task_emb.dim() == 1:
            task_emb = task_emb.unsqueeze(0).expand(mol_emb.size(0), -1)

        if context is None:
            context = torch.zeros(mol_emb.size(0), self.context_dim,
                                  device=mol_emb.device)

        x = torch.cat([mol_emb, task_emb, context], dim=-1)
        return self.net(x).squeeze(-1)


if __name__ == '__main__':
    # Test the scorer
    print("Testing SeedScorer:")

    scorer = SeedScorer(mol_dim=1024, task_dim=9)
    print(f"  Number of parameters: {scorer.num_params()}")

    # Test forward pass
    batch_size = 100
    mol_emb = torch.randn(batch_size, 1024)
    task_emb = torch.zeros(9)
    task_emb[0] = 1.0  # One-hot

    scores = scorer(mol_emb, task_emb)
    print(f"  Input shapes: mol={mol_emb.shape}, task={task_emb.shape}")
    print(f"  Output shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test top-k selection
    top_indices = scorer.get_top_k_indices(mol_emb, task_emb, k=10)
    print(f"  Top-10 indices: {top_indices.tolist()}")

    # Test flat params
    flat = scorer.get_flat_params()
    print(f"  Flat params shape: {flat.shape}")
