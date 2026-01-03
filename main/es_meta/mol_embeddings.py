"""
Molecular embeddings for seed selection.

Uses Morgan fingerprints (1024-bit) for molecular representation.
Supports caching for efficient reuse across optimization runs.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


def fp_embedding(smi: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    Compute Morgan fingerprint embedding for a SMILES string.

    Args:
        smi: SMILES string
        radius: Morgan fingerprint radius (default 2)
        n_bits: Number of bits in fingerprint (default 1024)

    Returns:
        numpy array of shape (n_bits,)
    """
    if smi is None:
        return np.zeros(n_bits)

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_embeddings(smiles_list: list, radius: int = 2, n_bits: int = 1024,
                       show_progress: bool = True) -> torch.Tensor:
    """
    Compute embeddings for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius
        n_bits: Number of bits
        show_progress: Whether to show progress bar

    Returns:
        Tensor of shape (len(smiles_list), n_bits)
    """
    embeddings = []
    iterator = tqdm(smiles_list, desc="Computing embeddings") if show_progress else smiles_list

    for smi in iterator:
        emb = fp_embedding(smi, radius, n_bits)
        embeddings.append(emb)

    return torch.tensor(np.array(embeddings), dtype=torch.float32)


def get_zinc_embeddings(smiles_list: list, cache_path: str = None,
                        radius: int = 2, n_bits: int = 1024) -> torch.Tensor:
    """
    Get embeddings for ZINC molecules, with optional caching.

    Args:
        smiles_list: List of SMILES strings
        cache_path: Path to cache file (optional)
        radius: Morgan fingerprint radius
        n_bits: Number of bits

    Returns:
        Tensor of shape (len(smiles_list), n_bits)
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        data = torch.load(cache_path)
        # Verify cache matches current data
        if data.shape[0] == len(smiles_list):
            return data
        else:
            print(f"Cache size mismatch ({data.shape[0]} vs {len(smiles_list)}), recomputing...")

    # Compute embeddings
    embeddings = compute_embeddings(smiles_list, radius, n_bits)

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"Saved embeddings cache to {cache_path}")

    return embeddings


def get_embedding_for_smiles(smiles_list: list, zinc_smiles: list,
                             zinc_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Get embeddings for specific SMILES, looking up from precomputed ZINC embeddings.

    Args:
        smiles_list: SMILES to look up
        zinc_smiles: Full ZINC SMILES list
        zinc_embeddings: Precomputed ZINC embeddings

    Returns:
        Tensor of embeddings for requested SMILES
    """
    # Create lookup dictionary
    smi_to_idx = {smi: i for i, smi in enumerate(zinc_smiles)}

    embeddings = []
    for smi in smiles_list:
        if smi in smi_to_idx:
            embeddings.append(zinc_embeddings[smi_to_idx[smi]])
        else:
            # Compute on the fly if not in ZINC
            emb = fp_embedding(smi)
            embeddings.append(torch.tensor(emb, dtype=torch.float32))

    return torch.stack(embeddings)


class EmbeddingCache:
    """
    Manages embedding cache for efficient lookup.
    """

    def __init__(self, smiles_list: list, cache_path: str = None,
                 radius: int = 2, n_bits: int = 1024):
        """
        Initialize embedding cache.

        Args:
            smiles_list: List of SMILES to precompute
            cache_path: Path to save/load cache
            radius: Morgan fingerprint radius
            n_bits: Number of bits
        """
        self.smiles_list = smiles_list
        self.smi_to_idx = {smi: i for i, smi in enumerate(smiles_list)}
        self.radius = radius
        self.n_bits = n_bits

        # Load or compute embeddings
        self.embeddings = get_zinc_embeddings(
            smiles_list, cache_path, radius, n_bits
        )

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.embeddings[idx]
        elif isinstance(idx, str):
            return self.get_embedding(idx)
        elif isinstance(idx, list):
            return self.get_embeddings(idx)
        else:
            return self.embeddings[idx]

    def get_embedding(self, smi: str) -> torch.Tensor:
        """Get embedding for a single SMILES."""
        if smi in self.smi_to_idx:
            return self.embeddings[self.smi_to_idx[smi]]
        else:
            emb = fp_embedding(smi, self.radius, self.n_bits)
            return torch.tensor(emb, dtype=torch.float32)

    def get_embeddings(self, smiles_list: list) -> torch.Tensor:
        """Get embeddings for multiple SMILES."""
        return torch.stack([self.get_embedding(smi) for smi in smiles_list])

    def get_all_embeddings(self) -> torch.Tensor:
        """Get all precomputed embeddings."""
        return self.embeddings


if __name__ == '__main__':
    # Test embeddings
    test_smiles = [
        'CCO',  # Ethanol
        'CC(=O)O',  # Acetic acid
        'c1ccccc1',  # Benzene
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
    ]

    print("Testing molecular embeddings:")
    for smi in test_smiles:
        emb = fp_embedding(smi)
        print(f"  {smi}: shape={emb.shape}, sum={emb.sum():.0f}")

    # Test batch computation
    batch_emb = compute_embeddings(test_smiles, show_progress=False)
    print(f"\nBatch embeddings shape: {batch_emb.shape}")
