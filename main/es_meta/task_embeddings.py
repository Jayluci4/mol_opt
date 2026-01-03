"""
Task embeddings for PMO oracles.

Categorizes the 23 PMO oracles into 8 task types and provides
one-hot encoding for meta-learning.
"""

import torch
import numpy as np


# PMO oracle categorization based on task type
TASK_CATEGORIES = {
    'similarity': [
        'Albuterol_Similarity',
        'Mestranol_Similarity',
    ],
    'rediscovery': [
        'Celecoxib_Rediscovery',
        'Thiothixene_Rediscovery',
        'Troglitazone_Rediscovery',
    ],
    'mpo': [
        'Amlodipine_MPO',
        'Fexofenadine_MPO',
        'Osimertinib_MPO',
        'Perindopril_MPO',
        'Ranolazine_MPO',
        'Sitagliptin_MPO',
        'Zaleplon_MPO',
    ],
    'bioactivity': [
        'DRD2',
        'GSK3B',
        'JNK3',
    ],
    'property': [
        'QED',
    ],
    'hop': [
        'Deco_Hop',
        'Scaffold_Hop',
    ],
    'isomers': [
        'Isomers_C7H8N2O2',
        'Isomers_C7H8N2O3',
        'Isomers_C9H10N2O2PF2Cl',
    ],
    'median': [
        'Median 1',
        'Median 2',
    ],
    'smarts': [
        'Valsartan_Smarts',
    ],
}

# Create reverse mapping: oracle_name -> category
ORACLE_TO_CATEGORY = {}
for category, oracles in TASK_CATEGORIES.items():
    for oracle in oracles:
        ORACLE_TO_CATEGORY[oracle] = category

# Create category index mapping
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(TASK_CATEGORIES.keys())}
NUM_CATEGORIES = len(TASK_CATEGORIES)


def get_oracle_category(oracle_name: str) -> str:
    """
    Get the category for an oracle.

    Args:
        oracle_name: Name of the PMO oracle

    Returns:
        Category string or 'unknown' if not found
    """
    return ORACLE_TO_CATEGORY.get(oracle_name, 'unknown')


def get_task_embedding(oracle_name: str, dim: int = None) -> torch.Tensor:
    """
    Get one-hot encoding of task category.

    Args:
        oracle_name: Name of the PMO oracle
        dim: Embedding dimension (defaults to NUM_CATEGORIES)

    Returns:
        One-hot tensor of shape (dim,)
    """
    if dim is None:
        dim = NUM_CATEGORIES

    category = get_oracle_category(oracle_name)
    idx = CATEGORY_TO_IDX.get(category, -1)

    emb = torch.zeros(dim)
    if 0 <= idx < dim:
        emb[idx] = 1.0

    return emb


def get_task_embedding_batch(oracle_names: list, dim: int = None) -> torch.Tensor:
    """
    Get batch of task embeddings.

    Args:
        oracle_names: List of oracle names
        dim: Embedding dimension

    Returns:
        Tensor of shape (len(oracle_names), dim)
    """
    embeddings = [get_task_embedding(name, dim) for name in oracle_names]
    return torch.stack(embeddings)


# Training/test split for meta-learning
TRAINING_ORACLES = [
    'QED',
    'GSK3B',
    'JNK3',
    'DRD2',
    'Albuterol_Similarity',
    'Amlodipine_MPO',
    'Celecoxib_Rediscovery',
    'Isomers_C7H8N2O2',
    'Isomers_C7H8N2O3',
    'Median 1',
    'Median 2',
    'Mestranol_Similarity',
    'Perindopril_MPO',
    'Ranolazine_MPO',
    'Scaffold_Hop',
]

TEST_ORACLES = [
    'Deco_Hop',
    'Fexofenadine_MPO',
    'Isomers_C9H10N2O2PF2Cl',
    'Osimertinib_MPO',
    'Sitagliptin_MPO',
    'Thiothixene_Rediscovery',
    'Troglitazone_Rediscovery',
    'Zaleplon_MPO',
    'Valsartan_Smarts',
]


if __name__ == '__main__':
    # Test the embeddings
    print(f"Number of categories: {NUM_CATEGORIES}")
    print(f"Categories: {list(TASK_CATEGORIES.keys())}")
    print()

    for oracle in ['QED', 'GSK3B', 'Sitagliptin_MPO', 'Unknown_Oracle']:
        emb = get_task_embedding(oracle)
        cat = get_oracle_category(oracle)
        print(f"{oracle}: category={cat}, embedding={emb.tolist()}")
