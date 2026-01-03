"""
Oracle-specific task embeddings using target molecule fingerprints.

Instead of one-hot category encoding, we use:
1. Target molecule fingerprint (for similarity/rediscovery tasks)
2. Learned embedding (for others)
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# Oracle target molecules (from PMO/TDC)
ORACLE_TARGETS = {
    # Similarity tasks - target SMILES
    'Albuterol_Similarity': 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',
    'Mestranol_Similarity': 'C#CC1(O)CCC2C3CCc4cc(OC)ccc4C3CCC21C',

    # Rediscovery tasks - target SMILES
    'Celecoxib_Rediscovery': 'Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1',
    'Thiothixene_Rediscovery': 'CN(C)S(=O)(=O)c1ccc2c(c1)N(CCCN1CCN(C)CC1)c1ccccc1S2',
    'Troglitazone_Rediscovery': 'Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2',

    # MPO tasks - target SMILES (from PMO paper)
    'Sitagliptin_MPO': 'Nc1cc(F)c(F)cc1CC(=O)N1CCn2c(nnc2C(F)(F)F)C1',
    'Zaleplon_MPO': 'CCN(C(C)=O)c1cccc(-c2ccnc3c(C#N)cnn23)c1',
    'Fexofenadine_MPO': 'CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1',
    'Osimertinib_MPO': 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1',
    'Amlodipine_MPO': 'CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl',
    'Ranolazine_MPO': 'COc1ccccc1OCC(O)CN1CCN(CC(=O)Nc2c(C)cccc2C)CC1',
    'Perindopril_MPO': 'CCCC(NC(C)C(=O)N1C2CCCCC2CC1C(=O)O)C(=O)OCC',

    # Hop tasks
    'Deco_Hop': 'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)C',  # Decorated scaffold
    'Scaffold_Hop': 'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)C',  # Same base

    # Isomers - formula, not structure (use learned embedding)
    'Isomers_C7H8N2O2': None,
    'Isomers_C7H8N2O3': None,
    'Isomers_C9H10N2O2PF2Cl': None,

    # Median - no target
    'Median 1': None,
    'Median 2': None,

    # SMARTS
    'Valsartan_Smarts': 'CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1)C(C(=O)O)C(C)C',

    # Bioactivity - no target molecule
    'DRD2': None,
    'GSK3B': None,
    'JNK3': None,

    # Property
    'QED': None,
}


def fp_from_smiles(smiles: str, radius: int = 2, n_bits: int = 512) -> np.ndarray:
    """Compute Morgan fingerprint from SMILES."""
    if smiles is None:
        return np.zeros(n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    arr = np.zeros(n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# Precompute target fingerprints
TARGET_FPS = {}
for oracle, smiles in ORACLE_TARGETS.items():
    TARGET_FPS[oracle] = fp_from_smiles(smiles, n_bits=512)


# Category encoding (fallback for oracles without targets)
CATEGORIES = ['similarity', 'rediscovery', 'mpo', 'bioactivity', 'property', 'hop', 'isomers', 'median', 'smarts']
ORACLE_TO_CATEGORY = {
    'Albuterol_Similarity': 'similarity', 'Mestranol_Similarity': 'similarity',
    'Celecoxib_Rediscovery': 'rediscovery', 'Thiothixene_Rediscovery': 'rediscovery', 'Troglitazone_Rediscovery': 'rediscovery',
    'Sitagliptin_MPO': 'mpo', 'Zaleplon_MPO': 'mpo', 'Fexofenadine_MPO': 'mpo', 'Osimertinib_MPO': 'mpo',
    'Amlodipine_MPO': 'mpo', 'Ranolazine_MPO': 'mpo', 'Perindopril_MPO': 'mpo',
    'DRD2': 'bioactivity', 'GSK3B': 'bioactivity', 'JNK3': 'bioactivity',
    'QED': 'property',
    'Deco_Hop': 'hop', 'Scaffold_Hop': 'hop',
    'Isomers_C7H8N2O2': 'isomers', 'Isomers_C7H8N2O3': 'isomers', 'Isomers_C9H10N2O2PF2Cl': 'isomers',
    'Median 1': 'median', 'Median 2': 'median',
    'Valsartan_Smarts': 'smarts',
}


def get_task_embedding_v2(oracle_name: str) -> torch.Tensor:
    """
    Get oracle-specific task embedding.

    Combines:
    - Target molecule fingerprint (512 bits) if available
    - Category one-hot (9 bits)

    Total: 521 dimensions
    """
    # Target fingerprint
    oracle_key = oracle_name
    # Handle case variations
    for key in TARGET_FPS:
        if key.lower() == oracle_name.lower():
            oracle_key = key
            break

    target_fp = TARGET_FPS.get(oracle_key, np.zeros(512))

    # Category one-hot
    category = ORACLE_TO_CATEGORY.get(oracle_key, 'property')
    cat_idx = CATEGORIES.index(category) if category in CATEGORIES else 0
    cat_onehot = np.zeros(len(CATEGORIES))
    cat_onehot[cat_idx] = 1.0

    # Combine
    embedding = np.concatenate([target_fp, cat_onehot])
    return torch.tensor(embedding, dtype=torch.float32)


def get_embedding_dim() -> int:
    """Return embedding dimension."""
    return 512 + len(CATEGORIES)  # 521


if __name__ == '__main__':
    print(f"Embedding dimension: {get_embedding_dim()}")
    print("\nTest embeddings:")

    for oracle in ['QED', 'Sitagliptin_MPO', 'Zaleplon_MPO', 'Fexofenadine_MPO', 'Osimertinib_MPO']:
        emb = get_task_embedding_v2(oracle)
        target_bits = emb[:512].sum().item()
        cat = emb[512:].argmax().item()
        print(f"  {oracle}: target_bits={target_bits:.0f}, category={CATEGORIES[cat]}")
