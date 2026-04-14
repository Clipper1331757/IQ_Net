import torch
import numpy as np
import itertools

# === basic setting ===

n_map = {'A':0,'C':1,'G':2,'T':3,'-':4,'N':4}
nu_weights = np.array([125, 25, 5, 1])[:, np.newaxis]

feature_names = {
    ''.join(p): (np.array([125, 25, 5, 1]) * np.array([n_map[e] for e in p])).sum()
    for p in itertools.product('ACGT-', repeat=4)
}
num_feature = len(feature_names)

# === define permutation ===

P0 = [(0,1,2,3),(1,0,3,2),(3,2,1,0),(2,3,0,1)]
P1 = [(0,1,3,2),(1,0,2,3),(2,3,1,0),(3,2,0,1)]
P2 = [(0,3,2,1),(1,2,3,0),(2,1,0,3),(3,0,1,2)]
P3 = [(0,2,1,3),(1,3,0,2),(2,0,3,1),(3,1,2,0)]

PA = [(0,1,2,3),(0,1,3,2),(0,2,1,3),(0,2,3,1),(0,3,2,1),(0,3,1,2)]
PB = [(1,0,2,3),(1,0,3,2),(1,2,0,3),(1,2,3,0),(1,3,2,0),(1,3,0,2)]
PC = [(2,1,0,3),(2,1,3,0),(2,0,1,3),(2,0,3,1),(2,3,0,1),(2,3,1,0)]
PD = [(3,1,2,0),(3,1,0,2),(3,2,1,0),(3,2,0,1),(3,0,2,1),(3,0,1,2)]

# === build permutation index ===

def build_permutation_tensor(P):
    tensor = np.zeros((len(P), num_feature), dtype=int)
    for i, permu in enumerate(P):
        permu_indices = np.zeros(num_feature, dtype=int)
        for s_pattern in feature_names.keys():
            new_s_pattern = ''.join([s_pattern[j] for j in permu])
            permu_indices[feature_names[new_s_pattern]] = feature_names[s_pattern]
        tensor[i] = permu_indices
    return torch.tensor(tensor, dtype=torch.long)

invariant_permutations = build_permutation_tensor(P0)
top1_permutations = build_permutation_tensor(P1)
top2_permutations = build_permutation_tensor(P2)
top3_permutations = build_permutation_tensor(P3)
top_invariant_permutations = [top1_permutations, top2_permutations, top3_permutations]

bla_permutations = build_permutation_tensor(PA)
blb_permutations = build_permutation_tensor(PB)
blc_permutations = build_permutation_tensor(PC)
bld_permutations = build_permutation_tensor(PD)
bls_invariant_permutations = [bla_permutations, blb_permutations, blc_permutations, bld_permutations]

# === Pure PyTorch permutation functions ===

def get_invariant_permutation_torch(data: torch.Tensor) -> torch.Tensor:
    """
    data: [batch_size, num_feature]
    return: [batch_size, 4, num_feature]
    """
    perms = []
    for i in range(invariant_permutations.shape[0]):
        idx = invariant_permutations[i].to(data.device)
        perm = data[:, idx]
        perms.append(perm.unsqueeze(1))
    return torch.cat(perms, dim=1)

def get_topology_invariant_permutation_torch(data: torch.Tensor, topology: int) -> torch.Tensor:
    """
    data: [batch_size, num_feature]
    topology: int in [1,2,3]
    return: [batch_size, 4, num_feature]
    """
    assert 1 <= topology <= 3
    p_tensor = top_invariant_permutations[topology - 1]
    perms = []
    for i in range(p_tensor.shape[0]):
        idx = p_tensor[i].to(data.device)
        perm = data[:, idx]
        perms.append(perm.unsqueeze(1))
    return torch.cat(perms, dim=1)

def get_branch_length_invariant_permutation_torch(data: torch.Tensor, branch: int) -> torch.Tensor:
    """
    data: [batch_size, num_feature]
    branch: int in [1,2,3,4]
    return: [batch_size, 6, num_feature]
    """
    assert 1 <= branch <= 4
    p_tensor = bls_invariant_permutations[branch - 1]
    perms = []
    for i in range(p_tensor.shape[0]):
        idx = p_tensor[i].to(data.device)
        perm = data[:, idx]
        perms.append(perm.unsqueeze(1))
    return torch.cat(perms, dim=1)