import numpy as np
import itertools


# P4 = list(itertools.permutations([0,1,2,3]))
n_map = {'A':0,'C':1,'G':2,'T':3,'-':4,'N':4}
nu_weights =np.array([125,25,5,1])[:,np.newaxis]
feature_names = {''.join(p):( np.array([125,25,5,1]) * np.array([n_map[e] for e in p])).sum() for p in itertools.product('ACGT-',repeat=4)}

# feature_names.pop('----')
num_feature = len(feature_names)
# print(feature_names)
# print(P4)
# permutation 0 is invariant for all tree topology
P0 = [(0,1,2,3),(1,0,3,2),(3,2,1,0),(2,3,0,1)]
# permutation 1,2,3 are invariant for topology 'AB' , 'AC', 'AD', respectively
P1 = [(0,1,3,2),(1,0,2,3),(2,3,1,0),(3,2,0,1)]
P2 = [(0,3,2,1),(1,2,3,0),(2,1,0,3),(3,0,1,2)]
P3 = [(0,2,1,3),(1,3,0,2),(2,0,3,1),(3,1,2,0)]

PA = [(0,1,2,3),(0,1,3,2),(0,2,1,3),(0,2,3,1),(0,3,2,1),(0,3,1,2)]
PB = [(1,0,2,3),(1,0,3,2),(1,2,0,3),(1,2,3,0),(1,3,2,0),(1,3,0,2)]
PC = [(2,1,0,3),(2,1,3,0),(2,0,1,3),(2,0,3,1),(2,3,0,1),(2,3,1,0)]
PD = [(3,1,2,0),(3,1,0,2),(3,2,1,0),(3,2,0,1),(3,0,2,1),(3,0,1,2)]

invariant_permutations = np.zeros((4,num_feature),dtype=int)

top1_permutations = np.zeros((4,num_feature),dtype=int)
top2_permutations = np.zeros((4,num_feature),dtype=int)
top3_permutations = np.zeros((4,num_feature),dtype=int)

for i, permu in enumerate(P0):
    permu_indices = np.zeros(num_feature)
    for s_pattern in feature_names.keys():
        new_s_pattern = ''.join([s_pattern[i] for i in permu])
        permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
    invariant_permutations[i] = permu_indices

for i, permu in enumerate(P1):
    permu_indices = np.zeros(num_feature)
    for s_pattern in feature_names.keys():
        new_s_pattern = ''.join([s_pattern[i] for i in permu])
        permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
    top1_permutations[i] = permu_indices

for i, permu in enumerate(P2):
    permu_indices = np.zeros(num_feature)
    for s_pattern in feature_names.keys():
        new_s_pattern = ''.join([s_pattern[i] for i in permu])
        permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
    top2_permutations[i] = permu_indices

for i, permu in enumerate(P3):
    permu_indices = np.zeros(num_feature)
    for s_pattern in feature_names.keys():
        new_s_pattern = ''.join([s_pattern[i] for i in permu])
        permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
    top3_permutations[i] = permu_indices

top_invariant_permutations  = [top1_permutations,top2_permutations,top3_permutations]


# get branch length invariant permutations
bla_permutations = np.zeros((6,num_feature),dtype=int)
blb_permutations = np.zeros((6,num_feature),dtype=int)
blc_permutations = np.zeros((6,num_feature),dtype=int)
bld_permutations = np.zeros((6,num_feature),dtype=int)

for i, permu in enumerate(PA):
    permu_indices = np.zeros(num_feature)
    for s_pattern in feature_names.keys():
        new_s_pattern = ''.join([s_pattern[i] for i in permu])
        permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
    bla_permutations[i] = permu_indices

for i, permu in enumerate(PB):
    permu_indices = np.zeros(num_feature)
    for s_pattern in feature_names.keys():
        new_s_pattern = ''.join([s_pattern[i] for i in permu])
        permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
    blb_permutations[i] = permu_indices

for i, permu in enumerate(PC):
    permu_indices = np.zeros(num_feature)
    for s_pattern in feature_names.keys():
        new_s_pattern = ''.join([s_pattern[i] for i in permu])
        permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
    blc_permutations[i] = permu_indices

for i, permu in enumerate(PD):
    permu_indices = np.zeros(num_feature)
    for s_pattern in feature_names.keys():
        new_s_pattern = ''.join([s_pattern[i] for i in permu])
        permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
    bld_permutations[i] = permu_indices


bls_invariant_permutations = [bla_permutations,blb_permutations,blc_permutations,bld_permutations]

# get all invariant permutations
def get_invariant_permutation(data):
    m = data.shape[0]

    permutated_data = np.zeros((m, 4, num_feature))
    # permutated_data = np.zeros((4, 625))

    # get all permutations
    for row in range(4):
        permutated_data[:, row] = data[:, invariant_permutations[row]]
        # permutated_data[ row, ] = data[invariant_permutations[row]]

    return permutated_data


# get all topology invariant permutations
def get_topology_invariant_permutation(data, topology):
    m = data.shape[0]

    permutated_data = np.zeros((m, 4, num_feature))
    # permutated_data = np.zeros((4, 625))
    p = top_invariant_permutations[topology - 1]

    for row in range(4):
        permutated_data[:, row] = data[:, p[row]]
        # permutated_data[row] = data[p[row]]
    return permutated_data

def get_branch_length_invariant_permutation(data, branch):
    m = data.shape[0]

    permutated_data = np.zeros((m, 6, num_feature))
    # permutated_data = np.zeros((4, 625))
    p = bls_invariant_permutations[branch - 1]

    for row in range(6):
        permutated_data[:, row] = data[:, p[row]]
        # permutated_data[row] = data[p[row]]
    return permutated_data

