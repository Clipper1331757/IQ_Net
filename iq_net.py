import torch
import torch.nn as nn

from permutations import (
    get_invariant_permutation_torch,
    get_topology_invariant_permutation_torch,
    get_branch_length_invariant_permutation_torch
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_feature=625, hidden_feature=256, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feature, hidden_feature, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_feature, hidden_feature, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_feature, in_feature, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.layers(x)

class IQ_Net_top(nn.Module):
    def __init__(self,
                 in_feature=625,
                 res_hidden_feature=256,
                 clf_hidden_1=256,
                 clf_hidden_2=64,
                 clf_hidden_3=16,
                 clf_out=1,
                 dropout_rate=0.2):
        super(IQ_Net_top, self).__init__()

        self.ResBlock = ResBlock(in_feature, res_hidden_feature, dropout_rate)

        # single head for different topology
        self.head1 = nn.Sequential(
            nn.Linear(in_feature, clf_hidden_1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(clf_hidden_1, clf_hidden_2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(clf_hidden_2, clf_hidden_3, bias=False),
            nn.ReLU(),

            nn.Linear(clf_hidden_3, clf_out, bias=False)
        )

        # self.head2 = nn.Sequential(
        #     nn.Linear(in_feature, clf_hidden_1, bias=False),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #
        #     nn.Linear(clf_hidden_1, clf_hidden_2, bias=False),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #
        #     nn.Linear(clf_hidden_2, clf_hidden_3, bias=False),
        #     nn.ReLU(),
        #
        #     nn.Linear(clf_hidden_3, clf_out, bias=False)
        # )
        #
        # self.head3 = nn.Sequential(
        #     nn.Linear(in_feature, clf_hidden_1, bias=False),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #
        #     nn.Linear(clf_hidden_1, clf_hidden_2, bias=False),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #
        #     nn.Linear(clf_hidden_2, clf_hidden_3, bias=False),
        #     nn.ReLU(),
        #
        #     nn.Linear(clf_hidden_3, clf_out, bias=False)
        # )

    def forward(self, data):
        device = data.device

        P0 = get_invariant_permutation_torch(data)
        P1 = get_topology_invariant_permutation_torch(data, 1)
        P2 = get_topology_invariant_permutation_torch(data, 2)
        P3 = get_topology_invariant_permutation_torch(data, 3)

        # T1 = torch.cat([P0, P1], dim=1).view(-1, 2, 625).mean(dim=1)
        # print(T1)

        def process(P):
            P = P.view(-1, 625)           # [batch*4, 625]
            P = self.ResBlock(P)
            P = P.view(-1, 4, 625)        # [batch, 4, 625]
            return torch.mean(P, dim=1)   # [batch, 625]

        P0 = process(P0)
        P1 = process(P1)
        P2 = process(P2)
        P3 = process(P3)

        T1 = torch.cat([P0, P1], dim=1).view(-1, 2, 625).mean(dim=1)
        # print(T1)
        T2 = torch.cat([P0, P2], dim=1).view(-1, 2, 625).mean(dim=1)
        T3 = torch.cat([P0, P3], dim=1).view(-1, 2, 625).mean(dim=1)

        score1 = self.head1(T1)
        score2 = self.head1(T2)
        score3 = self.head1(T3)

        result = torch.cat([score1, score2, score3], dim=1)
        # print(result[:5])
        return result


class IQ_Net_bls(nn.Module):
    def __init__(self, in_feature=625, res_hidden_feature=256,
                 clf1_hidden_1=256, clf1_hidden_2=128, clf1_hidden_3=16, clf1_out=1,
                 clf2_hidden_1=256, clf2_hidden_2=128, clf2_hidden_3=16, clf2_out=1,
                 dropout_rate=0.2):
        super(IQ_Net_bls, self).__init__()

        self.ResBlock = ResBlock(in_feature, res_hidden_feature, dropout_rate)

        # predictor 1 for external branches
        self.predictor_1 = nn.Sequential(
            nn.Linear(in_feature, clf1_hidden_1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(clf1_hidden_1, clf1_hidden_2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(clf1_hidden_2, clf1_hidden_3, bias=False),
            nn.ReLU(),

            nn.Linear(clf1_hidden_3, clf1_out, bias=False),
            nn.ReLU()
        )

        # predictor 2 for internal branch
        self.predictor_2 = nn.Sequential(
            nn.Linear(in_feature, clf2_hidden_1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(clf2_hidden_1, clf2_hidden_2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(clf2_hidden_2, clf2_hidden_3, bias=False),
            nn.ReLU(),

            nn.Linear(clf2_hidden_3, clf2_out, bias=False),
            nn.ReLU()
        )

    def forward(self, data):
        data = data.view(-1, 625)

        PA = get_branch_length_invariant_permutation_torch(data, branch=1)
        PB = get_branch_length_invariant_permutation_torch(data, branch=2)
        PC = get_branch_length_invariant_permutation_torch(data, branch=3)
        PD = get_branch_length_invariant_permutation_torch(data, branch=4)

        PA = self.ResBlock(PA)
        PB = self.ResBlock(PB)
        PC = self.ResBlock(PC)
        PD = self.ResBlock(PD)

        PA = torch.mean(PA, dim=1)  # [batch, in_feature]
        PB = torch.mean(PB, dim=1)
        PC = torch.mean(PC, dim=1)
        PD = torch.mean(PD, dim=1)

        ext_scores = []
        for P in [PA, PB, PC, PD]:
            score = self.predictor_1(P)  # [batch, 1]
            ext_scores.append(score)
        ext_b = torch.cat(ext_scores, dim=1)  # [batch, 4]

        P0 = torch.stack([PA, PB, PC, PD], dim=1)  # [batch, 4, in_feature]
        P_int = torch.mean(P0, dim=1)  # [batch, in_feature]
        int_b = self.predictor_2(P_int)  # [batch, 1]

        result = torch.cat([ext_b, int_b], dim=1)  # [batch, 5]
        return result