import torch.nn as nn
import torch
# from tensorflow.python.ops.ragged.ragged_math_ops import softmax
# from torch.nn import softmax


from permutate_pattern_frequency import get_invariant_permutation,get_topology_invariant_permutation,get_branch_length_invariant_permutation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# residual block
class ResBlock(torch.nn.Module):
    def __init__(self, in_feature= 625, hidden_feature = 256,dropout_rate= 0.2):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_feature, hidden_feature,bias = False),
            # torch.nn.LayerNorm(hidden_feature),
            torch.nn.ReLU(),
            nn.Dropout(dropout_rate),

            torch.nn.Linear(hidden_feature, hidden_feature, bias=False),
            # torch.nn.LayerNorm(hidden_feature),
            torch.nn.ReLU(),
            nn.Dropout(dropout_rate),

            torch.nn.Linear(hidden_feature, in_feature, bias=False),
            # torch.nn.LayerNorm(in_feature),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = x.to(device,dtype=torch.float32)
        output = self.layers(x).to(device,dtype=torch.float32)
        # print(x.dtype)
        # print(output.dtype)
        return x+output
        # return x.to(torch.float32) + self.layers(x).to(torch.float32)



# bls predictor
class Quartet_Net_bls(nn.Module):

    def __init__(self,in_feature=625,res_hidden_feature=256,clf1_hidden_1 = 256,clf1_hidden_2 = 128, clf1_hidden_3 = 16, clf1_out = 1,clf2_hidden_1 = 256,clf2_hidden_2 = 128, clf2_hidden_3 = 16, clf2_out = 1,dropout_rate = 0.2):
        super(Quartet_Net_bls, self).__init__()
        # activation
        self.ResBlock = ResBlock(in_feature,res_hidden_feature,dropout_rate)

        # for external branches
        self.predictor_1 = torch.nn.Sequential(
            torch.nn.Linear(in_feature, clf1_hidden_1, bias=False),
            # torch.nn.LayerNorm(clf_hidden_1),
            torch.nn.ReLU(),
            nn.Dropout(dropout_rate),

            torch.nn.Linear(clf1_hidden_1, clf1_hidden_2, bias=False),
            # torch.nn.LayerNorm(clf_hidden_2),
            torch.nn.ReLU(),
            nn.Dropout(dropout_rate),

            torch.nn.Linear(clf1_hidden_2, clf1_hidden_3, bias=False),
            # torch.nn.LayerNorm(clf_hidden_3),
            torch.nn.ReLU(),

            torch.nn.Linear(clf1_hidden_3, clf1_out, bias=False),
            torch.nn.ReLU()
        )
        # for internal branch
        self.predictor_2 = torch.nn.Sequential(
            torch.nn.Linear(in_feature, clf2_hidden_1, bias=False),
            # torch.nn.LayerNorm(clf_hidden_1),
            torch.nn.ReLU(),
            nn.Dropout(dropout_rate),

            torch.nn.Linear(clf2_hidden_1, clf2_hidden_2, bias=False),
            # torch.nn.LayerNorm(clf_hidden_2),
            torch.nn.ReLU(),
            nn.Dropout(dropout_rate),

            torch.nn.Linear(clf2_hidden_2, clf2_hidden_3, bias=False),
            # torch.nn.LayerNorm(clf_hidden_3),
            torch.nn.ReLU(),

            torch.nn.Linear(clf2_hidden_3, clf2_out, bias=False),
            torch.nn.ReLU()
        )


    def forward(self, data):
        # # data = data.view(-1,625)
        data = data.view(-1,625)
        data = data.cpu().numpy()
        # get all permutations
        PA = torch.tensor(get_branch_length_invariant_permutation(data,branch = 1))
        PB = torch.tensor(get_branch_length_invariant_permutation(data,branch =2))
        PC = torch.tensor(get_branch_length_invariant_permutation(data, branch =3))
        PD = torch.tensor(get_branch_length_invariant_permutation(data, branch =4))
        PA = PA.to(device,dtype=torch.float32)
        PB = PB.to(device,dtype=torch.float32)
        PC = PC.to(device,dtype=torch.float32)
        PD = PD.to(device,dtype=torch.float32)

        # extract features
        PA = self.ResBlock(PA)
        PB = self.ResBlock(PB)
        PC = self.ResBlock(PC)
        PD = self.ResBlock(PD)

        PA = torch.mean(PA, dim=1, keepdim=True)
        PB = torch.mean(PB, dim=1, keepdim=True)
        PC = torch.mean(PC, dim=1, keepdim=True)
        PD = torch.mean(PD, dim=1, keepdim=True)

        P0 = torch.cat((PA, PB,PC,PD), dim=1)
        P_int = torch.mean(P0, dim=1, keepdim=True)


        # m x 4 matrix
        ext_b = self.predictor_1(P0)
        ext_b = ext_b.view(-1,4)

        # m x 1 matrix
        int_b = self.predictor_2(P_int)
        int_b = int_b.view(-1,1)
        result = torch.cat((ext_b, int_b), dim=1)
        result = result.view(-1, 5)

        return result


# topology classifier
class Quartet_Net_top(nn.Module):

    def __init__(self,in_feature=625,res_hidden_feature=256,clf_hidden_1 = 256,clf_hidden_2 = 64, clf_hidden_3 = 16, clf_out = 1,dropout_rate = 0.2):
        super(Quartet_Net_top, self).__init__()
        # activation
        self.ResBlock = ResBlock(in_feature,res_hidden_feature,dropout_rate)


        self.softmax = nn.Softmax(dim=0)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_feature, clf_hidden_1, bias=False),
            # torch.nn.LayerNorm(clf_hidden_1),
            torch.nn.ReLU(),
            nn.Dropout(dropout_rate),

            torch.nn.Linear(clf_hidden_1, clf_hidden_2, bias=False),
            # torch.nn.LayerNorm(clf_hidden_2),
            torch.nn.ReLU(),
            nn.Dropout(dropout_rate),

            torch.nn.Linear(clf_hidden_2, clf_hidden_3, bias=False),
            # torch.nn.LayerNorm(clf_hidden_3),
            torch.nn.ReLU(),

            torch.nn.Linear(clf_hidden_3, clf_out, bias=False),
        )


    def forward(self, data):
        data = data.view(-1,625)
        # data = data.view(-1,624)
        data = data.cpu().numpy()
        # get all permutations
        P0 = torch.tensor(get_invariant_permutation(data))
        P1 = torch.tensor(get_topology_invariant_permutation(data,1))
        P2 = torch.tensor(get_topology_invariant_permutation(data, 2))
        P3 = torch.tensor(get_topology_invariant_permutation(data, 3))
        P0 = P0.to(device,dtype=torch.float32)
        P1 = P1.to(device,dtype=torch.float32)
        P2 = P2.to(device,dtype=torch.float32)
        P3 = P3.to(device,dtype=torch.float32)

        # extract features
        P0 = self.ResBlock(P0)
        P1 = self.ResBlock(P1)
        P2 = self.ResBlock(P2)
        P3 = self.ResBlock(P3)


        # sum all features based on the tree topology
        T1 = torch.cat((P0, P1), dim=1)
        T2 = torch.cat((P0, P2), dim=1)
        T3 = torch.cat((P0, P3), dim=1)

        # compute the mean of all permutations
        T1 = torch.mean(T1, dim=1, keepdim=True)
        T2 = torch.mean(T2, dim=1, keepdim=True)
        T3 = torch.mean(T3, dim=1, keepdim=True)

        result = torch.cat((T1, T2,T3), dim=1)

        # result is a 3 x 3 matrix
        result = self.classifier(result)
        # print(result)
        # print(result)
        # row_max_values, row_indices = result.max(dim=1)

        # find the maximum value
        # max_value = row_max_values.max()
        # max_row_index = row_max_values.argmax()
        # result = result[max_row_index.item(),:]
        # row_max_values, _ = result.max(dim=2)
        # max_row_index = row_max_values.argmax(dim=1)
        #
        # max_row_index = max_row_index.view(-1, 1)
        #
        # index_expanded = max_row_index.unsqueeze(2).expand(-1, -1, 3)
        # result = torch.gather(result, 1, index_expanded).squeeze(1)


        result = result.view(-1,3)
        # result = self.softmax(result)
        # if not self.training:
        #     print(result)
        # print(result)
        return result