import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, h_sizes, out_size):
        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList()
        for i in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[i], h_sizes[i + 1], bias=False))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        # Feedforward
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            # print(x.shape)
            x = torch.relu(x)
            # print(x.shape)
            #x = torch.relu((self.hidden[i](x))
        output = self.out(x)
        # print(output.shape)
        return output

# class SimpleNet_4bn(nn.Module):
#     def __init__(self, number_of_features, hl1, hl2, hl3, hl4):
#         super(SimpleNet_4bn, self).__init__()
#
#         self.fc1 = nn.Linear(number_of_features, hl1, bias=False)
#         self.bn1 = nn.BatchNorm1d(hl1)
#         self.fc2 = nn.Linear(hl1, hl2, bias=False)
#         self.bn2 = nn.BatchNorm1d(hl2)
#         self.fc3 = nn.Linear(hl2, hl3, bias=False)
#         self.bn3 = nn.BatchNorm1d(hl3)
#         self.fc4 = nn.Linear(hl3, hl4, bias=False)
#         self.bn4 = nn.BatchNorm1d(hl4)
#         self.fc5 = nn.Linear(hl4, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.bn1(self.fc1(x)))
#         x = torch.relu(self.bn2(self.fc2(x)))
#         x = torch.relu(self.bn3(self.fc3(x)))
#         x = torch.relu(self.bn4(self.fc4(x)))
#         x = self.fc5(x)
#         return x
