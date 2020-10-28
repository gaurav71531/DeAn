from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor

import mc_dropout


# class BayesianNet(mc_dropout.BayesianModule):
#     def __init__(self, num_classes):
#         super().__init__(num_classes)

#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
#         self.conv1_drop = mc_dropout.MCDropout2d()
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
#         self.conv2_drop = mc_dropout.MCDropout2d()
#         self.fc1 = nn.Linear(1024, 128)
#         self.fc1_drop = mc_dropout.MCDropout()
#         self.fc2 = nn.Linear(128, num_classes)

#     def mc_forward_impl(self, input: Tensor):
#         input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
#         input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
#         input = input.view(-1, 1024)
#         input = F.relu(self.fc1_drop(self.fc1(input)))
#         input = self.fc2(input)
#         input = F.log_softmax(input, dim=1)

#         return input
class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.convBlock = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 4, 1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			mc_dropout.MCDropout2d(0.25)
        )
        self.Dense = nn.Linear(32 * 11 * 11, 128)
        self.linearBlock = nn.Sequential(
            nn.ReLU(inplace=True),
            mc_dropout.MCDropout(0.5),
            nn.Linear(128, num_classes)
        )

        # self.conv1 = nn.Conv2d(1, 32, 4, 1)
        # self.conv1_drop = mc_dropout.MCDropout2d(p=0.25)
        # self.conv2 = nn.Conv2d(32, 32, 4, 1)
        # self.conv2_drop = mc_dropout.MCDropout2d(0.25)
        # self.fc1 = nn.Linear(32*11*11, 128)
        # self.fc1_drop = mc_dropout.MCDropout()
        # self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: Tensor):
        input = self.convBlock(input)
        input = input.view(-1, 32 * 11 * 11)
        input = self.Dense(input)
        input = self.linearBlock(input)
        # input = F.relu(self.conv1(input))
        # input = self.conv2_drop(F.max_pool2d(F.relu(self.conv2(input)),2))
        # # input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        # input = input.view(-1, 32*11*11)
        # input = self.fc1_drop(F.relu(self.fc1(input)))
        # # input = F.relu(self.fc1_drop(self.fc1(input)))
        # input = self.fc2(input)
        input = F.log_softmax(input, dim=1)
        return input