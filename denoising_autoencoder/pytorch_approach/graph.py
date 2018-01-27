import torch.nn as nn

class mlp_def(nn.Module):

    def __init__(self, chunk_size, num_classes, hid1_size=300, hid2_size=128):
        super(mlp_def, self).__init__()

        self.lin1 = nn.Linear(chunk_size, hid1_size)
        self.act1 = nn.ReLU(True)

        self.lin2 = nn.Linear(hid1_size, hid2_size)
        self.act2 = nn.ReLU(True)

        self.lin3 = nn.Linear(hid2_size, num_classes)

    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.lin3(x)
        return x
