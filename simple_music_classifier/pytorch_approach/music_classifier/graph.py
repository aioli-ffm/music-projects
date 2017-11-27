import torch
import torch.nn as nn
from torch.autograd import Variable

def mlp_def(chunk_size, num_classes, hid1_size=128, hid2_size=64):
    lin1 = nn.Linear(chunk_size, hid1_size)
    lin1.weight.data.normal_(0, 0.1)
    lin2 = nn.Linear(hid1_size, hid2_size)
    lin2.weight.data.normal_(0, 0.1)
    lin3 = nn.Linear(hid2_size, num_classes)
    lin3.weight.data.normal_(0, 0.1)
    return nn.Sequential(
        lin1, nn.ReLU(),
        lin2, nn.ReLU(),
        lin3
    )

mlp = nn.Sequential(
    #torch.nn.ReLU(),
    #torch.nn.Sigmoid(),
    torch.nn.Conv1d(1, 10, 100),
    #torch.nn.ReLU(),
    torch.nn.MaxPool1d(5),
    #torch.nn.BatchNorm1d(10),
    torch.nn.Conv1d(10, 1, 10),
    torch.nn.Linear(3971, 200),
    #torch.nn.ReLU(),
    torch.nn.Linear(200, 10),
    #torch.nn.ReLU(),
    #torch.nn.Linear(200, 2)

    #torch.nn.Linear(20000, 10),
    #torch.nn.ReLU(),
    #torch.nn.Linear(200, 10),
)
