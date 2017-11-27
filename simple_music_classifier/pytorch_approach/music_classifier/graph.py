import torch
import torch.nn as nn
from torch.autograd import Variable

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