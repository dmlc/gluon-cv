import torch
from torch.nn.functional import softmax

def rmse(outputs, target):
    return torch.sqrt(torch.mean((softmax(outputs, dim=0)-target)**2))
