import torch
from torch import nn
from torch.nn.functional import softmax
from torch_geometric.nn import DenseGCNConv


class KnowledgeRetrieval(nn.Module):
    def __init__(self, config):
        super(KnowledgeRetrieval, self).__init__()
        self.config = config

    def forward(self, I, H, Q):
        print("I.shape = ", I.shape)
        print("H.shape = ", H.shape)
        print("Q.shape = ", Q.shape)
        return 1