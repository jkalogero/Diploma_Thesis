import torch
from torch import nn
from torch.nn.functional import softmax
from torch_geometric.nn import DenseGCNConv


class KnowledgeRetrieval(nn.Module):
    def __init__(self, config):
        super(KnowledgeRetrieval, self).__init__()
        self.config = config

    def forward(self):
        # print("\n\n\t\tRETRIEVAL\n\n")
        return 1