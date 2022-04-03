import torch
from torch import nn
import math
import numpy as np

numb = '/data/scratch/jkalogero/numberbatch_visdial.json.npy'

class GraphConvolution(nn.Module):
    """
    A simple GCN layer that takes as input adjacency list.
    """
    def __init__(self, config):
        super(GraphConvolution, self).__init__()
        self.config = config
        self.emb_token = torch.Tensor(np.load(numb))
        self.embedding = nn.Embedding(5,300)
        self.w_embed = nn.Embedding(
            5, 300
        )
        self.w_embed.weight.data = self.emb_token
        # self.w_gcn = nn.Linear(config['numberbatch_dim'], config["lstm_hidden_size"])
        self.w_gcn = nn.Linear(300, 512)

        # self.weight = nn.Parameter(torch.FloatTensor(config['numberbatch_dim'], config["lstm_hidden_size"]))
        # self.register_parameter('bias', None)
        # self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj_list):
        """
        adj_list: shape = (b,n_nodes,max_rel_num, embedding_size)
        """
        # support = torch.matmul(input, self.weight)
        # output = torch.spmm(adj, support)
        print("adj_list.shape = ", adj_list.shape)
        emb = self.embedding(adj_list) #shape = (b,n_nodes,n_rel,emb_size)
        print("emb.shape = ", emb.shape)
        deg = torch.count_nonzero(adj_list, -1)
        print("deg = ", deg)
        deg_inv_sqrt = deg.pow(-0.5) #shape = (b,n_nodes)
        we_d = torch.randn((4,4,3,1))
        # we_dd = 
        print("deg_inv_sqrt.shape = ", deg_inv_sqrt.shape)
        _sum = torch.sum(emb,-2) #shape = (b,n_nodes,emb_size)
        print("_sum.shape = ", _sum.shape)

        return True


if __name__ == '__main__':
    model = GraphConvolution(1)
    
    adj_list=[
        [[1,2,0], [1,2,3], [2,0,0], [2,3,0]], 
        [[1,2,0], [1,2,3], [2,0,0], [2,3,0]],
        [[1,2,0], [1,2,3], [2,0,0], [2,3,0]], 
        [[1,2,0], [1,2,3], [2,0,0], [2,3,0]], 
    ]
    adj_list = torch.tensor(adj_list, dtype=torch.int64).unsqueeze(1).repeat(1,10,1,1)
    print("adj_list.shape = ", adj_list.shape)
    b,_,n_nodes,n_edges = adj_list.shape
    n_rounds = 10
    q = torch.rand(b,n_rounds,512)
    res = model(adj_list)
