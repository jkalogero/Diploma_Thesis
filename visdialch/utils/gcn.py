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
        self.w_gcn = nn.Linear(config['numberbatch_dim'], config["lstm_hidden_size"])

        # self.weight = nn.Parameter(torch.FloatTensor(config['numberbatch_dim'], config["lstm_hidden_size"]))
        # self.register_parameter('bias', None)
        # self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj_list, original_limit, batch_size, max_original_nodes=30, keep_original=True):
        """
        adj_list: shape = (b,n_nodes,max_rel_num, embedding_size)
        """
        # support = torch.matmul(input, self.weight)
        # output = torch.spmm(adj, support)
        print("adj_list.shape = ", adj_list.shape) # shape = (b, n_rounds, iniital_nodes, max_rel)
        
        deg = torch.count_nonzero(adj_list, -1)
        print("deg.shape = ", deg.shape)
        deg_inv_sqrt = deg.pow(-0.5) #shape = (b,n_nodes)
        we_d = torch.randn((4,4,3,1))
        # we_dd = 
        print("deg_inv_sqrt.shape = ", deg_inv_sqrt.shape)
        node_embeddings = torch.sum(adj_list,-2) #shape = (b,n_nodes,emb_size)


        # keep only original nodes
        if keep_original:
            batch_size, n_rounds, n_rel, emb_size = node_embeddings.shape
            node_embeddings = node_embeddings.view(batch_size*n_rounds, n_rel, emb_size)

            original_mask = torch.zeros(node_embeddings.shape[0], node_embeddings.shape[1], dtype=node_embeddings.dtype, device=node_embeddings.device)
            # assing 1 on the index of the original limit
            original_mask[(torch.arange(node_embeddings.shape[0]), original_limit.view(batch_size*n_rounds))] = 1
            # all values after limit will be one
            original_mask = original_mask.cumsum(dim=1)

            node_embeddings = node_embeddings * (1. - original_mask[..., None])     # use mask to zero after each column
            node_embeddings = node_embeddings.view(batch_size, n_rounds, n_rel, emb_size)
            node_embeddings = node_embeddings[:,:,:max_original_nodes,:]
            # node_embeddings.shape = (b, n_rounds, max_rel_nodes, emb_size)

        return node_embeddings


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
