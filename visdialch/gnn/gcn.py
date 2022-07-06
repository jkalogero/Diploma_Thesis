import torch
from torch import nn
import math
import numpy as np
from torch.nn.functional import softmax


class GraphConvolution(nn.Module):
    """
    A simple GCN layer that takes as input an adjacency list.
    """
    def __init__(self, config):
        super(GraphConvolution, self).__init__()
        self.config = config
        node_emb_size = config[config['ext_knowledge_emb']+'_embedding_size']
        self.w_adj = nn.Linear(node_emb_size, node_emb_size)
        self.w_gcn = nn.Linear(node_emb_size+config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_sum = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, question, adj_list, node_degrees, batch_size, max_original_nodes=30, original_nodes=None, keep_original=True):
        """
        adj_list: shape = (b,n_nodes,max_rel_num, embedding_size)
        """
        
        # deg_inv_sqrt = (node_degrees).unsqueeze(-1)
        # compute node degrees
        node_degrees=node_degrees.float()
        node_degrees[node_degrees.nonzero(as_tuple=True)]= torch.pow(node_degrees[node_degrees.nonzero(as_tuple=True)],-2)
        deg_inv_sqrt = node_degrees.unsqueeze(-1)
        # shape = (b,n_rounds,n_nodes,1)

        adj_list = self.w_adj(adj_list)
        # adj_list.shape = [b, n_rounds, n_nodes, n_rel, emb_size]

        question = question.view(batch_size, question.shape[1],1,1,question.shape[-1]).repeat(1,1,adj_list.shape[-3],adj_list.shape[-2],1)
        
        adj_list_q = self.w_gcn(torch.cat((adj_list,question),-1))
        # question = question.view(batch_size, question.shape[1],1,1,question.shape[-1])

        # coef = torch.softmax(self.w_gcn(question * adj_list),-2)
        # adj_list_q = coef * adj_list
        # adj_list_q.shape = [b, n_rounds, n_nodes, n_rel, lstm_hidden_size]

        # node_embeddings = self.w_sum(torch.sum(adj_list_q,-2)) 
        node_embeddings = self.w_sum(deg_inv_sqrt * torch.sum(adj_list_q,-2)) 
        node_embeddings = torch.softmax(node_embeddings, -1)
        # node_embeddings.shape = (b,n_rounds,n_nodes,emb_size)

        return node_embeddings


if __name__ == '__main__':
    import yaml
    config = config = yaml.load(open('configs/default.yml'))
    model = GraphConvolution(config['model'])
    
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
