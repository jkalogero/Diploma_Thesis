import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Modified to take as input an adjacency list.
    """
    def __init__(self, config, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.config = config
        self.dropout = config['gnn_dropout']
        self.slope = config["slope"]
        self.concat = concat
        self.in_features = config['numberbatch_dim']
        self.out_features = config["lstm_hidden_size"]

        self.W = nn.Parameter(torch.empty(size=(self.in_features+config["lstm_hidden_size"], self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.slope)

    def forward(self, adj_list, original_nodes, ques_embed):
        """
        original_nodes.shape: (b,n_rounds, n_nodes, numb_emb)
        adj_list.shape: (b,n_rounds, n_nodes, n_neighbours, numb_emb)
        ques_embed.shape: (b,n_rounds, lstm_hidden_size)
        """
        batch_size, num_rounds, _ = ques_embed.shape
        ques_embed1 = ques_embed.view(batch_size, num_rounds, 1,-1).repeat(
            1,1,self.config['max_nodes'],1
            )
        ques_embed2 = ques_embed.view(batch_size, num_rounds, 1, 1,-1).repeat(
            1,1,self.config['max_nodes'], self.config['max_edges'],1
            )
        original_nodes_q = torch.cat((original_nodes,ques_embed1),-1)
        adj_list_q = torch.cat((adj_list,ques_embed2),-1)

        w_original = torch.matmul(original_nodes_q, self.W) # (b,n_rounds, n_nodes, lstm_hidden_size)
        Wh = torch.matmul(adj_list_q, self.W) # (b,n_rounds, n_nodes, n_edges, lstm_hidden_size)

        attention = self.compute_e(w_original, Wh)


        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention.shape: (b,n_rounds, n_nodes)
        h_prime = torch.matmul(attention, w_original) # (b,n_rounds, n_nodes, lstm_hidden_size)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def compute_e(self, original, Wh):
        """
        Compute e from (2) equation.

        Parameters:
        ===========

        original: Tensor
            Embeddings of the parameterized original nodes (W*original).
            Shape: (b, n_rounds, n_nodes, numb_dim)
        wh: Tensor
            Embeddings of the parameterized neighbours (W*neighbours). 
            Shape: (b, n_rounds, n_nodes, n_neighbours, numb_dim)
        """

        a_w_original = torch.matmul(original, self.a[:self.out_features, :]) # (b,n_rounds, n_nodes, 1)
        a_w_n = torch.matmul(Wh, self.a[self.out_features:, :]).squeeze(-1) # (b,n_rounds, n_nodes, n_edges)
        
        # broadcast add
        e = a_w_original + a_w_n # (b,n_rounds, n_nodes, n_edges)

        return self.leakyrelu(e)

    

class GraphAttentionNetworkQ(nn.Module):
    def __init__(self, config):
        """Dense version of GraphAttentionNetwork."""
        super(GraphAttentionNetworkQ, self).__init__()
        
        self.dropout = config['gnn_dropout']
        self.nheads = config['n_heads']
        self.out_features = config["lstm_hidden_size"]

        self.attentions = [GraphAttentionLayer(config,  concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.out_w = nn.Parameter(torch.empty(size=(self.nheads*self.out_features, self.out_features)))
        nn.init.xavier_uniform_(self.out_w.data, gain=1.414)


    def forward(self, ques_embed, adj_list, deg, batch_size, original_nodes):
        
        node_embeddings = F.dropout(adj_list, self.dropout, training=self.training)
        node_embeddings = torch.cat([att(node_embeddings, original_nodes, ques_embed) for att in self.attentions], dim=-1)
        # adj_list = F.dropout(adj_list, self.dropout, training=self.training)
        node_embeddings = torch.matmul(node_embeddings, self.out_w).squeeze(-2)

        return node_embeddings
        # return F.log_softmax(x, dim=1)
