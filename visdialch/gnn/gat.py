import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Modified to take as input adjacency list.
    """
    def __init__(self, config, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = config['gnn_dropout']
        self.slope = config["slope"]
        self.concat = concat
        self.in_features = config['numberbatch_dim']
        self.out_features = config["lstm_hidden_size"]

        # self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        self.W = nn.Linear(self.in_features, self.out_features)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.slope)

    def forward(self, adj_list):
        Wh = torch.mm(adj_list, self.W) # h.shape: (N, numberbatch_dim), Wh.shape: (N, lstm_hidden_size)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e) # initialize
        attention = torch.where(adj_list > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    

class GAT(nn.Module):
    def __init__(self, config):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = config['gnn_dropout']
        self.nheads = config['n_heads']

        self.attentions = [GraphAttentionLayer(config,  concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, ques_embed, adj_list, deg, batch_size):
        print('adj_list.shape = ', adj_list.shape)
        adj_list = F.dropout(adj_list, self.dropout, training=self.training)
        adj_list = torch.cat([att(adj_list) for att in self.attentions], dim=1)
        # adj_list = F.dropout(adj_list, self.dropout, training=self.training)
        
        return adj_list
        # return F.log_softmax(x, dim=1)
