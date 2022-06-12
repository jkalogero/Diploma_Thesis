from torch import nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math
import torch
from .rgcn_utils import *


class RelationalGraphConvolution(Module):
    """
    Relational Graph Convolution (RGC) Layer for Node Classification
    (as described in https://arxiv.org/abs/1703.06103)
    Accepts the graph in an adjacency list and also a list
    of the original nodes.
    """
    def __init__(self,
                config,
                bias=True,
                reset_mode='glorot_uniform'):
        super(RelationalGraphConvolution, self).__init__()


        self.num_nodes = config['max_nodes']
        self.num_relations = config['reduced_num_relations']*2 + 1 # bidirectional + self relation
        self.in_features = config['numberbatch_embedding_size']
        self.out_features = config['lstm_hidden_size']
        self.weight_decomp = config['decomposition_type']
        self.num_bases = config['num_bases']
        
        self.dropout = config['gnn_dropout']

        
        # Weight Regularisation through Basis Decomposition
        self.bases = Parameter(torch.FloatTensor(self.num_bases, self.in_features, self.out_features))
        self.comps = Parameter(torch.FloatTensor(self.num_relations, self.num_bases))
    
        
        # Instantiate biases
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else: 
            self.register_parameter('bias', None)
            
        self.reset_parameters(reset_mode)
    
    def reset_parameters(self, reset_mode='glorot_uniform'):
        """ Initialise biases and weights (glorot_uniform or uniform) """

        if reset_mode == 'glorot_uniform':
            if self.weight_decomp == 'block':
                nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))
            elif self.weight_decomp == 'basis':
                nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        
        elif reset_mode == 'schlichtkrull':
            if self.weight_decomp == 'block':
                nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))
            elif self.weight_decomp == 'basis':
                nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        elif reset_mode == 'uniform':
            stdv = 1.0 / math.sqrt(self.weights.size(1))
            if self.weight_decomp == 'block':
                self.blocks.data.uniform_(-stdv, stdv)
            elif self.weight_decomp == 'basis':
                self.bases.data.uniform_(-stdv, stdv)
                self.comps.data.uniform_(-stdv, stdv)
            else:
                self.weights.data.uniform_(-stdv, stdv)

            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise NotImplementedError(f'{reset_mode} parameter initialisation method has not been implemented')

    def forward(self, ques_embed, adj_list, deg, batch_size, original_nodes):
        """
        Perform a single pass of message propagation.

        original_nodes.shape: (b,n_rounds, n_nodes, numb_emb)
        adj_list.shape: (b,n_rounds, n_nodes, n_neighbours, numb_emb)
        ques_embed.shape: (b,n_rounds, lstm_hidden_size)
        
        """


        in_dim = self.in_features
        out_dim = self.out_features
        num_relations = self.num_relations
        num_rounds = ques_embed.shape[1]

        # Choose weights
        weights = torch.einsum('rb, bio -> rio', self.comps, self.bases)
        
        # Apply normalization
        # compute node degrees
        norm=deg.float()
        norm[norm.nonzero(as_tuple=True)]= torch.pow(norm[norm.nonzero(as_tuple=True)],-1)
        # norm = 1 / deg # shape: (b,n_rounds,n_nodes*n_rels)
        

        assert weights.size() == (num_relations, in_dim, out_dim) # shape: (n_rels,numb_size,lstm_hidden_size)
        
        
        # Message passing for each relation
        # af = torch.mm(adj_list, norm) # first sum
        sum_per_rel = torch.sum(adj_list,-2)
        sum_per_rel = (sum_per_rel*(norm.unsqueeze(-1))).view(
            batch_size,num_rounds, self.num_relations,self.num_nodes,self.in_features) # shape: (b,n_rounds, n_rel, n_nodes, numb_size)

        # sum all relations using the weights
        output = torch.einsum('bdrni, rio -> bdno', sum_per_rel, weights) # shape: (b,n_rounds, n_nodes, lstm_hidden_size)
        
        if self.bias is not None:
            output = torch.add(output, self.bias)
        
        return output


