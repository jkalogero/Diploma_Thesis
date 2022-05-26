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
                triples=None,
                bias=True,
                diag_weight_matrix=False,
                reset_mode='glorot_uniform'):
        super(RelationalGraphConvolution, self).__init__()

        # If featureless, use number of nodes instead as input dimension
        # in_dim = in_features if in_features is not None else num_nodes
        # out_dim = out_features

        self.triples = triples
        self.num_nodes = config['max_nodes']
        self.num_relations = config['num_relations']*2 + 1 # bidirectional + self relation
        self.in_features = config['numberbatch_embedding_size']
        self.out_features = config['lstm_hidden_size']
        self.weight_decomp = config['decomposition_type']
        self.num_bases = config['num_bases']
        self.num_blocks = config['num_blocks']
        
        self.diag_weight_matrix = diag_weight_matrix
        self.dropout = config['gnn_dropout']

        # If diagonal matrix
        if self.diag_weight_matrix:
            self.weights = torch.nn.Parameter(torch.empty((self.num_relations, self.in_features)), requires_grad=True)
            self.out_features = self.in_features
            self.weight_decomp = None
            bias = False

        # Instantiate weights
        elif self.weight_decomp is None:
            self.weights = Parameter(torch.FloatTensor(self.num_relations, self.in_features, self.out_features))
        
        # ===================================================================================
        elif self.weight_decomp == 'basis':
            # Weight Regularisation through Basis Decomposition
            assert self.num_bases > 0, \
                'Number of bases should be set to higher than zero for basis decomposition!'
            
            self.bases = Parameter(torch.FloatTensor(self.num_bases, self.in_dim, self.out_dim))
            self.comps = Parameter(torch.FloatTensor(self.num_relations, self.num_bases))
        
        # ===================================================================================
        elif self.weight_decomp == 'block':
            # Weight Regularisation through Block Diagonal Decomposition
            assert self.num_blocks > 0, \
                'Number of blocks should be set to a value higher than zero for block diagonal decomposition!'
            assert self.in_dim % self.num_blocks == 0 and self.out_dim % self.num_blocks == 0,\
                f'For block diagonal decomposition, input dimensions ({self.in_dim}, {self.out_dim}) must be divisible ' \
                f'by number of blocks ({self.num_blocks})'
            self.blocks = nn.Parameter(
                torch.FloatTensor(self.num_relations, self.num_blocks, self.in_dim // self.num_blocks, self.out_dim // self.num_blocks))
        else:
            raise NotImplementedError(f'{self.weight_decomp} decomposition has not been implemented')

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

    def forward(self, ques_embed, adj_list_emb, deg, batch_size, original_nodes_emb): # features
        """ Perform a single pass of message propagation """


        in_dim = self.in_features
        triples = self.triples
        out_dim = self.out_features
        edge_dropout = self.edge_dropout
        weight_decomp = self.weight_decomp
        num_nodes = self.num_nodes
        num_relations = self.num_relations
        vertical_stacking = self.vertical_stacking
        general_edge_count = int((triples.size(0) - num_nodes)/2) # remove self relations and divide by 2 because of bidirectional
        self_edge_count = num_nodes # self relations

        # Choose weights
        if weight_decomp is None:
            weights = self.weights
        elif weight_decomp == 'basis':
            weights = torch.einsum('rb, bio -> rio', self.comps, self.bases)
        elif weight_decomp == 'block':
            weights = block_diag(self.blocks)
        


        # Stack adjacency matrices either vertically or horizontally
        # adj_indices, adj_size = stack_matrices(
        #     triples,
        #     num_nodes,
        #     num_relations,
        #     vertical_stacking=vertical_stacking,
        #     device=device
        # )
        num_triples = adj_indices.size(0)
        vals = torch.ones(num_triples, dtype=torch.float, device=device)

        # Apply normalization (vertical-stacking -> row-wise sum & horizontal-stacking -> column-wise sum)
        sums = sum_sparse(adj_indices, vals, adj_size, row_normalisation=vertical_stacking, device=device)
        
        vals = vals / sums

        # Construct adjacency matrix
        adj = torch.cuda.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)

        if self.diag_weight_matrix:
            assert weights.size() == (num_relations, in_dim)
        else:
            assert weights.size() == (num_relations, in_dim, out_dim)

        # if self.in_features is None:
        #     # Message passing if no features are given
        #     output = torch.mm(adj, weights.view(num_relations * in_dim, out_dim))
        # elif self.diag_weight_matrix:
        #     fw = torch.einsum('ij,kj->kij', features, weights)
        #     fw = torch.reshape(fw, (self.num_relations * self.num_nodes, in_dim))
        #     output = torch.mm(adj, fw)
        if self.vertical_stacking:
            # Message passing if the adjacency matrix is vertically stacked
            af = torch.spmm(adj, features)
            af = af.view(self.num_relations, self.num_nodes, in_dim)
            output = torch.einsum('rio, rni -> no', weights, af)
        
        # else:
        #     # Message passing if the adjacency matrix is horizontally stacked
        #     fw = torch.einsum('ni, rio -> rno', features, weights).contiguous()
        #     output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))

        assert output.size() == (self.num_nodes, out_dim)
        
        if self.bias is not None:
            output = torch.add(output, self.bias)
        
        return output


