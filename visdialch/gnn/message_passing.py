import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassing(nn.Module):
    def __init__(self, config):
        """
        A custom message passing layer.
        """
        super(MessagePassing, self).__init__()
        
        self.config = config
        self.dropout = config['gnn_dropout']
        self.w_1 = nn.Linear(2*config["numberbatch_embedding_size"], config["lstm_hidden_size"])
        self.w_q = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_e = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_2 = nn.Linear(config["numberbatch_embedding_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_v = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_adj = nn.Linear(config["numberbatch_embedding_size"], config["lstm_hidden_size"])
        

        # nn.init.xavier_uniform_(self.out_w.data, gain=1.414)


    def forward(self, ques_embed, adj_list, deg, batch_size, original_nodes):
        
        batch_size, n_rounds, _ = ques_embed.shape

        # create edges by concatenating the neighbouring nodes
        edges = torch.cat(
            (original_nodes.unsqueeze(-2).repeat(1,1,1,self.config['max_edges'],1), adj_list),
            -1
        )
        ques_embed = self.w_q(ques_embed).view(batch_size, n_rounds, 1, 1, self.config["lstm_hidden_size"])

        # update the edges
        edges = self.w_1(edges)
        a = torch.softmax(self.w_e(torch.mul(ques_embed, edges)), -2)

        edges = a * edges

        # compute the coefficients of the sum
        b = torch.softmax(
            self.w_v(torch.mul(
                ques_embed, 
                self.w_2(torch.cat((adj_list, edges), -1)))),
            -2)
        adj_list = self.w_adj(adj_list)

        
        # apply a weighted sum over all the neighbours
        node_embeddings = torch.sum(b * adj_list, -2)
        # print('node_embeddings.shape', node_embeddings.shape)

        return node_embeddings
        # return F.log_softmax(x, dim=1)
