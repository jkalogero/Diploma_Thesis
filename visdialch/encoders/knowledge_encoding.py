import torch
from torch import nn

class KnowledgeEncoding(nn.Module):
    def __init__(self, config):
        super(KnowledgeEncoding, self).__init__()

        # weights for Query-Guided Relation Selection
        self.w1 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w2 = nn.Linear(config["img_feature_size"], config["lstm_hidden_size"])
        self.we = nn.Linear(config["img_feature_size"], config["lstm_hidden_size"])
        
        # Query-Guided Graph Convolution
        # self.w3 = nn.Linear(config["img_feature_size"], config["lstm_hidden_size"])

    
    def forward(self, batch):
        print("inside Knowledge Enc forward")
        for i in range(len(batch)):
            print('graph ', i, ': ',batch[i])
            print('\t num_nodes = ', batch[i].num_nodes)
            print('\t num_edges = ', batch[i].num_edges)
            break
        res = "1"
        return res
    