import torch
from torch import nn
from torch.nn.functional import softmax


class KnowledgeRetrieval(nn.Module):
    def __init__(self, config):
        super(KnowledgeRetrieval, self).__init__()
        self.config = config

        self.w_r = nn.Linear(5*config["lstm_hidden_size"], 5*config["lstm_hidden_size"])
        self.w_11 = nn.Linear(5*config["lstm_hidden_size"], config["lstm_hidden_size"])


    def forward(self, I_t, I_extk, H_i, H_extk, Q):

        gate_r = torch.sigmoid(self.w_r(torch.cat((Q,I_t, I_extk, H_i, H_extk), -1)))
        K = self.w_11(gate_r * torch.cat((Q,I_t, I_extk, H_i, H_extk), -1))
        print("K.shape = ", K.shape)
        return K