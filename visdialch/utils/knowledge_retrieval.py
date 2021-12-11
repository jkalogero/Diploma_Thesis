import torch
from torch import nn
from torch.nn.functional import softmax


class KnowledgeRetrieval(nn.Module):
    def __init__(self, config):
        super(KnowledgeRetrieval, self).__init__()
        self.config = config

        self.w_r = nn.Linear(3*config["lstm_hidden_size"], 3*config["lstm_hidden_size"])
        self.w_11 = nn.Linear(3*config["lstm_hidden_size"], config["lstm_hidden_size"])


    def forward(self, I, H, Q, batch_size, num_rounds):

        gate_r = torch.sigmoid(self.w_r(torch.cat((Q,I,H), -1)))
        K = self.w_11(gate_r * torch.cat((Q,I,H), -1))

        return K