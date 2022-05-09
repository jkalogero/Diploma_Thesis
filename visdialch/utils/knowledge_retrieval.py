import torch
from torch import nn
from torch.nn.functional import softmax


class KnowledgeRetrieval(nn.Module):
    def __init__(self, config):
        super(KnowledgeRetrieval, self).__init__()
        self.config = config

        self.w_i = nn.Linear(2*config["lstm_hidden_size"], 2*config["lstm_hidden_size"])
        self.w_h = nn.Linear(2*config["lstm_hidden_size"], 2*config["lstm_hidden_size"])
        self.w_r = nn.Linear(3*config["lstm_hidden_size"], 3*config["lstm_hidden_size"])
        self.w_i_final = nn.Linear(2*config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_h_final = nn.Linear(2*config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_final = nn.Linear(3*config["lstm_hidden_size"], config["lstm_hidden_size"])


    def forward(self, I_t, I_extk, H_i, H_extk, Q):
        gate_i = torch.sigmoid(self.w_i(torch.cat((I_t, I_extk), -1)))
        I = self.w_i_final(gate_i * torch.cat((I_t, I_extk), -1))

        gate_h = torch.sigmoid(self.w_h(torch.cat((H_i, H_extk), -1)))
        H = self.w_h_final(gate_h * torch.cat((H_i, H_extk), -1))

        gate_r = torch.sigmoid(self.w_r(torch.cat((Q, I, H), -1)))
        K = self.w_final(gate_r * torch.cat((Q,I, H), -1))
        return K