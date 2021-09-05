import torch
from torch import nn
from torch.nn import functional as F
from visdialch.utils import DynamicRNN
from visdialch.encoders.knowledge_encoding import KnowledgeEncoding

class KBGN(nn.Module):
    def __init__(self, config, vocabulary, glove):
        super(KBGN, self).__init__()
        self.config = config

        self.glove_embed = nn.Embedding(
            len(vocabulary), config["glove_embedding_size"]
        )
        self.glove_embed.weight.data = glove

        self.q_rnn = nn.LSTM(
            config["glove_embedding_size"] + config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.q_rnn = DynamicRNN(self.q_rnn)

        self.KnowldgeEncoder = KnowledgeEncoding(config)
    
    def forward(self, batch):
        print("inside KBGN forward")
        # Visual Knowledge Encoding
        final_embedding = self.KnowldgeEncoder(batch)
        return final_embedding