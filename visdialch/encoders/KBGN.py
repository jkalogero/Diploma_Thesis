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

        self.dropout = nn.Dropout(p=config["dropout"])

        self.KnowldgeEncoder = KnowledgeEncoding(config)
    
    def forward(self, batch):
        for i, obj in enumerate(batch):
            print('graph ', i, ': ',batch[i])
            print('\t num_nodes = ', batch[i].num_nodes)
            print('\t num_edges = ', batch[i].num_edges)
            question = batch[i]['questions']
            print("question = ", question)
            break
        
        # Get data
        batch_size = batch.num_graphs
        ques = batch['questions']
        print(ques.size())
        _, max_sequence_length = ques.size()

        # Embed questions
        # ques = ques.view(batch_size * num_rounds, max_sequence_length)
        print(ques.device)
        ques_embed_glove = self.glove_embed(ques)
        ques_embed_glove = self.dropout(ques_embed_glove)
        # ques_embed_elmo = self.elmo_embed(ques)
        # ques_embed_elmo = self.dropout(ques_embed_elmo)
        # ques_embed_elmo = self.embed_change(ques_embed_elmo)
        # ques_embed = torch.cat((ques_embed_glove,ques_embed_elmo),-1)
        _, (ques_embed, _) = self.ques_rnn(ques_embed_glove, batch["ques_len"])
        print(ques_embed.shape)

        # Visual Knowledge Encoding
        final_embedding = self.KnowldgeEncoder(batch)
        return final_embedding