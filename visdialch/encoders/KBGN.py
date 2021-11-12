import torch
from torch import nn
from torch.nn import functional as F
from visdialch.utils import DynamicRNN
from torch.nn.utils.rnn import pad_sequence
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
            # config["glove_embedding_size"] + config["word_embedding_size"],   CHANGE
            config["glove_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.q_rnn = DynamicRNN(self.q_rnn)

        self.hist_rnn = nn.LSTM(
            config["glove_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )

        self.hist_rnn = DynamicRNN(self.hist_rnn)

        self.dropout = nn.Dropout(p=config["dropout"])

        self.KnowldgeEncoder = KnowledgeEncoding(config)



    def forward(self, batch):
        # Get data

        img = batch["img_feat"]
        print("img.device = ", img.device)
        v_relations = batch["relations"]
        ques = batch["ques"]
        hist = batch["hist"]
        batch_size, num_rounds, max_sequence_length = ques.size()
        print("hist.shape = ", hist.shape)
        print(hist[0][0])
        # Embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        # print("ques.device = ",ques.device)
        ques_embed_glove = self.glove_embed(ques)
        ques_embed_glove = self.dropout(ques_embed_glove)# delete
        # ques_embed_elmo = self.elmo_embed(ques)
        # ques_embed_elmo = self.dropout(ques_embed_elmo)
        # ques_embed_elmo = self.embed_change(ques_embed_elmo)
        # ques_embed = torch.cat((ques_embed_glove,ques_embed_elmo),-1)
        _, (ques_embed, _) = self.q_rnn(ques_embed_glove, batch["ques_len"])
        # print("ques_embed.shape = ", ques_embed.shape)

        # Embed history
        hist = hist.view(batch_size * num_rounds, max_sequence_length * 2)
        hist_embed_glove = self.glove_embed(hist)
        hist_embed_glove = self.dropout(hist_embed_glove) # delete
        # hist_embed_elmo = self.elmo_embed(hist)
        # hist_embed_elmo = self.dropout(hist_embed_elmo)
        # hist_embed_elmo = self.embed_change(hist_embed_elmo)
        # hist_embed = torch.cat((hist_embed_glove, hist_embed_elmo), -1)
        # print("hist_embed_glove.shape = ", hist_embed_glove.shape)
        
        _, (hist_embed, _) = self.hist_rnn(hist_embed_glove, batch["hist_len"])
        
        # print("hist_embed.shape = ", hist_embed.shape)
        hist_embed = hist_embed.view(batch_size, num_rounds, -1)
        print("hist_embed.shape = ", hist_embed.shape)
        
        # construct semantic graph
        for index, b in enumerate(hist_embed):
            concatenated_history = []
            for i in range(len(b)):
                concatenated_history.append(b[:i+1])
            
            print([t.size() for t in concatenated_history])
            maxpadded_history = torch.full(
            (len(concatenated_history),len(concatenated_history), 512),
            fill_value=0, #self.vocabulary.PAD_INDEX
            )
            padded_history = pad_sequence(
                [round_history for round_history in concatenated_history],
                batch_first=True, padding_value=0 #self.vocabulary.PAD_INDEX
            )
            # print("maxpadded_history.shape = ", maxpadded_history.shape)
            # print("padded_history.shape = ", padded_history.shape)

            # print("print(padded_history[0][0]) = ", padded_history[0][0])
            # print("print(padded_history[1][0]) = ", padded_history[1][0])
            # tmp = maxpadded_history
            # maxpadded_history[:, :padded_history.size(1),:] = padded_history
            maxpadded_history = padded_history # FIX
            # print("\n\n\t\tEQUAL ", torch.equal(tmp, maxpadded_history))
            # maxpadded_history[:,:padded_history.size(1)]=padded_history
            # print("print(maxpadded_history[0][0]) = ", maxpadded_history[0][0])
            # print("\n\nNONZERO: ", torch.nonzero(padded_history))
            # print("\n\nNONZERO: ", torch.nonzero(maxpadded_history))
            maxpadded_history = maxpadded_history.unsqueeze(0)
            print(maxpadded_history[0][padded_history.size(1)-1])
            if index == 0 :
                print("index = 0")
                f_history = maxpadded_history
            else:
                f_history = torch.cat((f_history, maxpadded_history), 0)
        
        
        print("HIST = ", f_history.shape)
        # print("first round")
        # print(f_history[0][0][0])
        # print(f_history[0][0][1])
        # print("second round")
        # print(f_history[0][1][0])
        # print(f_history[0][1][1])
        # print(f_history[0][1][2])
        # print("third round")
        # print(f_history[0][2][0])
        # print(f_history[0][2][1])
        # print(f_history[0][2][2])
        # print(f_history[0][2][3])
        # Create semantic relationships
        t_rel = f_history.view(batch_size, num_rounds, num_rounds, 1, self.config["lstm_hidden_size"]).repeat(1,1, 1, num_rounds, 1)
        print("t_rel.shape = ", t_rel.shape)
        tmp = f_history.view(batch_size, num_rounds, 1,  num_rounds, self.config["lstm_hidden_size"]).repeat(1,1, num_rounds, 1, 1)
        print("tmp.shape = ", tmp.shape)
        text_rel = torch.cat((t_rel, tmp), -1)
        print("text_rel.shape = ", text_rel.shape)
        print()

        # Visual Knowledge Encoding
        updated_v_nodes, updated_s_nodes = self.KnowldgeEncoder(img, ques_embed, v_relations, f_history, text_rel, batch_size, num_rounds)
        final_embedding = 1
        # final_embedding should have shape (batch_size, num_rounds, -1)
        return final_embedding