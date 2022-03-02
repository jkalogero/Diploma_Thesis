import torch
from torch import nn
from torch.nn import functional as F
from visdialch.utils import DynamicRNN
from torch.nn.utils.rnn import pad_sequence
from visdialch.utils.knowledge_encoding import KnowledgeEncoding
from visdialch.utils.knowledge_storage import KnowledgeStorage
from visdialch.utils.knowledge_retrieval import KnowledgeRetrieval

class KBGN(nn.Module):
    def __init__(self, config, vocabulary, embedding, elmo, numberbatch):
        """
        Parameters:
        ===========
        embedding: glove || numberbatch

        """
        super(KBGN, self).__init__()
        self.config = config
        
        emb_size = "numberbatch_embedding_size" if numberbatch else "glove_embedding_size"

        self.w_embed = nn.Embedding(
            len(vocabulary), config[emb_size]
        )
        self.w_embed.weight.data = embedding
        
        # self.numberbatch_embed = nn.Embedding(len(vocabulary), config["numberbatch_embedding_size"])
        # self.numberbatch_embed.weight.data.copy_(embedding)
        
        self.elmo_embed = nn.Embedding(
            len(vocabulary), config["elmo_embedding_size"]
        )
        
        self.elmo_embed.weight.data = elmo
        # self.glove_embed.weight.requires_grad = False
        self.elmo_embed.weight.requires_grad = False

        self.embed_change = nn.Linear(
            config["elmo_embedding_size"],config["word_embedding_size"]
        )


        self.q_rnn = nn.LSTM(
            config[emb_size] + config["word_embedding_size"],
            # config["glove_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.q_rnn = DynamicRNN(self.q_rnn)

        self.hist_rnn = nn.LSTM(
            config[emb_size] + config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )

        self.hist_rnn = DynamicRNN(self.hist_rnn)

        self.dropout = nn.Dropout(p=config["dropout"])

        self.KnowldgeEncoder = KnowledgeEncoding(config)
        self.KnowldgeStorage = KnowledgeStorage(config)
        self.KnowldgeRetrieval = KnowledgeRetrieval(config)



    def forward(self, batch):
        # Get data

        img = batch["img_feat"]
        v_relations = batch["relations"]
        ques = batch["ques"]
        hist = batch["hist"]
        batch_size, num_rounds, max_sequence_length = ques.size()
        
        # Embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        # print("ques.device = ",ques.device)
        ques_embed_emb = self.w_embed(ques)
        ques_embed_elmo = self.elmo_embed(ques)
        ques_embed_elmo = self.dropout(ques_embed_elmo)
        ques_embed_elmo = self.embed_change(ques_embed_elmo)
        ques_embed = torch.cat((ques_embed_emb,ques_embed_elmo),-1)
        _, (ques_embed, _) = self.q_rnn(ques_embed, batch["ques_len"])
        # print("ques_embed.shape = ", ques_embed.shape)
        ques_embed = ques_embed.view(batch_size, num_rounds, -1)
        # Embed history
        # print('batch_size= ', batch_size , 'num_rounds = ', num_rounds)
        hist = hist.view(batch_size * num_rounds, max_sequence_length * 2)
        # print("hi.shape = ", hist.shape)

        hist_embed_emb = self.w_embed(hist)
        # hist_embed_emb = self.dropout(hist_embed_emb) # delete
        hist_embed_elmo = self.elmo_embed(hist)
        hist_embed_elmo = self.dropout(hist_embed_elmo)
        hist_embed_elmo = self.embed_change(hist_embed_elmo)
        # print("hist_embed_elmo.shape = ", hist_embed_elmo.shape)
        # print("hist_embed_emb.shape = ", hist_embed_emb.shape)
        hist_embed = torch.cat((hist_embed_emb, hist_embed_elmo), -1)
        # print("hist_embed_emb.shape = ", hist_embed_emb.shape)
        
        _, (hist_embed, _) = self.hist_rnn(hist_embed, batch["hist_len"])
        
        # print("hist_embed.shape = ", hist_embed.shape)
        hist_embed = hist_embed.view(batch_size, num_rounds, -1)
        
        # construct semantic graph
        for index, b in enumerate(hist_embed):
            concatenated_history = []
            for i in range(len(b)):
                concatenated_history.append(b[:i+1])
            
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
            # maxpadded_history[:, :padded_history.size(1),:] = padded_history
            maxpadded_history = padded_history # FIX
            
            maxpadded_history = maxpadded_history.unsqueeze(0)
            
            if index == 0 :
                f_history = maxpadded_history
            else:
                f_history = torch.cat((f_history, maxpadded_history), 0)
        
        
        # print("HIST = ", f_history.shape)   
        # f_history.shape = (b, n_rounds, n_rounds, 512)
        # Create semantic relationships
        t_rel = f_history.view(batch_size, num_rounds, num_rounds, 1, self.config["lstm_hidden_size"]).repeat(1,1, 1, num_rounds, 1)
        # print("t_rel.shape = ", t_rel.shape)
        mask1 = t_rel.abs().sum(dim=-1).bool()
        # print(mask1.shape)
        tmp = f_history.view(batch_size, num_rounds, 1,  num_rounds, self.config["lstm_hidden_size"]).repeat(1,1, num_rounds, 1, 1)
        # print("tmp.shape = ", tmp.shape)
        mask2 = tmp.abs().sum(dim=-1).bool()
        # print(mask2.shape)

        # 
        t_rel[~mask2] = torch.zeros((self.config["lstm_hidden_size"]), device=t_rel.device)
        tmp[~mask1] = torch.zeros((self.config["lstm_hidden_size"]), device=tmp.device)
        text_rel = torch.cat((t_rel, tmp), -1)
        # delete
        # del t_rel
        # del tmp

        # text_rel.shape = (4, 10, 10, 10, 1024)
        
        # torch.set_printoptions(threshold=10_000)
        # torch.set_printoptions(linewidth=200)
        # print("text_rel.shape = ", text_rel.shape)
        # print("first round")
        # print(text_rel[0][0])
        # print("second round")
        # print(text_rel[0][1])
        # print("third round")
        # print(text_rel[0][2])
        
        # print()

        # Knowledge Encoding
        updated_v_nodes, updated_t_nodes = self.KnowldgeEncoder(img, ques_embed, v_relations, f_history, text_rel, batch_size, num_rounds)
        # Knowledge Storage
        I, H = self.KnowldgeStorage(updated_v_nodes, updated_t_nodes, ques_embed, batch_size, num_rounds)
        # Knowledge Retrieval
        final_embedding = self.KnowldgeRetrieval(I, H, ques_embed)
        final_embedding = final_embedding.view(batch_size, num_rounds, -1)
        
        return final_embedding