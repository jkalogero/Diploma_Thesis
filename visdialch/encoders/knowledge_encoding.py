import torch
from torch import nn
from torch.nn.functional import softmax
from torch_geometric.nn import DenseGCNConv


class KnowledgeEncoding(nn.Module):
    def __init__(self, config):
        super(KnowledgeEncoding, self).__init__()
        self.config = config

        self.dropout = nn.Dropout(p=config["dropout"])

        # weights for Query-Guided Relation Selection
        self.w_v_1 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_v_2 = nn.Linear(config["img_feature_size"], config["lstm_hidden_size"])
        self.w_v_e = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])

        self.w_t_1 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t_2 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t_e = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        
        # Query-Guided Graph Convolution
        self.w_v_3 = nn.Linear(config["img_feature_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_v = nn.Linear(config["lstm_hidden_size"],1)
        self.w_t_3 = nn.Linear(config["lstm_hidden_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t = nn.Linear(config["lstm_hidden_size"],1)

    
    def forward(self, img, ques_embed, v_relations, hist_embed, text_rel, batch_size, num_rounds):
        print("KE FORWARD")
        

        # Vision Knowledge Encoding
        img = img.view(batch_size, 1, -1, self.config["img_feature_size"]).repeat(1, num_rounds, 1, 1).view(batch_size * num_rounds, -1, self.config["img_feature_size"])
        
        n_img_objs = img.shape[1] # N = 36
        # Query-Guided Relation Selection
        v_relations = self.w_v_2(v_relations) #
        new_v_relations_dims = v_relations.size(-1)
        # change to shape (batch_size, 36, 36, new_v_relations_dims)
        v_relations = v_relations.view(batch_size,n_img_objs, n_img_objs, new_v_relations_dims).repeat(1,1,num_rounds,1).view(-1,n_img_objs,n_img_objs,new_v_relations_dims)


        ques_embed = self.w_v_1(ques_embed)
        ques_embed = ques_embed.repeat(1,n_img_objs*n_img_objs).view(int(batch_size*num_rounds),n_img_objs,n_img_objs,-1)

        projected_ques_image = ques_embed * v_relations
        
        relation_weight = self.w_v_e(projected_ques_image)
        relation_weight = torch.softmax(relation_weight,-2)
        v_relations = relation_weight * v_relations
        # v_relations.shape =  torch.Size([b*num_rounds, 36, 36, 512])
        # ques_embed.shape =  torch.Size([b*num_rounds, 36, 36, 512])
        print("v_relations.shape = ", v_relations.shape)
        

        # Query-Guided Graph Convolution
        img = img.repeat(1,1,n_img_objs,1).view(int(batch_size*num_rounds),n_img_objs,n_img_objs,-1)
        img_rel_cat = torch.cat((img, v_relations), -1)
        img_rel_cat = self.dropout(img_rel_cat)
        img_rel_cat = self.w_v_3(img_rel_cat)
        
        graph_weight = ques_embed * img_rel_cat
        graph_weight = self.dropout(graph_weight)
        graph_weight = self.w_v(graph_weight)
        graph_weight = torch.softmax(graph_weight,-2)
        updated_v_nodes = torch.sum(graph_weight*img,-2)
        print("updated_v_nodes.shape = ", updated_v_nodes.shape)

        # ===============================================================================================
        # Text Knowledge Encoding


        # Text Knowledge Encoding
        res = ("1", "1")
        return res
    