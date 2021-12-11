import torch
from torch import nn
from torch.nn.functional import softmax


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
        self.w_t_2 = nn.Linear(2*config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t_e = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        
        # Query-Guided Graph Convolution
        self.w_v_3 = nn.Linear(config["img_feature_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_v = nn.Linear(config["lstm_hidden_size"],1)

        self.w_t_3 = nn.Linear(config["lstm_hidden_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t = nn.Linear(config["lstm_hidden_size"],1)

    
    def forward(self, img, ques_embed, v_relations, hist_embed, text_rel, batch_size, num_rounds):
        
        # Vision Knowledge Encoding
        img = img.view(batch_size, 1, -1, self.config["img_feature_size"]).repeat(1, num_rounds, 1, 1).view(batch_size * num_rounds, -1, self.config["img_feature_size"])
        # img.shape = (40, 36, 2048)
        n_img_objs = img.shape[1] # N = 36
        # Query-Guided Relation Selection
        v_relations = self.w_v_2(v_relations) #
        new_v_relations_dims = v_relations.size(-1)
        # change to shape (batch_size, 36, 36, new_v_relations_dims)
        v_relations = v_relations.view(batch_size,n_img_objs, n_img_objs, new_v_relations_dims).repeat(1,1,num_rounds,1).view(-1,n_img_objs,n_img_objs,new_v_relations_dims)


        ques_embed_v = self.w_v_1(ques_embed)
        ques_embed_v = ques_embed_v.repeat(1,n_img_objs*n_img_objs).view(int(batch_size*num_rounds),n_img_objs,n_img_objs,-1)
        # ques_embed_v.shape = (b*num_rounds, 36, 36, 512)

        projected_ques_image = ques_embed_v * v_relations
        # projected_ques_image.shape = (b*num_rounds, 36, 36, 512)
        relation_weight_v = self.w_v_e(projected_ques_image)
        relation_weight_v = torch.softmax(relation_weight_v,-2)
        # relation_weight_v.shape = (b*num_rounds, 36, 36, 512)
        v_relations = relation_weight_v * v_relations
        # v_relations.shape = (b*num_rounds, 36, 36, 512)
        

        # Query-Guided Graph Convolution
        img = img.repeat(1,1,n_img_objs,1).view(int(batch_size*num_rounds),n_img_objs,n_img_objs,-1)
        # img.shape = (b*num_rounds, 36, 36, 2048)
        img_rel_cat = torch.cat((img, v_relations), -1)
        img_rel_cat = self.dropout(img_rel_cat)
        img_rel_cat = self.w_v_3(img_rel_cat)
        # img_rel_cat.shape = (b*num_rounds, 36, 36, 512)

        graph_weight_v = ques_embed_v * img_rel_cat
        graph_weight_v = self.dropout(graph_weight_v)
        graph_weight_v = self.w_v(graph_weight_v)
        graph_weight_v = torch.softmax(graph_weight_v,-2)
        # graph_weight_v.shape = b*num_rounds, 36, 36, 1
        updated_v_nodes = torch.sum(graph_weight_v*img,-2)
        # updated_v_nodes.shape = (b*num_rounds, 36, 2048)

        # ===============================================================================================
        # Text Knowledge Encoding
        text_rel = text_rel.view(int(batch_size*num_rounds), num_rounds, num_rounds, 2*self.config["lstm_hidden_size"])
        text_rel = self.w_t_2(text_rel) # text_rel.shape =  (b*num_rounds, 10, 10, 512)

        ques_embed_t = self.w_t_1(ques_embed)
        ques_embed_t = ques_embed_t.repeat(1,num_rounds*num_rounds).view(int(batch_size*num_rounds),num_rounds,num_rounds,-1)
        # ques_embed_t.shape =  (b*num_rounds, 10, 10, 512)

        projected_ques_text = ques_embed_t * text_rel
        # projected_ques_text.shape = (b*num_rounds, 10, 10, 512)
        relation_weight_t = self.w_t_e(projected_ques_text)
        relation_weight_t = torch.softmax(relation_weight_t,-2)
        # relation_weight_t.shape = (b*num_rounds, 10, 10, 512)
        t_relations = relation_weight_t * text_rel
        # t_relations.shape = (b*num_rounds, 10, 10, 512)
        
        # Query-Guided Graph Convolution
        # hist_embed.shape = (4, 10, 10, 512)
        hist_embed = hist_embed.view(
            batch_size*num_rounds, -1, 1, self.config["lstm_hidden_size"]).repeat(
            1,1,num_rounds,1)
        # hist_embed.shape = (40, 10, 10, 512)
        text_rel_cat = torch.cat((hist_embed, text_rel), -1)
        text_rel_cat = self.dropout(text_rel_cat)
        text_rel_cat = self.w_t_3(text_rel_cat)
        # text_rel_cat.shape = (b*num_rounds, 10, 10, 512)

        graph_weight_t = ques_embed_t * text_rel_cat
        graph_weight_t = self.dropout(graph_weight_t)
        graph_weight_t = self.w_t(graph_weight_t)
        graph_weight_t = torch.softmax(graph_weight_t,-2)
        # graph_weight_t.shape = (b*num_rounds, 10, 10, 1)
        updated_t_nodes = torch.sum(graph_weight_t*hist_embed,-2)
        # updated_t_nodes.shape = (b*num_rounds, 10, 512)

        res = (updated_v_nodes, updated_t_nodes)
        return res
    