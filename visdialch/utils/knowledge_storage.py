import torch
from torch import nn
from torch.nn.functional import softmax


class KnowledgeStorage(nn.Module):
    def __init__(self, config):
        super(KnowledgeStorage, self).__init__()
        self.config = config

        self.dropout = nn.Dropout(p=config["dropout"])

        # Query-Guided Bridge Update
        self.w_t2v_4 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t2v_5 = nn.Linear(config["img_feature_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t2v_b = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])

        self.w_v2t_4 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_v2t_5 = nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"])
        self.w_v2t_b = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        
        # Query-Guided Cross Graph Convolution
        self.w_t2v_6 = nn.Linear(2*config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t2v_c = nn.Linear(config["lstm_hidden_size"], 1)

        self.w_v2t_6 = nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"])
        self.w_v2t_c = nn.Linear(config["lstm_hidden_size"], 1)

        # Local Knowledge Storage
        self.w_t2v_l = nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"] + config["img_feature_size"])
        self.w_t2v_7 = nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"])

        self.w_v2t_l = nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"] + config["img_feature_size"])
        self.w_v2t_7 = nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"])

        # Global Knowledge Storage
        self.w_t2v_e = nn.Linear(config["lstm_hidden_size"], 1)
        self.w_t2v_8 = nn.Linear(config["img_feature_size"], config["lstm_hidden_size"])
        self.w_t2v_a = nn.Linear(config["lstm_hidden_size"], 1)
        self.w_t2v_9 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_t2v_lg = nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"] + config["img_feature_size"])
        self.w_t2v_num_rounds = nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"])

        self.w_v2t_e = nn.Linear(config["lstm_hidden_size"], 1)
        self.w_v2t_8 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_v2t_a = nn.Linear(config["lstm_hidden_size"], 1)
        self.w_v2t_9 = nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"])
        self.w_v2t_lg = nn.Linear(config["lstm_hidden_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"] + config["lstm_hidden_size"])
        self.w_v2t_num_rounds = nn.Linear(config["lstm_hidden_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"])

    
    def forward(self, v_nodes, t_nodes, ques_embed, batch_size, num_rounds):
        # v_nodes = v_nodes.view(batch_size*num_rounds, v_nodes.shape[2], v_nodes.shape[3])
        # t_nodes = t_nodes.view(batch_size*num_rounds, t_nodes.shape[2], t_nodes.shape[3])
        n_objects = v_nodes.shape[2]
        # Vision Knowledge Storage - Cross Bridge
        # Query-Guided Bridge Update
        # B_v_ij = [v_i, s_j], shape = (b, num_rounds, n_objects, num_rounds, 2560)
        b_v = self.constructCrossGraphEdges(v_nodes, t_nodes, num_rounds, n_objects)
        b_v = self.w_t2v_5(b_v) # shape: (b*num_rounds, n_objects, num_rounds, 512)
        ques_embed_t2v = self.w_t2v_4(ques_embed)
        # convert question from shape (b*num_rounds, 512) to (b*num_rounds, n_objects, num_rounds, 512)
        ques_embed_t2v=ques_embed_t2v.view(batch_size,num_rounds, 1, 1, self.config["lstm_hidden_size"])
        # .repeat(1,1,n_objects,num_rounds,1)
        prod_t2v = ques_embed_t2v * b_v
        prod_t2v = self.w_t2v_b(prod_t2v)
        updated_bridge_t2v = torch.softmax(prod_t2v, -2) # γ_ij
        updated_b_v = updated_bridge_t2v * b_v # (6) shape: (b,num_rounds, n_objects, num_rounds, 512)

        # Query-Guided Cross Graph Convolution
        t_nodes_t2v = t_nodes.unsqueeze(2) # shape: (b, num_rounds, 1, num_rounds, 512)
        # .repeat(1,1,n_objects, 1, 1) 
        s_b_v = torch.cat((t_nodes_t2v.expand(-1,-1,n_objects,-1,-1), updated_b_v), -1)
        s_b_v = self.w_t2v_6(s_b_v) # shape: (b, num_rounds, n_objects, num_rounds, 512)
        proj_q_s_b_v = ques_embed_t2v * s_b_v # shape: (b, num_rounds, n_objects, num_rounds, 512)
        d_t2v = torch.softmax(self.w_t2v_c(proj_q_s_b_v), -2) # δ_ij

        updated_v_nodes = torch.sum(d_t2v*t_nodes_t2v,-2) # shape: (b, num_rounds, n_objects, 512)


        # ==========================================================================================
        # Text Knowledge Storage - Cross Bridge
        # Construct V2T edges
        # B_s_ij = [s_i, v_j], shape = (b, num_rounds, num_rounds, n_objects, 2560])
        b_s = self.constructCrossGraphEdges(t_nodes, v_nodes, n_objects, num_rounds)
        b_s = self.w_v2t_5(b_s) # shape: (b*num_rounds, num_rounds, n_objects, 512)
        ques_embed_v2t = self.w_v2t_4(ques_embed)
        # convert question from shape (b, num_rounds, 512) to (b, num_rounds, 1, 1, 512)
        ques_embed_v2t=ques_embed_v2t.view(batch_size,num_rounds, 1, 1, self.config["lstm_hidden_size"])
        # ques_embed_v2t.shape = (b, num_rounds, 1, 1, 512)
        prod_v2t = ques_embed_v2t * b_s #shape = (b, num_rounds, num_rounds, n_objects, 512)
        prod_v2t = self.w_v2t_b(prod_v2t)
        updated_bridge_v2t = torch.softmax(prod_v2t, -2) # γ_ij
        updated_b_s = updated_bridge_v2t * b_s # (6) shape: (b, num_rounds, num_rounds, n_objects, 512)

        # Query-Guided Cross Graph Convolution
        v_nodes_v2t = v_nodes.unsqueeze(2) # shape: (b, num_rounds, 1, n_objects, 2048)
        # .repeat(1,1,num_rounds, 1, 1)
        v_b_s = torch.cat((v_nodes_v2t.expand(-1,-1, num_rounds, -1, -1), updated_b_s), -1)
        v_b_s = self.w_v2t_6(v_b_s) # shape: (b, num_rounds, num_rounds, n_objects, 512)
        proj_q_v_b_s = ques_embed_v2t * v_b_s # shape: (b, num_rounds, num_rounds, n_objects, 512)
        d_v2t = torch.softmax(self.w_v2t_c(proj_q_v_b_s), -2) # δ_ij
        updated_t_nodes = torch.sum(d_v2t*v_nodes_v2t,-2) # shape: (b, num_rounds, num_rounds, 2048)
        # ==========================================================================================
        # ==========================================================================================
        # Vision Knowledge Storage - Storage
        # Local Knowledge Storage
        concated_v = torch.cat((v_nodes, updated_v_nodes), -1) # shape: (b, num_rounds, n_objects, 2560)
        gate_t2v = torch.sigmoid(self.w_t2v_l(concated_v))
        local_v = self.w_t2v_7(gate_t2v * concated_v) # shape: (b, num_rounds, n_objects, 512)

        # Global Knowledge Storage
        # (11), (12)
        ques_embed = ques_embed.unsqueeze(2) # shape: (b, num_rounds, 1, 512)
        # .repeat(1,1, n_objects, 1) # shape: (b, num_rounds, n_objects, 512)
        v_8 = self.w_t2v_8(v_nodes) # shape: (b, num_rounds, n_objects, 512)
        h_v = torch.softmax(self.w_t2v_e(ques_embed * v_8),-2)
        I_o = torch.sum(h_v * v_nodes, -2) # shape: (b, num_rounds, 2048)
        # (13), (14)
        v_9 = self.w_t2v_9(local_v) # shape: (b, num_rounds, n_objects, 512)
        m_v = torch.softmax(self.w_t2v_a(ques_embed * v_9),-2)
        I_c = torch.sum(m_v * local_v, -2) # shape: (b, num_rounds, 512)
        # (15), (16)
        gate_v_g = torch.sigmoid(self.w_t2v_lg(torch.cat((I_o, I_c), -1)))
        I = self.w_t2v_num_rounds(gate_v_g * torch.cat((I_o, I_c), -1)) # shape: (b, num_rounds, 512)
        

        # Text Knowledge Storage - Storage
        # Local Knowledge Storage
        concated_t = torch.cat((t_nodes, updated_t_nodes), -1)        
        gate_v2t = torch.sigmoid(self.w_v2t_l(concated_t))
        local_t = self.w_v2t_7(gate_v2t * concated_t) # shape: (b, num_rounds, num_rounds, 512)
        # (11), (12)
        # Global Knowledge Storage
        # q_v2t = ques_embed.unsqueeze(2).repeat(1,1, num_rounds, 1)# shape: (b*num_rounds, num_rounds, 512)
        t_8 = self.w_v2t_8(t_nodes)
        h_t = torch.softmax(self.w_v2t_e(ques_embed * t_8),-2)
        H_o = torch.sum(h_t * t_nodes, -2) # shape: (b, num_rounds, 512)
        # (13), (14)
        t_9 = self.w_v2t_9(local_t) # shape: (b, num_rounds, num_rounds, 512)
        m_t = torch.softmax(self.w_v2t_a(ques_embed * t_9),-2)
        H_c = torch.sum(m_t * local_t, -2) # shape: (b, num_rounds, 512)
        # (15), (16)
        gate_t_g = torch.sigmoid(self.w_v2t_lg(torch.cat((H_o, H_c), -1)))
        H = self.w_v2t_num_rounds(gate_t_g * torch.cat((H_o, H_c), -1)) # shape: (b, num_rounds, 512)

        return (I, H)

        
    
    def constructCrossGraphEdges(self, center_nodes, cross_nodes, n_cross, n_center):
        """
        Function that creates a graph with a center node, connected to all cross nodes.

        Parameters
        ----------
        center_nodes: Tensor, shape: (b, num_rounds, n_objects (num_rounds), emd_dim)
            Contains the node that will be the center node for each graph.
        
        cross_nodes: Tensor, shape: (b, num_rounds, num_rounds (n_objects), emd_dim)
            Contains the nodes that will be connected to the center node for each graph.
        """
        # Center node for each graph will be connected to every cross node
        center_nodes = center_nodes.unsqueeze(-2)
        # mask that indicates where center nodes are NOT empty
        mask1 = center_nodes.abs().sum(dim=-1).bool()
        cross_nodes = cross_nodes.unsqueeze(2)
        # mask that indicates where cross nodes are NOT empty
        mask2 = cross_nodes.abs().sum(dim=-1).bool()
        mask = mask1 & mask2
        
        # empty every center node that will be connected to empty cross nodes
        # center_nodes[~mask2] = torch.zeros((center_nodes.shape[-1]), device=center_nodes.device)
        # empty every cross node that will be connected to an empty center node
        # concatenate so that the edge of the cross graph will be
        # either B_v_ij = [v_i, s_j] or B_s_ij = [s_i, v_j]
        bridge_edges = torch.cat((center_nodes.expand(-1, -1, -1, n_cross, -1), cross_nodes.expand(-1, -1, n_center, -1, -1)), -1)
        bridge_edges[~mask] = torch.zeros((bridge_edges.shape[-1]), device=bridge_edges.device)
        # bridge_edges = bridge_edges.masked_fill_(~mask, bridge_edges.shape[-1])
        # shape = (b, num_rounds, num_rounds, n_objects, lstm_hidden + image_feature_dim])
        return bridge_edges