import torch
from torch import nn
from torch.nn.functional import softmax


class KnowledgeStorage(nn.Module):
    def __init__(self, config):
        super(KnowledgeStorage, self).__init__()
        self.config = config

        self.dropout = nn.Dropout(p=config["dropout"])

        # Query-Guided Bridge Update
        self.w_t2v = nn.ModuleDict({
            '4': nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"]),
            '5': nn.Linear(config["img_feature_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"]),
            'b': nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"]),
            '6': nn.Linear(2*config["lstm_hidden_size"], config["lstm_hidden_size"]),
            'c': nn.Linear(config["lstm_hidden_size"], 1),
            'l': nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"] + config["img_feature_size"]),
            '7': nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"]),
            'e': nn.Linear(config["lstm_hidden_size"], 1),
            '8': nn.Linear(config["img_feature_size"], config["lstm_hidden_size"]),
            'a': nn.Linear(config["lstm_hidden_size"], 1),
            '9': nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"]),
            'lg': nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"] + config["img_feature_size"]),
            'num_rounds': nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"])
        })
        self.w_v2t = nn.ModuleDict({
            '4': nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"]),
            '5': nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"]),
            'b': nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"]),
            '6': nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"]),
            'c': nn.Linear(config["lstm_hidden_size"], 1),
            'l': nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"] + config["img_feature_size"]),
            '7': nn.Linear(config["lstm_hidden_size"] + config["img_feature_size"], config["lstm_hidden_size"]),
            'e': nn.Linear(config["lstm_hidden_size"], 1),
            '8': nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"]),
            'a': nn.Linear(config["lstm_hidden_size"], 1),
            '9': nn.Linear(config["lstm_hidden_size"], config["lstm_hidden_size"]),
            'lg': nn.Linear(config["lstm_hidden_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"] + config["lstm_hidden_size"]),
            'num_rounds': nn.Linear(config["lstm_hidden_size"] + config["lstm_hidden_size"], config["lstm_hidden_size"])
        })
        
        
    def forward(self, v_nodes, t_nodes, ques_embed, batch_size, num_rounds):
        
        n_objects = v_nodes.shape[2]
        # Vision Knowledge Storage - Cross Bridge
        updated_v_nodes = self.CrossBridge(
            ques_embed, batch_size, v_nodes, n_objects, t_nodes, num_rounds, self.w_t2v, num_rounds)

        # Text Knowledge Storage - Cross Bridge
        updated_t_nodes = self.CrossBridge(
            ques_embed, batch_size, t_nodes, num_rounds, v_nodes, n_objects, self.w_v2t, num_rounds)
        
        # Vision Knowledge Storage - Storage
        I = self.Storage(ques_embed, v_nodes, updated_v_nodes, self.w_t2v)
        # Text Knowledge Storage - Storage
        H = self.Storage(ques_embed, t_nodes, updated_t_nodes, self.w_v2t)
        
        return (I, H)

    
    def CrossBridge(self, question, batch_size, center_nodes, center_nodes_dim, cross_nodes, cross_nodes_dim, weights, num_rounds):
        """
        Cross Bridge consists of: 
        1. Query-Guided Bridge Update, which updates the edges between center and cross nodes.
        2. Query-Guided Cross Graph Convolution, which performs a GCN between the cross graph 
        and each center node.

        Parameters:
        ===========
        question:
            The embedding of the question, shape: (b, num_rounds, emb_dim)
        batch_size:
            The batch size
        center_nodes: 
            The nodes of the modality to be enriched with cross nodes
        center_nodes_dim: int
            Dimension length of center nodes (n_objects/n_rounds)
        cross_nodes: 
            The nodes of the modality that will enrich center nodes
        cross_nodes_dim: int
            Dimension length of cross nodes (n_rounds/n_objects)
        weights: nn.ModuleDict
            Dictionary with all the weights (t2v/vt2)
        """
        # Query-Guided Bridge Update
        # B_ij = [center_i, cross_j], shape = (b, cross_nodes_dim, center_nodes_dim, cross_nodes_dim, 2560)
        b = self.constructCrossGraphEdges(center_nodes, cross_nodes, cross_nodes_dim, center_nodes_dim)
        b = weights['5'](b) # shape: (b,cross_nodes_dim, center_nodes_dim, cross_nodes_dim, 512)
        question_emb = weights['4'](question)
        # convert question from shape (b,num_rounds, 512) to (b,num_rounds, 1, 1, 512)
        question_emb=question_emb.view(batch_size,num_rounds, 1, 1, self.config["lstm_hidden_size"])
        # .repeat(1,1,center_nodes_dim,cross_nodes_dim,1)

        product = question_emb * b
        product = weights['b'](product)
        updated_bridge = torch.softmax(product, -2) # γ_ij
        updated_b = updated_bridge * b # (6) shape: (b,cross_nodes_dim, center_nodes_dim, cross_nodes_dim, 512)

        # Query-Guided Cross Graph Convolution
        cross_nodes = cross_nodes.unsqueeze(2) # shape: (b, cross_nodes_dim, 1, cross_nodes_dim, 512)
        # .repeat(1,1,center_nodes_dim, 1, 1) 
        s_b = torch.cat((cross_nodes.expand(-1,-1,center_nodes_dim,-1,-1), updated_b), -1)
        s_b = weights['6'](s_b) # shape: (b, cross_nodes_dim, center_nodes_dim, cross_nodes_dim, 512)
        proj_q_s_b = question_emb * s_b # shape: (b, cross_nodes_dim, center_nodes_dim, cross_nodes_dim, 512)
        d = torch.softmax(weights['c'](proj_q_s_b), -2) # δ_ij

        updated_center_nodes = torch.sum(d*cross_nodes,-2) # shape: (b, cross_nodes_dim, center_nodes_dim, 512)
        return updated_center_nodes

    
    def Storage(self, question, nodes, updated_nodes, weights):
        # Local Knowledge Storage
        concated_v = torch.cat((nodes, updated_nodes), -1) # shape: (b, num_rounds, n_objects, 2560)
        gate_t2v = torch.sigmoid(weights['l'](concated_v))
        local_v = weights['7'](gate_t2v * concated_v) # shape: (b, num_rounds, n_objects, 512)

        # Global Knowledge Storage
        # (11), (12)
        ques_embed = question.unsqueeze(2) # shape: (b, num_rounds, 1, 512)
        # .repeat(1,1, n_objects, 1) # shape: (b, num_rounds, n_objects, 512)
        v_8 = weights['8'](nodes) # shape: (b, num_rounds, n_objects, 512)
        h_v = torch.softmax(weights['e'](ques_embed * v_8),-2)
        K_o = torch.sum(h_v * nodes, -2) # shape: (b, num_rounds, 2048)
        # (13), (14)
        v_9 = weights['9'](local_v) # shape: (b, num_rounds, n_objects, 512)
        m_v = torch.softmax(weights['a'](ques_embed * v_9),-2)
        K_c = torch.sum(m_v * local_v, -2) # shape: (b, num_rounds, 512)

        # (15), (16)
        gate_v_g = torch.sigmoid(weights['lg'](torch.cat((K_o, K_c), -1)))
        global_knowledge = weights['num_rounds'](gate_v_g * torch.cat((K_o, K_c), -1)) # shape: (b, num_rounds, 512)

        return global_knowledge

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