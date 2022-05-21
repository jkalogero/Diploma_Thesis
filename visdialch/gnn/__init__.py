from visdialch.gnn.gcn import GraphConvolution
from visdialch.gnn.gat import GraphAttentionNetwork
from visdialch.gnn.rgcn import RelationalGraphConvolution

def GNN(model_config, *args):
    name_gnn_map = {
        'gcn': GraphConvolution,
        'gat': GraphAttentionNetwork,
        'rgcn':  RelationalGraphConvolution
    }
    return name_gnn_map[model_config["gnn"]](model_config, *args)