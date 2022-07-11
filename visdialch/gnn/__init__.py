from visdialch.gnn.gcn import GraphConvolution
from visdialch.gnn.gat import GraphAttentionNetwork
from visdialch.gnn.rgcn import RelationalGraphConvolution
from visdialch.gnn.message_passing import MessagePassing

def GNN(model_config, *args):
    name_gnn_map = {
        'gcn': GraphConvolution,
        'gat': GraphAttentionNetwork,
        'rgcn':  RelationalGraphConvolution,
        'message_passing': MessagePassing
    }
    return name_gnn_map[model_config["gnn"]](model_config, *args)