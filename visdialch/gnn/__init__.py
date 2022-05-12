from visdialch.gnn.gcn import GraphConvolution
from visdialch.gnn.gat import GAT
from visdialch.gnn.rgcn import RelationalGraphConvolutionalNetwork

def GNN(model_config, *args):
    name_gnn_map = {
        'gcn': GraphConvolution,
        'gat': GAT,
        'rgcn':  RelationalGraphConvolutionalNetwork
    }
    return name_gnn_map[model_config["gnn"]](model_config, *args)