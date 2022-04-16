from torch import nn

class RelationalGraphConvolutionalNetwork(nn.Module):

    def __init__(self, config):
            super(RelationalGraphConvolutionalNetwork, self).__init__()
            self.config = config

            self.num_relations = self.config['num_relations']
            self.num_nodes = self.config['max_nodes']

    def forward(self):
        raise NotImplemented