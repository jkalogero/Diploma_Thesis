import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Dataset, Data
import numpy as np
import os
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
from visdialch.data.readers import ImageFeaturesHdfReader, DialogsReader

class VisDialDataset(Dataset):
    def __init__(self,
                 config: Dict[str, Any],
                 dialogs_jsonpath: str,
                 root,
                 filename,
                 test=False,
                 transform=None,
                 pre_transform=None,
                 in_memory: bool = False):
        self.test = test
        self.filename = filename
        self.config = config
        self.dialogs_reader = DialogsReader(dialogs_jsonpath)

        # Initialize image features reader according to split.
#         image_features_hdfpath = config["image_features_train_h5"]
        image_features_hdfpath = '/home/jkalogero/KBGN-Implementation/data/mySubmat.h5'
        if "val" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_val_h5"]
        elif "test" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_test_h5"]
        
        self.hdf_reader = ImageFeaturesHdfReader(image_features_hdfpath, in_memory)

        super(VisDialDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return self.filename
    
    @property
    def processed_file_names(self):
        self.image_ids, self.features = self.hdf_reader.get_node_features()
        if self.test:
            return [f'data_test_{torch.IntTensor.item(i)}.pt' for i in list(self.image_ids)]
        else:
            return [f'data_{torch.IntTensor.item(i)}.pt' for i in list(self.image_ids)]
    
    def download(self):
        pass

    def process(self):
        self.image_ids, self.features = self.hdf_reader.get_node_features()
        for index, graph in tqdm(enumerate(self.features)):
            # construct graph
            # Get node features
            node_features = graph
            # Get edge features
            edge_features = self._get_edge_features(self.features.shape[1])
            # Get adjacency info
            edge_index = self._get_adjacency_info(self.features.shape[1])

            # Create data object
            data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        graph_id = self.image_ids[index])
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))


    def _get_node_features(self, image_ids, features):
        pass

    def _get_edge_features(self, num_of_nodes=36, edge_feature_size = 512):
        """
        For now it generates a random matrix of the shape 
        [num_of_edges, edge_feature_size = 512]
        """
        num_of_edges = num_of_nodes*(num_of_nodes - 1)
        edge_features = torch.rand([num_of_edges, edge_feature_size])
        
        return edge_features
    
    def _get_adjacency_info(self, num_of_nodes=36):
        """
        Return an edge list of a fully connected graph with the 
        given number of nodes
        """
        # Initialize edge index matrix
        E = torch.zeros((2, num_of_nodes * (num_of_nodes - 1)), dtype=torch.long)

        # Populate 1st row
        for node in range(num_of_nodes):
            for neighbor in range(num_of_nodes - 1):
                E[0, node * (num_of_nodes - 1) + neighbor] = node

        # Populate 2nd row
        neighbors = []
        for node in range(num_of_nodes):
            neighbors.append(list(np.arange(node)) + list(np.arange(node+1, num_of_nodes)))
        E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])

        return E
    
    
    def len(self):
        return self.features.shape[1]
    
    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data