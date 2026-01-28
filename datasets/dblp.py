from .dataset import NodeClassificationDataset
import pickle
import numpy as np
import json
import random


class DBLP(NodeClassificationDataset):
    def __init__(self, dataset_path, config):
        super().__init__(dataset_path, config)
        self.load_data()
        self.hetero = True
        if config.get('check_data_size', False):
            self.check_data_size()

        # avoid in train set, some classes are not included
        self.num_classes = max(len(set(i[1] for i in self.labels[0])),
                               len(set(i[1] for i in self.labels[1])),
                               len(set(i[1] for i in self.labels[2])))
        self.num_features = self.node_features.shape[1]

        if self.config.get("test_label_shuffle", False):
            self.labels[2] = np.array(random.sample(np.array(self.labels[2]).tolist(), len(self.labels[2]))).tolist()

    def load_data(self):
        with open(self.node_features_path, 'rb') as f:
            self.node_features = pickle.load(f)

        self.node_types = np.load(self.node_types_path)

        with open(self.edges_path, 'rb') as f:
            self.edges = pickle.load(f)

        if self.config.get('labels', None) is None:
            with open(self.labels_path, 'rb') as f:
                self.labels = pickle.load(f)
        else:
            self.labels = self.config['labels']
            with open(self.labels, 'rb') as f:
                self.labels = pickle.load(f)

        with open(self.edge_directions_path, 'rb') as f:
            self.edge_directions = json.load(f)

    def check_data_size(self):
        """
        Check the size of each data file
        :return:
        """
        num_nodes = self.node_features.shape[0]
        assert num_nodes == len(
            self.node_types), f"Node features and node types have different sizes"
        edge_shapes = [subgraph.shape for subgraph in self.edges]
        for i, shape in enumerate(edge_shapes):
            assert shape[0] == shape[1], f"Subgraph {i} is not square"

        for i, shape in enumerate(edge_shapes):
            assert num_nodes == shape[
                0], f"Node features and subgraph {i} have different sizes"

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        dataset = cls.__new__(cls)
        for key, value in data.items():
            setattr(dataset, key, value)
        return dataset
