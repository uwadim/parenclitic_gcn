"""
Models
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import BatchNorm1d, Linear, Module
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import ASAPooling, SAGPooling, GATConv, TAGConv


class GCN(Module):
    # The Class is derived from GCN
    def __init__(self, config, num_node_features):
        hidden_channels = config.model.embedding_size
        # Default values for embeddings
        self.embeddings = np.ones(2*hidden_channels)
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.bn1 = BatchNorm1d(num_features=hidden_channels)
        self.lin = Linear(in_features=hidden_channels, out_features=1)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        # Global Pooling (stack different aggregations)
        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = gmp(x, batch)
        # Batch norm for embeddings
        x = self.bn1(x)
        self.embeddings = x.detach().to('cpu').numpy()

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    def get_embeddings(self):
        return self.embeddings


class SkipGCN(Module):
    # GCN with skip connections
    def __init__(self, config, num_node_features):
        hidden_channels = config.model.embedding_size
        # Default values for embeddings
        self.embeddings = np.ones(2 * hidden_channels)
        super(SkipGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels, normalize=False)
        # skip connection: hidden_channels + one for residuals
        self.conv2 = GCNConv(hidden_channels + 1, hidden_channels + 1, normalize=False)
        # skip connection: hidden_channels + hidden_channels + residuals
        self.conv3 = GCNConv(2 * hidden_channels + 1, 2 * hidden_channels + 1, normalize=False)
        self.bn1 = BatchNorm1d(num_features=2 * hidden_channels + 1)
        self.lin = Linear(in_features=2 * hidden_channels + 1, out_features=1)

    def forward(self, x, edge_index, edge_weight, batch):
        residual_0 = x
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        residual_1 = x
        x = torch.cat([x, residual_0], dim=1)
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = torch.cat([x, residual_1], dim=1)
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        # Global Pooling (stack different aggregations)
        # x = torch.cat([gap(x, batch), gap(x, batch)], dim=1)
        x = gap(x, batch)
        # Batch norm for embeddings
        x = self.bn1(x)
        self.embeddings = x.detach().to('cpu').numpy()

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    def get_embeddings(self):
        return self.embeddings
