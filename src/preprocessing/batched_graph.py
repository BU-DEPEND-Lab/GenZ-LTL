import functools

import torch
from torch_geometric.data import Data


class BatchedGraph:
    def __init__(self, graphs: list[Data], device=None):
        self.batch_x, self.batch_edge_index, self.root_node_mask = self.batch(graphs, device)
        self.batch_num_nodes = torch.tensor([tg.num_nodes for tg in graphs], dtype=torch.long).to(device)

    @staticmethod
    def batch(graphs: list[Data], device=None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        max_nodes = max(tg.num_nodes for tg in graphs)
        max_edges = max(tg.num_edges for tg in graphs)
        feature_dim = graphs[0].x.shape[1]
        batch_size = len(graphs)
        x = torch.zeros((batch_size, max_nodes, feature_dim), dtype=torch.float).to(device)
        edge_index = torch.zeros((batch_size, 2, max_edges), dtype=torch.long).to(device)
        root_node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool).to(device)
        for i, g in enumerate(graphs):
            x[i, :g.num_nodes, :] = g.x
            edge_index[i, :, :g.num_edges] = g.edge_index
            edge_index[i, :, g.num_edges:] = g.num_nodes  # padding
            root_node_mask[i, 0] = True
        return x, edge_index, root_node_mask

    def __getitem__(self, index) -> Data:
        """Returns a sub-batch of the given graphs."""
        x = self.batch_x[index].reshape(-1, self.batch_x.shape[2])
        root_node_mask = self.root_node_mask[index].reshape(-1)
        edge_index = self.batch_edge_index[index]
        # Shift the edges of subsequent graphs in the batch
        cumsum = self.batch_num_nodes[index].cumsum(0) - self.batch_num_nodes[index[0]]
        edge_index += cumsum.reshape(-1, 1, 1)
        edge_index = edge_index.transpose(0, 1).reshape(2, -1).contiguous()
        data = Data(
            x=x,
            edge_index=edge_index,
            num_graphs=len(index),
            root_node_mask=root_node_mask
        )
        assert data.validate()
        return data

    @functools.cached_property
    def all(self):
        return self[range(self.batch_x.shape[0])]
