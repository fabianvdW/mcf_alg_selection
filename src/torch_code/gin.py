import torch
from torch_geometric.nn import MLP, GINEConv


class GIN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_gin_layers,
        num_mlp_layers,
        num_mlp_readout_layers,
        dropout=0.5,
        norm=None,
        skip_connections=True,
        vpa=True,
        train_eps=True,
    ):
        super().__init__()
        # TODO ? VPA on GINEConv layers? GNN-VPA only has implementation for GINConv layer.
        self.skip_connections = skip_connections
        self.vpa = vpa
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        for i in range(num_gin_layers):
            mlp = MLP(
                [in_channels if i == 0 else hidden_channels] + [hidden_channels for _ in range(num_mlp_layers)] + [hidden_channels],
                norm=norm,
                dropout=dropout,
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=train_eps, edge_dim=2))

        self.linears.append(
            MLP([in_channels] + [hidden_channels for _ in range(num_mlp_readout_layers)] + [out_channels], norm=norm, dropout=dropout)
        )
        for _ in range(num_gin_layers):
            self.linears.append(
                MLP(
                    [hidden_channels] + [hidden_channels for _ in range(num_mlp_readout_layers)] + [out_channels],
                    norm=norm,
                    dropout=dropout,
                )
            )

    def preprocess_graphpool(self, batch):
        # From https://github.com/ml-jku/GNN-VPA/blob/main/src/models/gin.py#L92C9-L92C29
        len_list = torch.bincount(batch)
        idx = []
        elem = []

        start_idx = 0
        for i, graph_len in enumerate(len_list):
            if self.vpa:
                elem.extend([1.0 / torch.sqrt(graph_len)] * graph_len)
            else:
                elem.extend([1] * graph_len)
            idx.extend([[i, j] for j in range(start_idx, start_idx + graph_len, 1)])
            start_idx += graph_len
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)

        graph_pool = torch.sparse_coo_tensor(idx, elem, torch.Size([len(len_list), len(batch)]))
        return graph_pool

    def forward(self, x, edge_index, edge_attr, batch):
        graphpool = self.preprocess_graphpool(batch)  # Used for VPA on readout layers, if enabled

        representations = [x]
        x = self.convs[0](x, edge_index, edge_attr).relu()
        representations += [x]
        for conv in self.convs[1:]:
            if self.skip_connections:
                x = conv(x, edge_index, edge_attr).relu() + x
            else:
                x = conv(x, edge_index, edge_attr).relu()
            representations += [x]
        sum_pool = None
        assert len(representations) == len(self.linears)
        for i in range(len(representations)):
            z = self.linears[i](torch.spmm(graphpool, representations[i]))
            if sum_pool is None:
                sum_pool = z
            else:
                sum_pool += z
        return sum_pool
