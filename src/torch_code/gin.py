import torch
import torch_geometric
from torch_geometric.nn import MLP, GINEConv


class GIN(torch.nn.Module):
    def __init__(
        self,
        device,
        in_channels,
        hidden_channels,
        out_channels,
        num_gin_layers,
        num_mlp_layers,
        num_mlp_readout_layers,
        skip_connections=True
    ):
        super().__init__()
        self.device = device
        self.skip_connections = skip_connections
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        for i in range(num_gin_layers):
            mlp = MLP(
                [in_channels if i == 0 else hidden_channels] + [hidden_channels for _ in range(num_mlp_layers)] + [hidden_channels],
                norm="batch_norm",
                dropout=0.5,
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True, edge_dim=2))

        self.linears.append(
            MLP([in_channels] + [hidden_channels for _ in range(num_mlp_readout_layers)] + [out_channels], norm="batch_norm", dropout=0.5)
        )
        for _ in range(num_gin_layers):
            self.linears.append(
                MLP(
                    [hidden_channels] + [hidden_channels for _ in range(num_mlp_readout_layers)] + [out_channels],
                    norm="batch_norm",
                    dropout=0.5,
                )
            )

    def forward(self, x, edge_index, edge_attr, batch):
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
        for i in range(len(representations)):

            z = self.linears[i](torch_geometric.nn.global_add_pool(representations[i], batch))
            if sum_pool is None:
                sum_pool = z
            else:
                sum_pool += z
        return sum_pool
