import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric
from torch_loader import MCFDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GINEConv, global_add_pool

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

path = osp.dirname(osp.realpath(__file__))
path = osp.join(path, '..', 'data', 'generated_data')
dataset = MCFDataset(path).shuffle()

BATCH_SIZE = 4
train_loader = DataLoader(dataset[:0.9], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset[0.9:], batch_size=BATCH_SIZE)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINEConv(nn=mlp, train_eps=False, edge_dim=2))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr).relu()
        # Pass the batch size to avoid CPU communication/graph breaks:
        x = global_add_pool(x, batch, size=batch_size)
        return self.mlp(x)


model = GIN(
    in_channels=1,
    hidden_channels=32,
    out_channels=4,
    num_layers=5,
).to(device)

# Compile the model into an optimized version:
#model = torch.compile(model, dynamic=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.batch_size)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.batch_size)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


times = []
for epoch in range(1, 101):
    start = time.time()
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    times.append(time.time() - start)
    print(times[-1])
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Test: {test_acc:.4f}')
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
