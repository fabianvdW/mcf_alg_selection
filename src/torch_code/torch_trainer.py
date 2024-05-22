import os.path as osp
import time

import torch
import torch.nn.functional as F
from gin import GIN
from torch_in_memory_loader import MCFDataset
import torch_geometric
from torch_geometric.loader import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

path = osp.dirname(osp.realpath(__file__))
path = osp.join(path, "../..", "data", "generated_data")
dataset = MCFDataset(path).shuffle()

NUM_CLASSES = 4
BATCH_SIZE = 5
train_loader = DataLoader(dataset[:0.8], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset[0.8:], batch_size=BATCH_SIZE)


model = GIN(
    in_channels=1,
    hidden_channels=32,
    out_channels=NUM_CLASSES,
    num_gin_layers=5,
    num_mlp_layers=1,
    num_mlp_readout_layers=1,
    skip_connections=True,
    train_eps=False
).to(device)
# Compile the model into an optimized version:
# model = torch_code.compile(model, dynamic=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(loader, epoch):
    # TODO: Random weighted sampling for rebalancing dataset / Think about loss again(Cross Entropy, Mean Runtime or GumbelSoftmax)

    model.train()
    samples_per_class = [0 for _ in range(NUM_CLASSES)]
    correct_per_class = [0 for _ in range(NUM_CLASSES)]
    runtime_sum = 0.0
    minruntime_sum = 0.0
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index.type(torch.int64), data.edge_attr, data.batch)
        pred = out.argmax(dim=-1)
        for i in range(len(data.y)):
            samples_per_class[data.y[i]] += 1
            correct_per_class[data.y[i]] += int(pred[i] == data.y[i])
            runtime_sum += data.label[i, pred[i]]
            minruntime_sum += min(data.label[i])

        #weights = 1/torch.min(data.label, dim=1)[0][:, None] * data.label
        #weights = torch.nan_to_num(input=weights, nan=1.0, posinf=1000)
        #loss = (weights * F.softmax(out, dim=1)).sum() / data.num_graphs
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    acc_per_class = " ".join(
        [
            f"{correct}/{num_samples}={1 if num_samples == 0 else correct / num_samples:.2f}"
            for (num_samples, correct) in zip(samples_per_class, correct_per_class)
        ]
    )
    total_acc = sum(correct_per_class) / sum(samples_per_class)
    print(
        f"Training in epoch {epoch}: Total Accuracy: {total_acc:.2f}, Accuracy per class: {acc_per_class}, Loss: {total_loss / len(train_loader.dataset)}"
    )
    print(
        f"Training in epoch {epoch}: Total pred runtimes: {runtime_sum} vs total true runtimes {minruntime_sum} (Ratio: {runtime_sum / minruntime_sum:.2f})"
    )

    return total_loss / len(train_loader.dataset), runtime_sum / minruntime_sum, total_acc


@torch.no_grad()
def eval(loader, epoch):
    model.eval()
    samples_per_class = [0 for _ in range(NUM_CLASSES)]
    correct_per_class = [0 for _ in range(NUM_CLASSES)]
    runtime_sum = 0.0
    minruntime_sum = 0.0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index.type(torch.int64), data.edge_attr, data.batch)
        pred = out.argmax(dim=-1)
        for i in range(len(data.y)):
            samples_per_class[data.y[i]] += 1
            correct_per_class[data.y[i]] += int(pred[i] == data.y[i])
            runtime_sum += data.label[i, pred[i]]
            minruntime_sum += min(data.label[i])
    acc_per_class = " ".join(
        [
            f"{correct}/{num_samples}={1 if num_samples == 0 else correct/num_samples:.2f}"
            for (num_samples, correct) in zip(samples_per_class, correct_per_class)
        ]
    )
    total_acc = sum(correct_per_class) / sum(samples_per_class)
    print(f"Testing in epoch {epoch}: Total Accuracy: {total_acc:.2f}, Accuracy per class: {acc_per_class}")
    print(
        f"Testing in epoch {epoch}: Total pred runtimes: {runtime_sum} vs total true runtimes {minruntime_sum} (Ratio: {runtime_sum/minruntime_sum:.2f})"
    )
    return runtime_sum / minruntime_sum, total_acc


if __name__ == "__main__":
    times = []
    for epoch in range(1, 101):
        start = time.time()
        _ = train(train_loader, epoch)
        eval(test_loader, epoch)
        times.append(time.time() - start)
        print(f"Epoch {epoch} finished in {times[-1]}s")
