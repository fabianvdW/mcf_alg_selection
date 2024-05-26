import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import skopt
from skopt.utils import use_named_args
import time
from constants import NUM_CLASSES
from gin import GIN


class Objective:
    def __init__(self, dataset, seed, device, **kwargs):
        self.dataset = dataset
        self.seed = seed
        self.device = device
        self.kwargs=kwargs

    def train(self, train_loader, eval_loader, lr, weight_decay, epochs, step_size):
        self.model.train()
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=10**lr, weight_decay=10**weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=0.5, step_size=int(step_size * epochs) if int(step_size * epochs) >= 1 else 1
        )
        for epoch in range(epochs):
            samples_per_class = [0 for _ in range(NUM_CLASSES)]
            correct_per_class = [0 for _ in range(NUM_CLASSES)]
            runtime_sum = 0.0
            minruntime_sum = 0.0
            total_loss = 0
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index.type(torch.int64), data.edge_attr, data.batch)
                pred = out.argmax(dim=-1)
                for i in range(len(data.y)):
                    samples_per_class[data.y[i]] += 1
                    correct_per_class[data.y[i]] += int(pred[i] == data.y[i])
                    runtime_sum += data.label[i, pred[i]]
                    minruntime_sum += min(data.label[i])
                # weights = 1/torch.min(data.label, dim=1)[0][:, None] * data.label
                # weights = torch.nan_to_num(input=weights, nan=1.0, posinf=1000)
                # loss = (weights * F.softmax(out, dim=1)).sum() / data.num_graphs
                loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * data.num_graphs
            scheduler.step()
            acc_per_class = " ".join(
                [
                    f"{correct}/{num_samples}={1 if num_samples == 0 else correct / num_samples:.2f}"
                    for (num_samples, correct) in zip(samples_per_class, correct_per_class)
                ]
            )
            total_acc = sum(correct_per_class) / sum(samples_per_class)
            print(
                f"Training in epoch {epoch}: Total Accuracy: {total_acc:.2f}, Accuracy per class: {acc_per_class}, Loss: {total_loss / len(self.train_loader.dataset)}"
            )
            print(
                f"Training in epoch {epoch}: Total pred runtimes: {runtime_sum} vs total true runtimes {minruntime_sum} (Ratio: {runtime_sum / minruntime_sum:.2f})"
            )
            self.eval(eval_loader, epoch)
            print(epoch, end="\r")
        print(epochs, end=" ")

    def eval(self, eval_loader, epoch):
        self.model.eval()

        samples_per_class = [0 for _ in range(NUM_CLASSES)]
        correct_per_class = [0 for _ in range(NUM_CLASSES)]
        runtime_sum = 0.0
        minruntime_sum = 0.0
        with torch.no_grad():
            for data in eval_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index.type(torch.int64), data.edge_attr, data.batch)
                pred = out.argmax(dim=-1)
                for i in range(len(data.y)):
                    samples_per_class[data.y[i]] += 1
                    correct_per_class[data.y[i]] += int(pred[i] == data.y[i])
                    runtime_sum += data.label[i, pred[i]]
                    minruntime_sum += min(data.label[i])
        acc_per_class = " ".join(
            [
                f"{correct}/{num_samples}={1 if num_samples == 0 else correct / num_samples:.2f}"
                for (num_samples, correct) in zip(samples_per_class, correct_per_class)
            ]
        )
        total_acc = sum(correct_per_class) / sum(samples_per_class)
        print(f"Testing in epoch {epoch}: Total Accuracy: {total_acc:.2f}, Accuracy per class: {acc_per_class}")
        print(
            f"Testing in epoch {epoch}: Total pred runtimes: {runtime_sum} vs total true runtimes {minruntime_sum} (Ratio: {runtime_sum / minruntime_sum:.2f})"
        )
        return -runtime_sum / minruntime_sum + total_acc

    def train_eval(self, train_loader, eval_loader, total_kwargs):
        self.train(train_loader, eval_loader, **total_kwargs)
        return self.eval(eval_loader, total_kwargs["epochs"])

    def __call__(self, **kwargs):
        total_kwargs = kwargs | self.kwargs
        self.model = GIN(
            in_channels=1,
            hidden_channels=total_kwargs["hidden_channels"],
            out_channels=NUM_CLASSES,
            num_gin_layers=total_kwargs["num_gin_layers"],
            num_mlp_layers=total_kwargs["num_mlp_layers"],
            num_mlp_readout_layers=total_kwargs["num_mlp_readout_layers"],
            skip_connections=total_kwargs["skip_connections"],
            train_eps=total_kwargs["train_eps"],
        )
        start = time.time()
        objective_values = []
        kf = KFold(n_splits=5, random_state=self.seed)
        gen = kf.split(list(range(len(self.dataset))))
        for (train_indices, eval_indices) in gen:
            train_loader = DataLoader(self.dataset[train_indices], batch_size=total_kwargs["batch_size"], num_workers=total_kwargs["num_workers"])
            eval_loader = DataLoader(self.dataset[eval_indices], batch_size=total_kwargs["batch_size"], num_workers=total_kwargs["num_workers"])
            objective_values.append(self.train_eval(train_loader, eval_loader, total_kwargs))
            print(f"Finished split with value {objective_values[-1]}")
        objective = np.mean(objective_values)
        end = time.time()
        calc_factor = 0
        # calc_factor += 1 - (self.kwargs['batch_size'] / 256)
        # calc_factor += self.kwargs['epochs'] / 512
        # calc_factor += kwargs['features'] / 128
        # calc_factor += kwargs['layers'] / 10
        # calc_factor += kwargs['ensembling'] / 64
        # calc_factor /= 100
        print(-objective + calc_factor, end=" ")
        print(end - start, "s")
        return -objective + calc_factor


def optimize(dataset, device, search_space, num_bayes_samples, seed, **kwargs):
    obj = Objective(dataset, seed, device, **kwargs )

    @use_named_args(dimensions=search_space)
    def objective(**kwargs):
        return obj(**kwargs)

    return skopt.gp_minimize(
        objective,
        search_space,
        n_calls=num_bayes_samples,
        random_state=seed,
        n_initial_points=int(num_bayes_samples / 5) if num_bayes_samples / 5 >= 1 else 1,
    )
