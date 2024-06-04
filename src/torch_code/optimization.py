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
import random

class Objective:
    def __init__(self, dataset, seed, device, num_workers):
        self.dataset = dataset
        self.seed = seed
        self.device = device
        self.num_workers=num_workers

    def train(self, train_loader, eval_loader, lr, weight_decay, epochs, step_size):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=10**lr, weight_decay=10**weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=0.5, step_size=int(step_size * epochs) if int(step_size * epochs) >= 1 else 1
        )
        for epoch in range(epochs):
            samples_per_class = [0 for _ in range(NUM_CLASSES)]
            correct_per_class = [0 for _ in range(NUM_CLASSES)]
            runtime_sum = 0.0
            minruntime_sum = 0.0
            total_loss = 0
            start = time.time()
            for data in train_loader:
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
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
                f"Training in epoch {epoch}: Total Accuracy: {total_acc:.2f}, Accuracy per class: {acc_per_class}, Loss: {total_loss / len(train_loader.dataset)}"
            )
            print(
                f"Training in epoch {epoch}: Total pred runtimes: {runtime_sum} vs total true runtimes {minruntime_sum} (Ratio: {runtime_sum / minruntime_sum:.2f})"
            )
            print(f"Training in epoch {epoch}, time: {time.time() - start}")
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
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
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

    #TODO:  LOG-Runtimes loss, Runtime loss rescale runtimes
    def train_eval(self, train_loader, eval_loader, total_kwargs):
        self.train(train_loader, eval_loader, total_kwargs["lr"], total_kwargs["weight_decay"], total_kwargs["epochs"], total_kwargs["step_size"])
        return self.eval(eval_loader, total_kwargs["epochs"])

    def __call__(self, **kwargs):
        print(kwargs)
        self.model = GIN(
            device=self.device,
            in_channels=1,
            hidden_channels=kwargs["hidden_channels"],
            out_channels=NUM_CLASSES,
            num_gin_layers=kwargs["num_gin_layers"],
            num_mlp_layers=kwargs["num_mlp_layers"],
            num_mlp_readout_layers=kwargs["num_mlp_readout_layers"],
            skip_connections=kwargs["skip_connections"],
            vpa=kwargs["vpa"]
        ).to(self.device)

        start = time.time()
        objective_values = []
        kf = KFold(n_splits=5, random_state=self.seed, shuffle=True)
        gen = kf.split(list(range(len(self.dataset))))
        for (train_indices, eval_indices) in gen:
            train_loader = DataLoader(self.dataset[list(train_indices)], batch_size=int(kwargs["batch_size"]))
            eval_loader = DataLoader(self.dataset[list(eval_indices)], batch_size=int(kwargs["batch_size"]))
            objective_values.append(self.train_eval(train_loader, eval_loader, kwargs).item())
            print(f"Finished split with value {objective_values[-1]}")
        objective = np.mean(objective_values)
        end = time.time()
        calc_factor = 0
        print(-objective + calc_factor, end=" ")
        print(end - start, "s")
        return -objective + calc_factor


def optimize(dataset, device, search_space, num_bayes_samples, num_workers, seed):
    obj = Objective(dataset, seed, device, num_workers)

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
