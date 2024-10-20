import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import skopt
from skopt.utils import use_named_args
import time
from constants import NUM_CLASSES
from gin import GIN, GINRes


class Objective:
    def __init__(self, dataset, seed, device, num_workers, compile_model):
        self.dataset = dataset
        self.seed = seed
        self.device = device
        self.num_workers = num_workers
        self.compile_model = compile_model
        self.log_info = []

    def train(self, train_loader, eval_loader, lr, weight_decay, epochs, step_size, loss_fn, loss_weight):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=10 ** lr, weight_decay=10 ** weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=0.5, step_size=int(step_size * epochs) if int(step_size * epochs) >= 1 else 1
        )
        epochs_info = []
        for epoch in range(epochs):
            samples_per_class = [0 for _ in range(NUM_CLASSES)]
            correct_per_class = [0 for _ in range(NUM_CLASSES)]
            runtime_sum = 0.0
            minruntime_sum = 0.0
            total_loss = 0
            start = time.time()
            for data in train_loader:
                optimizer.zero_grad()
                #data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch, data.batch_size)
                pred = out.argmax(dim=-1)

                for i in range(len(data.y)):
                    samples_per_class[data.y[i]] += 1
                    correct_per_class[data.y[i]] += int(pred[i] == data.y[i])
                    runtime_sum += data.label[i, pred[i]]
                    minruntime_sum += min(data.label[i])
                if loss_fn == "expected_runtime":
                    loss = torch.sum(F.softmax(out, dim=1) * data.label / 10 ** 5) / data.batch_size
                elif loss_fn == "mix_expected_runtime":
                    er_loss = (torch.sum(F.softmax(out, dim=1) * data.label, dim=1) -torch.min(data.label, dim=1)[0])/ 10 ** 4
                    loss = loss_weight * F.cross_entropy(out, data.y) + (1. - loss_weight) * torch.sum(er_loss) / data.batch_size
                elif loss_fn == "cross_entropy":
                    loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * data.batch_size
            scheduler.step()
            acc_per_class = " ".join(
                [
                    f"{correct}/{num_samples}={1 if num_samples == 0 else correct / num_samples:.2f}"
                    for (num_samples, correct) in zip(samples_per_class, correct_per_class)
                ]
            )
            total_acc = sum(correct_per_class) / sum(samples_per_class)
            total_loss /= len(train_loader.dataset)
            epoch_info = {}
            epoch_info['train_acc_per_class'] = acc_per_class
            epoch_info['train_total_acc'] = total_acc
            epoch_info['train_runtime_sum'] = float(runtime_sum)
            epoch_info['train_minruntime_sum'] = float(minruntime_sum)
            epoch_info['train_total_loss'] = total_loss

            print(
                f"Training in epoch {epoch}: Total Accuracy: {total_acc:.2f}, Accuracy per class: {acc_per_class}, Loss: {total_loss}"
            )
            print(
                f"Training in epoch {epoch}: Total pred runtimes: {runtime_sum} vs total true runtimes {minruntime_sum} (Ratio: {runtime_sum / minruntime_sum:.2f})"
            )
            print(f"Training in epoch {epoch}, time: {time.time() - start}")
            self.eval(eval_loader, epoch, epoch_info, loss_fn, loss_weight)
            print(epoch, end="\r")
            epochs_info.append(epoch_info)
        return epochs_info

    def eval(self, eval_loader, epoch, epoch_info, loss_fn, loss_weight):
        self.model.eval()

        samples_per_class = [0 for _ in range(NUM_CLASSES)]
        correct_per_class = [0 for _ in range(NUM_CLASSES)]
        runtime_sum = 0.0
        minruntime_sum = 0.0
        total_loss = 0.0
        with torch.no_grad():
            for data in eval_loader:
                #data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch, data.batch_size)
                pred = out.argmax(dim=-1)
                for i in range(len(data.y)):
                    samples_per_class[data.y[i]] += 1
                    correct_per_class[data.y[i]] += int(pred[i] == data.y[i])
                    runtime_sum += data.label[i, pred[i]]
                    minruntime_sum += min(data.label[i])
                if loss_fn == "expected_runtime":
                    loss = torch.sum(F.softmax(out, dim=1) * data.label / 10 ** 5) / data.batch_size
                elif loss_fn == "mix_expected_runtime":
                    er_loss = (torch.sum(F.softmax(out, dim=1) * data.label, dim=1) -torch.min(data.label, dim=1)[0])/ 10 ** 4
                    loss = loss_weight * F.cross_entropy(out, data.y) + (1. - loss_weight) * torch.sum(er_loss) / data.batch_size
                elif loss_fn == "cross_entropy":
                    loss = F.cross_entropy(out, data.y)
                total_loss += float(loss) * data.batch_size
        acc_per_class = " ".join(
            [
                f"{correct}/{num_samples}={1 if num_samples == 0 else correct / num_samples:.2f}"
                for (num_samples, correct) in zip(samples_per_class, correct_per_class)
            ]
        )
        total_acc = sum(correct_per_class) / sum(samples_per_class)
        total_loss /= len(eval_loader.dataset)
        epoch_info['eval_acc_per_class'] = acc_per_class
        epoch_info['eval_total_acc'] = total_acc
        epoch_info['eval_runtime_sum'] = float(runtime_sum)
        epoch_info['eval_minruntime_sum'] = float(minruntime_sum)
        epoch_info['eval_total_loss'] = total_loss
        epoch_info['eval_obj'] = float(-runtime_sum / minruntime_sum + total_acc)
        print(
            f"Testing in epoch {epoch}: Total Accuracy: {total_acc:.2f}, Accuracy per class: {acc_per_class}, Loss: {total_loss}")
        print(
            f"Testing in epoch {epoch}: Total pred runtimes: {runtime_sum} vs total true runtimes {minruntime_sum} (Ratio: {runtime_sum / minruntime_sum:.2f})"
        )

    def train_eval(self, train_loader, eval_loader, total_kwargs):
        epochs_info = self.train(train_loader, eval_loader, total_kwargs["lr"], total_kwargs["weight_decay"],
                                 total_kwargs["epochs"], total_kwargs["step_size"], total_kwargs["loss"],
                                 total_kwargs.get("loss_weight", None))
        return epochs_info

    def __call__(self, **kwargs):
        print(kwargs)
        start = time.time()
        objective_values = []
        kf = KFold(n_splits=5, random_state=self.seed, shuffle=True)
        gen = kf.split(list(range(len(self.dataset))))
        train_infos = []
        if kwargs["skip_connections"]:
            model_class = GINRes
        else:
            model_class = GIN
        self.model = model_class(
            device=self.device,
            in_channels=1,
            hidden_channels=kwargs["hidden_channels"],
            out_channels=NUM_CLASSES,
            num_gin_layers=kwargs["num_gin_layers"],
            num_mlp_layers=kwargs["num_mlp_layers"],
            num_mlp_readout_layers=kwargs["num_mlp_readout_layers"]
        ).to(self.device)
        if self.compile_model:
            self.model = torch.compile(self.model, dynamic=True, fullgraph=True)
        for (train_indices, eval_indices) in gen:
            train_loader = DataLoader(self.dataset[list(train_indices)], batch_size=int(kwargs["batch_size"]),
                                      shuffle=True, drop_last=True, num_workers=self.num_workers)
            # Need to enable drop_last so that there are no batches of size 1, which would error the batch norm layers.(No need during evaluation)
            eval_loader = DataLoader(self.dataset[list(eval_indices)], batch_size=int(kwargs["batch_size"]), num_workers=self.num_workers)
            epochs_info = self.train_eval(train_loader, eval_loader, kwargs)
            objective_values.append(epochs_info[-1]['eval_obj'])
            print(f"Finished split with value {objective_values[-1]}")
            train_infos.append(epochs_info)
            self.model.reset_parameters()
        objective = np.mean(objective_values)
        end = time.time()
        calc_factor = 0
        print(f"Finished current parameter set with {-objective + calc_factor} in {end - start}s")
        self.log_info.append((kwargs, train_infos, objective_values, end - start))
        return -objective + calc_factor


def optimize(dataset, device, search_space, num_bayes_samples, num_workers, seed, compile_model):
    obj = Objective(dataset, seed, device, num_workers, compile_model)

    @use_named_args(dimensions=search_space)
    def objective(**kwargs):
        return obj(**kwargs)

    res = skopt.gp_minimize(
        objective,
        search_space,
        n_calls=num_bayes_samples,
        random_state=seed,
        n_initial_points=int(num_bayes_samples / 5) if num_bayes_samples / 5 >= 1 else 1,
    )
    return res, obj.log_info
