"""
Training script for WaveHMSTGNN, with variable names aligned to the paper.

Expected flattened input layout:
    [B, T, N * V]
where N = num_nodes and V = num_vars.
The helper `select_nodes_and_flatten` converts data stored as [B, T, V, spatial]
into the model layout [B, T, N * V].
"""

import math
import pickle
import random
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import configs
from WaveHMSTGNN import WaveHMSTGNN


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(configs.seed)


class MarineForecastDataset(Dataset):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        super().__init__()
        self.input = train_x
        self.target = train_y

    def get_data_shape(self) -> Dict[str, torch.Size]:
        return {"input": self.input.shape, "target": self.target.shape}

    # Keep original method name for compatibility.
    def GetDataShape(self) -> Dict[str, torch.Size]:
        return self.get_data_shape()

    def __len__(self) -> int:
        return self.input.shape[0]

    def __getitem__(self, idx: int):
        return self.input[idx], self.target[idx]


# Backward-compatible alias.
dataset_package = MarineForecastDataset


class Trainer:
    def __init__(self, cfg):
        self.configs = cfg
        self.device = cfg.device
        self.input_dim = cfg.input_dim
        self.num_vars = int(getattr(cfg, "num_vars", 6))
        self.num_nodes = int(getattr(cfg, "num_nodes", cfg.enc_in // self.num_vars))
        self.variable_names = list(
            getattr(
                cfg,
                "variable_names",
                ["sss", "sst", "u_current", "v_current", "wind_stress_x", "wind_stress_y"],
            )
        )

        self.network = WaveHMSTGNN(cfg).to(self.device)
        self.opt = torch.optim.Adam(
            [{"params": self.network.parameters()}],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    def _reshape_to_node_variable(self, y: torch.Tensor) -> torch.Tensor:
        return y.reshape(y.shape[0], y.shape[1], self.num_nodes, self.num_vars)

    def rmse_for_channels(self, y_pred: torch.Tensor, y_true: torch.Tensor, channels: Sequence[int]) -> torch.Tensor:
        y_pred = self._reshape_to_node_variable(y_pred)
        y_true = self._reshape_to_node_variable(y_true)
        channels = list(channels)
        error = y_pred[..., channels] - y_true[..., channels]
        return torch.mean(error ** 2).sqrt()

    def rmse_for_variable(self, y_pred: torch.Tensor, y_true: torch.Tensor, variable_index: int) -> torch.Tensor:
        return self.rmse_for_channels(y_pred, y_true, [variable_index])

    def paper_group_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """RMSE groups used by the paper tables: SSS, SST, velocity, wind stress."""
        group_channels = getattr(
            self.configs,
            "variable_groups",
            {
                "SSS": [0],
                "SST": [1],
                "Sea water velocity": [2, 3],
                "Sea surface wind stress": [4, 5],
            },
        )
        return {name: self.rmse_for_channels(y_pred, y_true, channels) for name, channels in group_channels.items()}

    def train_once(self, input_series: torch.Tensor, target_series: torch.Tensor):
        prediction = self.network(input_series.float().to(self.device))
        target_series = target_series.float().to(self.device)

        self.opt.zero_grad()
        variable_losses = [
            self.rmse_for_variable(prediction, target_series, variable_index)
            for variable_index in range(self.num_vars)
        ]
        loss = sum(variable_losses)
        loss.backward()

        if self.configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.configs.clipping_threshold)

        self.opt.step()
        return [loss_item.item() for loss_item in variable_losses], loss.item()

    def test(self, dataloader_test: DataLoader) -> torch.Tensor:
        predictions = []
        with torch.no_grad():
            for input_series, _ in dataloader_test:
                prediction = self.network(input_series.float().to(self.device))
                predictions.append(prediction)
        return torch.cat(predictions, dim=0)

    def infer(self, dataset: MarineForecastDataset, dataloader: DataLoader) -> Dict[str, float]:
        self.network.eval()
        with torch.no_grad():
            prediction = self.test(dataloader)
            target = dataset.target.float().to(self.device)
            variable_metrics = {
                self.variable_names[idx]: self.rmse_for_variable(prediction, target, idx).item()
                for idx in range(self.num_vars)
            }
            group_metrics = {
                f"group/{name}": value.item()
                for name, value in self.paper_group_metrics(prediction, target).items()
            }
        return {**variable_metrics, **group_metrics}

    @staticmethod
    def _format_metrics(metrics: Dict[str, float]) -> str:
        return ", ".join([f"{name}: {value:.4f}" for name, value in metrics.items()])

    def train(self, dataset_train: MarineForecastDataset, dataset_eval: MarineForecastDataset, chk_path: str) -> None:
        print("loading train dataloader")
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print("loading eval dataloader")
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = math.inf

        for epoch in range(self.configs.num_epochs):
            print(f"\nepoch: {epoch + 1}")
            self.network.train()

            for batch_idx, (input_series, target_series) in enumerate(dataloader_train):
                variable_losses, loss = self.train_once(input_series, target_series)

                if (batch_idx + 1) % self.configs.display_interval == 0:
                    loss_text = ", ".join(
                        f"{name}: {value:.4f}" for name, value in zip(self.variable_names, variable_losses)
                    )
                    print(f"batch training loss: {loss_text}, total: {loss:.4f}")

                # Keep the original mid-epoch evaluation behavior.
                if (epoch + 1 >= 6) and (batch_idx + 1) % (self.configs.display_interval * 2) == 0:
                    eval_metrics = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
                    loss_eval = sum(eval_metrics[name] for name in self.variable_names)
                    print(f"batch eval loss: {self._format_metrics(eval_metrics)}, total: {loss_eval:.4f}")
                    if loss_eval < best:
                        self.save_model(f"{chk_path}_{loss_eval}.chk")
                        best = loss_eval
                        count = 0

            eval_metrics = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            loss_eval = sum(eval_metrics[name] for name in self.variable_names)
            print(f"epoch eval loss: {self._format_metrics(eval_metrics)}, total: {loss_eval:.4f}")

            if loss_eval >= best:
                count += 1
                print(f"eval loss is not reduced for {count} epoch")
            else:
                print(f"eval loss is reduced from {best:.5f} to {loss_eval:.5f}, saving model")
                self.save_model(f"{chk_path}_{loss_eval}.chk")
                best = loss_eval
                count = 0

            if count == self.configs.patience:
                print(f"early stopping reached, best score is {best:.5f}")
                break

    def save_configs(self, config_path: str) -> None:
        with open(config_path, "wb") as path:
            pickle.dump(self.configs, path)

    def save_model(self, path: str) -> None:
        torch.save({"net": self.network.state_dict(), "optimizer": self.opt.state_dict()}, path)


def select_nodes_and_flatten(data: np.ndarray, node_idx: np.ndarray) -> torch.Tensor:
    """
    Convert raw data from [B, T, V, spatial] to [B, T, N * V].

    The model internally views the flattened vector as [N, V]. Therefore we first
    put nodes before variables: [B, T, N, V], then flatten the last two dims.
    """
    tensor = torch.tensor(data)
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], -1)
    tensor = tensor[:, :, :, node_idx]  # [B, T, V, N]
    tensor = tensor.permute(0, 1, 3, 2).contiguous()  # [B, T, N, V]
    return tensor.flatten(-2, -1)  # [B, T, N * V]


if __name__ == "__main__":
    print("Configs:\n", configs.__dict__)

    node_idx = np.random.randint(0, 64, configs.num_nodes)
    print("Nodes id:")
    print(node_idx)

    data = np.load(configs.data_path)
    train_x = select_nodes_and_flatten(data["train_x"], node_idx)
    train_y = select_nodes_and_flatten(data["train_y"], node_idx)
    test_x = select_nodes_and_flatten(data["test_x"], node_idx)
    test_y = select_nodes_and_flatten(data["test_y"], node_idx)

    dataset_train = MarineForecastDataset(train_x=train_x, train_y=train_y)
    dataset_test = MarineForecastDataset(train_x=test_x, train_y=test_y)
    del train_x, train_y, test_x, test_y

    print("Dataset_train Shape:\n", dataset_train.get_data_shape())
    print("Dataset_test Shape:\n", dataset_test.get_data_shape())

    trainer = Trainer(configs)
    trainer.save_configs("config.pkl")
    trainer.train(dataset_train, dataset_test, "checkpoint")
