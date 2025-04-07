from typing import TypedDict

import torch
from tqdm import tqdm

from src.modules.dataloader.dataloader import DataBatch, Dataloader
from src.modules.model.configurable_model import ConfigurableModel
from src.modules.protein.protein_list import ProteinList
from src.modules.train.criterion import Criterion


class EpochResult(TypedDict):
    labels: torch.Tensor
    outputs: torch.Tensor
    pearsonrs: torch.Tensor
    mean_squared_errors: torch.Tensor
    mean_absolute_errors: torch.Tensor


class TrainerConfig(TypedDict):
    epoch: int
    data_split_ratio: tuple[float, float, float]


class Trainer:
    def __init__(self, config: TrainerConfig, model: ConfigurableModel, dataloader: Dataloader):
        self._config = config
        self._model = model
        self._dataloader = dataloader
        self._train_loader, self._evaluate_loader, self._validate_loader = self._dataloader.rational_split(
            [0.8, 0.1, 0.1]
        )
        self._criterion = Criterion()

        self._model.train()
        self._model.optimizer.train()

    def _create_epoch_result(self, labels: torch.Tensor, outputs: torch.Tensor):
        pearsonrs = self._criterion.pearsonr(target=labels, output=outputs)

        mean_squared_errors: list[float] = []
        for i in range(len(self._dataloader.state.output_props)):
            mean_squared_error = self._criterion.mean_squared_error(output=outputs[:, i], target=labels[:, i])
            mean_squared_errors.append(mean_squared_error.item())

        mean_absolute_errors: list[float] = []
        for i in range(len(self._dataloader.state.output_props)):
            mean_absolute_error = self._criterion.mean_absolute_error(output=outputs[:, i], target=labels[:, i])
            mean_absolute_errors.append(mean_absolute_error.item())

        epoch_result: EpochResult = {
            "outputs": outputs,
            "labels": labels,
            "pearsonrs": pearsonrs,
            "mean_squared_errors": torch.tensor(mean_squared_errors),
            "mean_absolute_errors": torch.tensor(mean_absolute_errors),
        }

        return epoch_result

    def _batch_train(self, batch: DataBatch):
        self._model.optimizer.zero_grad()
        input, label, protein = batch.use()

        output = self._model(input=input)

        loss = self._criterion.mean_squared_error(output, label)
        loss.backward()
        self._model.optimizer.step()

        return label, output, protein

    def _batch_evaluate(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor, ProteinList]:
        self._model.optimizer.zero_grad()
        input, label, protein = batch.use()

        output = self._model(input=input)

        return label, output, protein

    def _epoch_train(self) -> EpochResult:
        batch_labels: list[torch.Tensor] = []
        batch_outputs: list[torch.Tensor] = []
        for batch in self._train_loader.batches:
            label, output, protein = self._batch_train(batch=batch)
            batch_labels.append(label.squeeze(dim=1))
            batch_outputs.append(output.squeeze(dim=1))

        labels = torch.cat(batch_labels)
        outputs = torch.cat(batch_outputs)

        epoch_train_result = self._create_epoch_result(labels=labels, outputs=outputs)
        return epoch_train_result

    def _epoch_evaluate(self) -> EpochResult:
        batch_labels: list[torch.Tensor] = []
        batch_outputs: list[torch.Tensor] = []
        for batch in self._train_loader.batches:
            self._model.optimizer.zero_grad()
            label, output, protein = self._batch_evaluate(batch=batch)
            batch_labels.append(label.squeeze(dim=1))
            batch_outputs.append(output.squeeze(dim=1))

        labels = torch.cat(batch_labels)
        outputs = torch.cat(batch_outputs)

        epoch_evaluate_result = self._create_epoch_result(labels=labels, outputs=outputs)
        return epoch_evaluate_result

    def train(self) -> tuple[list[EpochResult], list[EpochResult]]:
        evaluate_results: list[EpochResult] = []
        train_results: list[EpochResult] = []
        for i in tqdm(range(self._config["epoch"])):
            train_result = self._epoch_train()
            train_results.append(train_result)

            evaluate_result = self._epoch_evaluate()
            evaluate_results.append(evaluate_result)

        return train_results, evaluate_results
