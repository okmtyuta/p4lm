from typing import Optional

import torch
from tqdm import tqdm

from src.modules.dataloader.dataloader import DataBatch, Dataloader
from src.modules.model.configurable_model import ConfigurableModel
from src.modules.protein.protein_list import ProteinList
from src.modules.train.criterion import Criterion
from src.modules.train.types import EpochResult, PropEpochResult


class TrainRecorder:
    def __init__(self):
        self._current_epoch = 1
        self._max_accuracy_epoch_result: Optional[EpochResult] = None

        self._train_results: list[EpochResult] = []
        self._evaluate_results: list[EpochResult] = []

    @property
    def current_epoch(self):
        return self._current_epoch

    @property
    def max_accuracy_epoch_result(self):
        if self._max_accuracy_epoch_result is None:
            raise Exception
        return self._max_accuracy_epoch_result

    def next_epoch(self):
        self._current_epoch += 1

    def append_train_results(self, train_epoch_result: EpochResult):
        self._train_results.append(train_epoch_result)

    def append_evaluate_results(self, evaluate_epoch_result: EpochResult):
        self._evaluate_results.append(evaluate_epoch_result)

    def is_max_accuracy_epoch_result(self, epoch_result: EpochResult):
        pearsonrs = map(lambda result: result["pearsonr"], epoch_result["results"])
        accuracy = sum(pearsonrs)

        if self._max_accuracy_epoch_result is None:
            return True

        max_accuracy_pearsonrs = map(lambda result: result["pearsonr"], self._max_accuracy_epoch_result["results"])
        max_accuracy = sum(max_accuracy_pearsonrs)

        return accuracy > max_accuracy

    def set_as_accuracy_epoch_result(self, epoch_result: EpochResult):
        self._max_accuracy_epoch_result = epoch_result

    def to_continue(self):
        if self._max_accuracy_epoch_result is None:
            return True

        max_accuracy_epoch = self._max_accuracy_epoch_result["epoch"]
        return self._current_epoch - max_accuracy_epoch < 500


class Trainer:
    def __init__(self, model: ConfigurableModel, dataloader: Dataloader):
        self._model = model
        self._recorder = TrainRecorder()

        self._dataloader = dataloader
        self._train_loader, self._evaluate_loader, self._validate_loader = self._dataloader.rational_split(
            [0.8, 0.1, 0.1]
        )
        self._criterion = Criterion()

        self._model.train()
        self._model.optimizer.train()

    def _create_epoch_results(self, labels: torch.Tensor, outputs: torch.Tensor):
        epoch_results_by_target: list[PropEpochResult] = []
        for i, prop_name in enumerate(self._dataloader.state.output_props):
            output = outputs[:, i]
            label = labels[:, i]
            mean_squared_error = self._criterion.mean_squared_error(output=output, label=label)
            root_mean_squared_error = self._criterion.root_mean_squared_error(output=output, label=label)
            mean_absolute_error = self._criterion.mean_absolute_error(output=output, label=label)
            pearsonr = self._criterion.pearsonr(label=label, output=output)

            result: PropEpochResult = {
                "prop_name": prop_name,
                "output": output.tolist(),
                "label": label.tolist(),
                "pearsonr": pearsonr.item(),
                "mean_squared_error": mean_squared_error.item(),
                "root_mean_squared_error": root_mean_squared_error.item(),
                "mean_absolute_error": mean_absolute_error.item(),
            }
            epoch_results_by_target.append(result)

        epoch_result: EpochResult = {"epoch": self._recorder.current_epoch, "results": epoch_results_by_target}
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
            batch_labels.append(label)
            batch_outputs.append(output)

        labels = torch.cat(batch_labels)
        outputs = torch.cat(batch_outputs)

        epoch_train_result = self._create_epoch_results(labels=labels, outputs=outputs)
        return epoch_train_result

    def _epoch_evaluate(self) -> EpochResult:
        batch_labels: list[torch.Tensor] = []
        batch_outputs: list[torch.Tensor] = []
        for batch in self._train_loader.batches:
            self._model.optimizer.zero_grad()
            label, output, protein = self._batch_evaluate(batch=batch)
            batch_labels.append(label)
            batch_outputs.append(output)

        labels = torch.cat(batch_labels)
        outputs = torch.cat(batch_outputs)

        epoch_evaluate_result = self._create_epoch_results(labels=labels, outputs=outputs)
        return epoch_evaluate_result

    def _post_epoch_evaluate(self, epoch_evaluate_result: EpochResult):
        if self._recorder.is_max_accuracy_epoch_result(epoch_result=epoch_evaluate_result):
            self._recorder.set_as_accuracy_epoch_result(epoch_result=epoch_evaluate_result)

    def train(self) -> None:
        while self._recorder.to_continue():
            for i in tqdm(range(500)):
                train_epoch_result = self._epoch_train()
                self._recorder.append_train_results(train_epoch_result=train_epoch_result)

                evaluate_epoch_result = self._epoch_evaluate()
                self._recorder.append_evaluate_results(evaluate_epoch_result=evaluate_epoch_result)
                self._post_epoch_evaluate(epoch_evaluate_result=evaluate_epoch_result)

                for r in evaluate_epoch_result["results"]:
                    mae = self._recorder._max_accuracy_epoch_result["epoch"]
                    ce = self._recorder._current_epoch
                    print(f'Current epoch is {ce}, {r["prop_name"]} pearson: {r["pearsonr"]}')
                mape = map(lambda r: r["pearsonr"], self._recorder.max_accuracy_epoch_result["results"])
                print(f'Max accuracy epoch is {mae}, {r["prop_name"]} pearson: {list(mape)}')

                self._recorder.next_epoch()

        # train_result_list.save_as_hdf5("train-test.h5")
        # evaluate_result_list.save_as_hdf5("evaluate-test.h5")
