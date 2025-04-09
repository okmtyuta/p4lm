import h5py
import torch
from tqdm import tqdm

from src.modules.dataloader.dataloader import DataBatch, Dataloader
from src.modules.model.configurable_model import ConfigurableModel
from src.modules.protein.protein_list import ProteinList
from src.modules.train.criterion import Criterion
from src.modules.train.types import EpochResult, PropEpochResult


class EpochResultList:
    def __init__(self, raw: list[EpochResult]):
        self._raw = raw

    @property
    def raw(self):
        return self._raw

    def append(self, result: EpochResult):
        self._raw.append(result)

    def save_as_hdf5(self, path: str):
        with h5py.File(name=path, mode="w") as f:
            f.create_group("result")
            print("result saving...")
            for epoch_result in tqdm(self._raw):
                epoch_group = f.create_group(f'result/{epoch_result["epoch"]}')
                epoch_group_attrs = epoch_group.attrs
                epoch_group_attrs["epoch"] = epoch_result["epoch"]

                for result in epoch_result["results"]:
                    prop_group = f.create_group(f'result/{epoch_result["epoch"]}/{result['prop_name']}')
                    f.create_dataset(
                        f'result/{epoch_result["epoch"]}/{result['prop_name']}/output', data=result["output"]
                    )
                    f.create_dataset(
                        f'result/{epoch_result["epoch"]}/{result['prop_name']}/label', data=result["label"]
                    )
                    prop_group_attrs = prop_group.attrs
                    prop_group_attrs["pearsonr"] = result["pearsonr"]
                    prop_group_attrs["mean_squared_error"] = result["mean_squared_error"]
                    prop_group_attrs["root_mean_squared_error"] = result["root_mean_squared_error"]
                    prop_group_attrs["mean_absolute_error"] = result["mean_absolute_error"]

    @classmethod
    def from_hdf5(cls, path: str):
        with h5py.File(path, mode="r") as f:
            raw: list[EpochResult] = []

            epoch_keys = f["result"].keys()
            print("result loading...")
            for epoch_key in tqdm(epoch_keys):
                epoch_group = f[f"result/{epoch_key}"]
                epoch_group_attrs = epoch_group.attrs
                epoch = epoch_group_attrs["epoch"]

                prop_names = epoch_group.keys()
                prop_epoch_results: list[PropEpochResult] = []
                for prop_name in prop_names:
                    prop_group = f[f"result/{epoch_key}/{prop_name}"]
                    output = f[f"result/{epoch_key}/{prop_name}/output"]
                    label = f[f"result/{epoch_key}/{prop_name}/label"]

                    prop_group_attrs = prop_group.attrs
                    pearsonr = prop_group_attrs["pearsonr"]
                    mean_squared_error = prop_group_attrs["mean_squared_error"]
                    root_mean_squared_error = prop_group_attrs["root_mean_squared_error"]
                    mean_absolute_error = prop_group_attrs["mean_absolute_error"]

                    prop_epoch_result: PropEpochResult = {
                        "prop_name": prop_name,
                        "label": label,
                        "output": output,
                        "pearsonr": pearsonr,
                        "mean_squared_error": mean_squared_error,
                        "root_mean_squared_error": root_mean_squared_error,
                        "mean_absolute_error": mean_absolute_error,
                    }
                    prop_epoch_results.append(prop_epoch_result)

                epoch_result: EpochResult = {"epoch": epoch, "results": prop_epoch_results}
                raw.append(epoch_result)

        return EpochResultList(raw=raw)


class Trainer:
    def __init__(self, model: ConfigurableModel, dataloader: Dataloader):
        self._model = model
        self._current_epoch = 1
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
                "pearsonr": pearsonr,
                "mean_squared_error": mean_squared_error.item(),
                "root_mean_squared_error": root_mean_squared_error.item(),
                "mean_absolute_error": mean_absolute_error.item(),
            }
            epoch_results_by_target.append(result)

        epoch_result: EpochResult = {"epoch": self._current_epoch, "results": epoch_results_by_target}
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

    def _to_be_continued(self):
        return self._current_epoch < 500

    def train(self) -> None:
        train_result_list = EpochResultList(raw=[])
        evaluate_result_list = EpochResultList(raw=[])
        while self._to_be_continued():
            for i in tqdm(range(500)):
                train_epoch_result = self._epoch_train()
                train_result_list.append(train_epoch_result)

                evaluate_epoch_result = self._epoch_evaluate()
                evaluate_result_list.append(evaluate_epoch_result)

                for r in evaluate_epoch_result["results"]:
                    print(r["prop_name"], r["pearsonr"])

                self._current_epoch += 1

        # train_result_list.save_as_hdf5("train-test.h5")
        # evaluate_result_list.save_as_hdf5("evaluate-test.h5")
