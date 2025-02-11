from typing import Optional, TypedDict

import torch

from src.modules.data_pipeline.data_pipeline import DataPipeline
from src.modules.protein.protein_list import ProteinList, ProteinProp


class DataloaderStateSource(TypedDict):
    protein_list: ProteinList
    batch_size: int
    input_props: list[ProteinProp]
    output_props: list[ProteinProp]
    pipeline: DataPipeline
    cacheable: bool


UsableDataBatch = tuple[torch.Tensor, torch.Tensor, ProteinList]


class DataloaderState:
    def __init__(self, source: DataloaderStateSource):
        self._source = source

    @property
    def protein_list(self):
        return self._source["protein_list"]

    @property
    def batch_size(self):
        return self._source["batch_size"]

    @property
    def input_props(self):
        return self._source["input_props"]

    @property
    def output_props(self):
        return self._source["output_props"]

    @property
    def pipeline(self):
        return self._source["pipeline"]

    @property
    def cacheable(self):
        return self._source["cacheable"]

    def as_source(self) -> DataloaderStateSource:
        return {
            "protein_list": self._source["protein_list"],
            "batch_size": self._source["batch_size"],
            "input_props": self._source["input_props"],
            "output_props": self._source["output_props"],
            "pipeline": self._source["pipeline"],
            "cacheable": self._source["cacheable"],
        }

    def rational_split(self, ratios: list[float]) -> list["DataloaderState"]:
        states: list["DataloaderState"] = []
        for protein_list in self._source["protein_list"].rational_split(ratios=ratios):
            source = self.as_source()
            source["protein_list"] = protein_list
            state = DataloaderState(source=source)

            states.append(state)

        return states

    def even_split(self, unit_size: int) -> list["DataloaderState"]:
        states: list["DataloaderState"] = []
        for protein_list in self._source["protein_list"].even_split(unit_size=unit_size):
            source = self.as_source()
            source["protein_list"] = protein_list
            state = DataloaderState(source=source)

            states.append(state)

        return states


class DataBatch:
    def __init__(self, state: DataloaderState):
        self._state = state

        self._cache: Optional[UsableDataBatch] = None

    def __len__(self):
        return len(self._state.protein_list)

    @property
    def input_props(self):
        return self._state.input_props

    @property
    def output_props(self):
        return self._state.output_props

    def use(self):
        if self._state.cacheable and self._cache is not None:
            return self._cache

        inputs = []
        outputs = []

        protein_list = self._state.pipeline(protein_list=self._state.protein_list)

        for protein in protein_list.proteins:
            piped = protein.piped
            input_props = torch.Tensor([protein.read_props(key) for key in self._state.input_props])
            input = torch.cat([piped, input_props], dim=0)
            inputs.append(input)

            output = [protein.read_props(key) for key in self._state.output_props]
            outputs.append(output)

        usable = (
            torch.stack(inputs).to(torch.float32),
            torch.Tensor(outputs).to(torch.float32),
            self._state.protein_list,
        )

        if self._state.cacheable:
            self._cache = usable

        return usable


class Dataloader:
    def __init__(self, state: DataloaderState):
        self._state = state
        self._batches: Optional[list[DataBatch]] = None

    def __len__(self):
        return len(self._state.protein_list)

    @property
    def batches(self):
        return self._generate_batch()

    def _generate_batch(self):
        if self._batches is not None:
            return self._batches

        states = self._state.even_split(unit_size=self._state.batch_size)

        batches = [DataBatch(state=state) for state in states]

        self._batches = batches
        return batches

    def rational_split(self, ratios: list[float]) -> list["Dataloader"]:
        dataloaders: list["Dataloader"] = []
        for state in self._state.rational_split(ratios=ratios):
            dataloader = Dataloader(state=state)
            dataloaders.append(dataloader)

        return dataloaders
