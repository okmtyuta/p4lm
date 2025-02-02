from typing import Optional

import torch

from src.modules.data_pipeline.data_pipeline import DataPipeline
from src.modules.protein.protein import ProteinList, ProteinProp


class DataBatch:
    def __init__(
        self,
        protein_list: ProteinList,
        batch_size: int,
        input_props: list[ProteinProp],
        output_props: list[ProteinProp],
        pipeline: DataPipeline,
        cacheable: bool,
    ):
        self._protein_list = protein_list
        self._batch_size = batch_size
        self._input_props = input_props
        self._output_props = output_props
        self._pipeline = pipeline
        self._cacheable = cacheable
        self._cache: Optional[tuple[torch.Tensor, torch.Tensor, ProteinList]] = None

    @property
    def size(self):
        return self._protein_list.size

    def use(self):
        if self._cacheable and self._cache is not None:
            return self._cache

        inputs = []
        outputs = []

        protein_list = self._pipeline(protein_list=self._protein_list)

        for protein in protein_list.proteins:
            piped = protein.piped
            input_props = torch.Tensor([protein.read_prop(key) for key in self._input_props])
            input = torch.cat([piped, input_props], dim=0)
            inputs.append(input)

            output = [protein.read_prop(key) for key in self._output_props]
            outputs.append(output)

        usable = (
            torch.stack(inputs).to(torch.float32),
            torch.Tensor(outputs).to(torch.float32),
            self._protein_list,
        )

        if self._cacheable:
            self._cache = usable

        return usable


class Dataloader:
    def __init__(
        self,
        protein_list: ProteinList,
        batch_size: int,
        input_props: list[ProteinProp],
        output_props: list[ProteinProp],
        pipeline: DataPipeline,
        cacheable: bool,
    ):
        self._protein_list = protein_list
        self._batch_size = batch_size
        self._batches: Optional[list[tuple[torch.Tensor, torch.Tensor, ProteinList]]] = None
        self._input_props = input_props
        self._output_props = output_props
        self._pipeline = pipeline
        self._cacheable = cacheable

    @property
    def size(self):
        return self._protein_list.size

    @property
    def batches(self):
        return self._generate_batch()

    def _generate_batch(self):
        if self._batches is not None:
            return self._batches

        protein_lists = self._protein_list.even_split(unit_size=self._batch_size)
        batches = [
            DataBatch(
                protein_list=protein_list,
                batch_size=self._batch_size,
                input_props=self._input_props,
                output_props=self._output_props,
                pipeline=self._pipeline,
                cacheable=self._cacheable,
            )
            for protein_list in protein_lists
        ]

        self._batches = batches
        return batches

    def _copy(self):
        return Dataloader(
            protein_list=self._protein_list,
            batch_size=self._batch_size,
            input_props=self._input_props,
            output_props=self._output_props,
            pipeline=self._pipeline,
            cacheable=self._cacheable,
        )

    def _replace_protein_list(self, protein_list: ProteinList):
        self._protein_list = protein_list
        return self

    def rational_split(self, ratios: list[float]) -> list["Dataloader"]:
        return [
            self._copy()._replace_protein_list(protein_list=protein_list)
            for protein_list in self._protein_list.rational_split(ratios=ratios)
        ]
