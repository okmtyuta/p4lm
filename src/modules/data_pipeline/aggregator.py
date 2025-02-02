import abc
from typing import Literal

import torch

from src.modules.data_pipeline.data_pipeline import DataPipe
from src.modules.protein.protein import Protein, ProteinList

AggregateMethod = Literal["mean"]


class Aggregator(DataPipe):
    def __init__(self, method: AggregateMethod) -> None:
        self._method: AggregateMethod = method

    def _act(self, protein):
        if self._method == "mean":
            mean = self._mean(input=protein.piped)
            return protein.set_piped(piped=mean)

        raise Exception

    def _mean(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(input=input, dim=0)
