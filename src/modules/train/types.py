from typing import TypedDict

from src.modules.protein.protein_list import ProteinProp


class PropEpochResult(TypedDict):
    prop_name: ProteinProp
    label: list[float]
    output: list[float]
    pearsonr: float
    mean_squared_error: float
    root_mean_squared_error: float
    mean_absolute_error: float


class EpochResult(TypedDict):
    epoch: int
    results: list[PropEpochResult]


class TrainResult(TypedDict):
    train: EpochResult
    evaluate: EpochResult
