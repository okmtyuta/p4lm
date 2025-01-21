import abc

import torch

from src.protein.protein import ProteinList


class _Language(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def __call__(self, proteins: ProteinList) -> torch.Tensor:
        raise NotImplementedError()
