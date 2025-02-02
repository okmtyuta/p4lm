import abc

import torch

from src.modules.protein.protein import ProteinList


class _Language(metaclass=abc.ABCMeta):
    def __call__(self, protein_list: ProteinList) -> torch.Tensor:
        raise NotImplementedError()
