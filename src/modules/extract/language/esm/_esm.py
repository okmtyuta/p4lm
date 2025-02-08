from typing import Literal, TypedDict

import torch

from src.modules.extract.language._language import _Language
from src.modules.extract.language.esm.esm_converter import ESMConverter
from src.modules.protein.protein_list import ProteinList

ESMModelName = Literal["esm2", "esm1b"]


class ESMModelResult(TypedDict):
    logits: torch.Tensor
    representations: dict[int, torch.Tensor]
    attentions: torch.Tensor
    contacts: torch.Tensor


class _ESMLanguage(_Language):
    def __init__(self, model_name: ESMModelName):
        super().__init__()
        self._converter = ESMConverter(model_name=model_name)

    def __call__(self, protein_list: ProteinList):
        self._set_representations(protein_list=protein_list)
        return protein_list

    def _set_representations(self, protein_list: ProteinList) -> ProteinList:
        batch: list[str] = []

        for protein in protein_list.proteins:
            batch += list(protein.seq)

        batch_representations = self._converter(seqs=batch)
        progress = 0

        for protein in protein_list.proteins:
            length = len(protein.seq)
            representations = batch_representations[progress : progress + length]  # noqa: E203

            protein.set_representations(representations=representations)
            progress += length

        return protein_list
