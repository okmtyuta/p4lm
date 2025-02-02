from typing import Literal, TypedDict

import esm
import numpy as np
import numpy.typing as npt
import torch

from src.modules.language._language import _Language
from src.modules.protein.protein import ProteinList

ESMModelName = Literal["esm2", "esm1b"]


class ESMModelResult(TypedDict):
    logits: torch.Tensor
    representations: dict[int, torch.Tensor]
    attentions: torch.Tensor
    contacts: torch.Tensor


class _ESMLanguage(_Language):
    def __init__(self, model_name: ESMModelName):
        super().__init__()
        self._model_name = model_name
        self._set_model_and_alphabet()
        self._batch_converter = self._alphabet.get_batch_converter()
        self._model.eval()

    def __call__(self, protein_list: ProteinList):
        self._set_representations(protein_list=protein_list)
        return protein_list

    def _set_model_and_alphabet(self):
        model, alphabet = self._get_model_alphabet()
        self._model = model
        self._alphabet = alphabet

    def _get_model_alphabet(self):
        if self._model_name == "esm2":
            return esm.pretrained.esm2_t33_650M_UR50D()
        if self._model_name == "esm1b":
            return esm.pretrained.esm1b_t33_650M_UR50S()
        else:
            raise Exception()

    def _convert(self, data: list[tuple[str, str]]):
        batch_tokens = self._batch_converter(data)[2]
        batch_lens = (batch_tokens != self._alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results: ESMModelResult = self._model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=True,
            )

        token_representations: torch.Tensor = results["representations"][33]

        sequence_representations: list[npt.NDArray] = []
        for i, tokens_len in enumerate(batch_lens):
            representation = token_representations[i, 1 : tokens_len - 1].mean(0).numpy()
            sequence_representations.append(representation)  # noqa: E203
        return torch.from_numpy(np.array(sequence_representations))

    def _set_representations(self, protein_list: ProteinList) -> ProteinList:
        data: list[tuple[str, str]] = []

        for protein in protein_list.proteins:
            for index, amino_acid in enumerate(list(protein.seq)):
                datum = (f"{protein.seq}_{index}", amino_acid)
                data.append(datum)

        batch_representations = self._convert(data=data)
        progress = 0

        for protein in protein_list.proteins:
            representations = batch_representations[progress : progress + protein.length]  # noqa: E203

            protein.set_representations(representations=representations)
            progress += protein.length

        return protein_list
