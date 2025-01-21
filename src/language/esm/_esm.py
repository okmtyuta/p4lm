from typing import Literal

import torch
from tqdm import tqdm

from src.language._language import _Language
from src.protein.protein import ProteinList

ESMModelName = Literal["esm2", "esm1b"]


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
        model_key = self._get_model_key()
        model, alphabet = torch.hub.load("facebookresearch/esm:main", model_key)
        self._model = model
        self._alphabet = alphabet

    def _get_model_key(self):
        if self._model_name == "esm2":
            return "esm2_t33_650M_UR50D"
        if self._model_name == "esm1b":
            return "esm1b_t33_650M_UR50S"
        else:
            raise Exception()

    def _convert(self, data: list[tuple[str, str]]) -> torch.Tensor:
        batch_tokens = self._batch_converter(data)[2]
        batch_lens = (batch_tokens != self._alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = self._model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=True,
            )

        token_representations: torch.Tensor = results["representations"][33]

        sequence_representations: list = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).tolist())  # noqa: E203

        return torch.Tensor(sequence_representations)

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
