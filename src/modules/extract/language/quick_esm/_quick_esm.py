import os
from typing import Literal, TypedDict

import h5py
import torch

from src.modules.extract.language._language import _Language
from src.modules.protein.protein_list import ProteinList


class _AminoAcidSource(TypedDict):
    name: str
    char: str
    representation: torch.Tensor


QuickESMModelName = Literal["esm2", "esm1b"]


class _QuickESMLanguage(_Language):
    def __init__(self, model_name: QuickESMModelName):
        super().__init__()
        self._model_name: QuickESMModelName = model_name
        self._source = self._load_source()

    def __call__(self, protein_list: ProteinList):
        for protein in protein_list.proteins:
            protein.set_representations(self._convert(protein.seq))

        return protein_list

    def _get_source_path(self):
        if self._model_name == "esm2":
            return os.path.join(os.path.dirname(__file__), "esm2.h5")
        elif self._model_name == "esm1b":
            return os.path.join(os.path.dirname(__file__), "esm1b.h5")

    def _load_source(self) -> dict[str, _AminoAcidSource]:
        source: dict[str, _AminoAcidSource] = {}
        path = self._get_source_path()
        with h5py.File(path, mode="r") as f:
            keys = f["amino_acid"].keys()

            for key in keys:
                data = f[f"amino_acid/{key}"]
                attrs = data.attrs

                char = attrs["char"]
                name = attrs["name"]
                representation = torch.from_numpy(data[:])
                source[key] = {"char": char, "name": name, "representation": representation}

        return source

    def _convert(self, chars: str):
        representations: list[torch.Tensor] = []
        for char in list(chars):
            representation = self._source[char]["representation"]
            representations.append(representation)

        return torch.stack(representations)
