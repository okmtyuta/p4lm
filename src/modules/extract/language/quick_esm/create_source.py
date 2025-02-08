import os
from typing import TypedDict

import h5py

from src.modules.extract.language.esm.esm_converter import ESMConverter
from src.modules.extract.language.esm.esm1b import ESM1bLanguage
from src.modules.extract.language.esm.esm2 import ESM2Language


class _AminoAcidSource(TypedDict):
    name: str
    char: str


_amino_acid_sources: list[_AminoAcidSource] = [
    {"name": "Alanine", "char": "A"},
    {"name": "Arginine", "char": "R"},
    {"name": "Asparagine", "char": "N"},
    {"name": "Aspartate", "char": "D"},
    {"name": "Cysteine", "char": "C"},
    {"name": "Glutamine", "char": "Q"},
    {"name": "Glutamate", "char": "E"},
    {"name": "Glycine", "char": "G"},
    {"name": "Histidine", "char": "H"},
    {"name": "Isoleucine", "char": "I"},
    {"name": "Leucine", "char": "L"},
    {"name": "Lysine", "char": "K"},
    {"name": "Methionine", "char": "M"},
    {"name": "Phenylalanine", "char": "F"},
    {"name": "Proline", "char": "P"},
    {"name": "Serine", "char": "S"},
    {"name": "Threonine", "char": "T"},
    {"name": "Tryptophan", "char": "W"},
    {"name": "Tyrosine", "char": "Y"},
    {"name": "Valine", "char": "V"},
]


def esm2():
    esm2_converter = ESMConverter("esm2")
    path = os.path.join(os.path.dirname(__file__), "esm2.h5")
    with h5py.File(name=path, mode="w") as f:
        f.create_group("amino_acid")
        for source in _amino_acid_sources:
            representations = esm2_converter(list(source["char"])).squeeze(dim=0)
            dataset = f.create_dataset(f"amino_acid/{source['char']}", data=representations, dtype="float32")
            attrs = dataset.attrs
            attrs["name"] = source["name"]
            attrs["char"] = source["char"]


def esm1b():
    esm1b_converter = ESMConverter("esm1b")
    path = os.path.join(os.path.dirname(__file__), "esm1b.h5")
    with h5py.File(name=path, mode="w") as f:
        f.create_group("amino_acid")
        for source in _amino_acid_sources:
            representations = esm1b_converter(list(source["char"])).squeeze(dim=0)
            dataset = f.create_dataset(f"amino_acid/{source['char']}", data=representations, dtype="float32")
            attrs = dataset.attrs
            attrs["name"] = source["name"]
            attrs["char"] = source["char"]


def main():
    esm1b()
    esm2()


if __name__ == "__main__":
    main()
