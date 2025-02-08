from typing import Literal

import h5py
import polars as pl
import torch
from tqdm import tqdm

from src.lib.utils.utils import Utils
from src.modules.data.hdf.hdf5 import HDF5
from src.modules.protein.protein import Protein, ProteinProps, ProteinRaw, ProteinSource

ProteinLanguageName = Literal["esm2", "esm1b"]
protein_language_names: list[ProteinLanguageName] = ["esm2", "esm1b"]

ProteinProp = Literal["ccs", "rt", "mass", "length", "charge"]


class ProteinList:
    proteins_dir = "proteins"

    def __init__(self, proteins: list[Protein]):
        self._proteins = proteins

    def __len__(self):
        return len(self._proteins)

    @property
    def proteins(self):
        return self._proteins

    @classmethod
    def join(self, protein_lists: list["ProteinList"]):
        proteins: list[Protein] = []
        for protein_list in protein_lists:
            for protein in protein_list.proteins:
                proteins.append(protein)

        return ProteinList(proteins=proteins)

    @classmethod
    def from_csv(self, path: str):
        df = pl.read_csv(path)

        proteins: list[Protein] = []
        for row in df.iter_rows(named=True):
            raw: ProteinRaw = {"seq": row["seq"], "representations": None, "piped": None}
            props: ProteinProps = {
                "ccs": row.get("ccs"),
                "rt": row.get("rt"),
                "mass": row.get("mass"),
                "charge": row.get("charge"),
                "length": row.get("length"),
            }
            source: ProteinSource = {
                "raw": raw,
                "props": props,
                "key": row["index"],
            }
            protein = Protein(source=source)
            proteins.append(protein)

        return ProteinList(proteins=proteins)

    @classmethod
    def from_hdf5(self, path: str):
        with h5py.File(path, mode="r") as f:
            keys = f["proteins"].keys()

            proteins: list[Protein] = []
            for key in tqdm(keys):
                data = f[f"{self.proteins_dir}/{key}"]
                attrs = data.attrs

                raw: ProteinRaw = {"seq": attrs["seq"], "representations": torch.Tensor(data[:]), "piped": None}
                props: ProteinProps = {
                    "ccs": HDF5.read_nullable_attrs("ccs", attrs),
                    "rt": HDF5.read_nullable_attrs("rt", attrs),
                    "mass": HDF5.read_nullable_attrs("mass", attrs),
                    "length": HDF5.read_nullable_attrs("length", attrs),
                    "charge": HDF5.read_nullable_attrs("charge", attrs),
                }

                source: ProteinSource = {
                    "raw": raw,
                    "props": props,
                    "key": key,
                }

                protein = Protein(source=source)
                proteins.append(protein)

        return ProteinList(proteins=proteins)

    def save_as_hdf5(self, path: str):
        with h5py.File(name=path, mode="w") as f:
            f.create_group(self.proteins_dir)
            for protein in self.proteins:
                dataset = f.create_dataset(f"{self.proteins_dir}/{protein.key}", data=protein.representations)
                attrs = dataset.attrs

                attrs["seq"] = protein.seq

                HDF5.set_nullable_attrs("length", protein.props["length"], attrs)
                HDF5.set_nullable_attrs("rt", protein.props["rt"], attrs)
                HDF5.set_nullable_attrs("ccs", protein.props["ccs"], attrs)
                HDF5.set_nullable_attrs("mass", protein.props["mass"], attrs)
                HDF5.set_nullable_attrs("charge", protein.props["charge"], attrs)

    def find_by_key(self, key: str):
        for protein in self._proteins:
            if protein.key == key:
                return protein

        return None

    def set_proteins(self, proteins: list[Protein]):
        self._proteins = proteins
        return self

    def rational_split(self, ratios: list[float]):
        return [
            ProteinList(proteins=proteins) for proteins in Utils.rational_split(target=self._proteins, ratios=ratios)
        ]

    def even_split(self, unit_size: int):
        return [
            ProteinList(proteins=proteins) for proteins in Utils.even_split(target=self._proteins, unit_size=unit_size)
        ]
