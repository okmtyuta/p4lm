from typing import Literal, Optional, Required, TypedDict

import h5py
import polars as pl
import torch
from tqdm import tqdm

from lib.utils.utils import Utils
from src.modules.hdf.hdf5 import HDF5
from src.modules.protein.exceptions import NotReadablePropException, ProteinRepresentationsUnavailableException

ProteinLanguageName = Literal["esm2", "esm1b"]
protein_language_names: list[ProteinLanguageName] = ["esm2", "esm1b"]

ProteinProp = Literal["ccs", "rt", "mass", "length", "charge"]


class ProteinRaw(TypedDict):
    seq: Required[str]
    representations: Optional[torch.Tensor]


class ProteinProps(TypedDict):
    ccs: Optional[float]
    rt: Optional[float]
    mass: Optional[float]
    charge: Optional[float]
    length: int


class Protein:
    def __init__(self, raw: ProteinRaw, props: ProteinProps, key: str):
        self._raw = raw
        self._representations = raw["representations"]
        self._piped: Optional[torch.Tensor] = None
        self._props = props
        self._key = key

    @property
    def raw(self):
        return self._raw

    @property
    def seq(self):
        return self._raw["seq"]

    @property
    def length(self):
        return self._props["length"]

    @property
    def representations(self):
        representations = self._representations
        if representations is None:
            raise ProteinRepresentationsUnavailableException()

        return representations

    @property
    def key(self):
        return self._key

    @property
    def props(self):
        return self._props

    @property
    def piped(self):
        piped = self._piped
        if piped is None:
            raise Exception

        return piped

    def set_representations(self, representations: torch.Tensor):
        self._representations = representations
        return self

    def set_piped(self, piped: torch.Tensor):
        self._piped = piped
        return self

    def read_prop(self, key: ProteinProp):
        prop = self._props[key]
        if prop is None:
            raise NotReadablePropException(key=key)

        return prop


class ProteinList:
    proteins_dir = "proteins"

    def __init__(self, proteins: list[Protein]):
        self._proteins = proteins

    @property
    def proteins(self):
        return self._proteins

    @property
    def size(self):
        return len(self._proteins)

    @classmethod
    def join(self, protein_lists: list["ProteinList"]):
        proteins: list[Protein] = []
        for protein_list in protein_lists:
            for protein in protein_list.proteins:
                proteins.append(protein)

        return ProteinList(proteins=proteins)

    @classmethod
    def from_dataset_csv(self, path: str):
        df = pl.read_csv(path)

        proteins: list[Protein] = []
        for i, row in enumerate(df.iter_rows(named=True)):
            raw: ProteinRaw = {"seq": row["seq"], "representations": None}
            props: ProteinProps = {
                "ccs": row["ccs"],
                "rt": row["rt"],
                "mass": row["mass"],
                "charge": row["charge"],
                "length": len(row["seq"]),
            }
            protein = Protein(raw=raw, props=props, key=str(i))
            proteins.append(protein)

        return ProteinList(proteins=proteins)

    @classmethod
    def from_hdf(self, path: str):
        with h5py.File(path, mode="r") as f:
            keys = f["proteins"].keys()

            proteins: list[Protein] = []
            for key in tqdm(keys):
                data = f[f"{self.proteins_dir}/{key}"]
                attrs = data.attrs

                raw: ProteinRaw = {"seq": attrs["seq"], "representations": torch.Tensor(data[:])}
                props: ProteinProps = {
                    "ccs": HDF5.read_nullable_attrs("ccs", attrs),
                    "rt": HDF5.read_nullable_attrs("rt", attrs),
                    "mass": HDF5.read_nullable_attrs("mass", attrs),
                    "length": HDF5.read_nullable_attrs("length", attrs),
                    "charge": HDF5.read_nullable_attrs("charge", attrs),
                }

                protein = Protein(raw=raw, props=props, key=key)
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
