from typing import Literal, Optional, Required, TypedDict

import h5py
import polars as pl
import torch
from tqdm import tqdm

ProteinLanguageName = Literal["esm2", "esm1b"]
protein_language_names: list[ProteinLanguageName] = ["esm2", "esm1b"]

ProteinProp = Literal["ccs", "rt", "mass", "length"]


class ProteinRaw(TypedDict):
    seq: Required[str]
    representations: Optional[torch.Tensor]


class ProteinProps(TypedDict):
    ccs: Optional[float]
    rt: Optional[float]
    mass: Optional[float]
    length: int


class NullableHDF:
    @staticmethod
    def set_nullable_attrs[T](key: str, value: Optional[T], attrs: h5py.AttributeManager):
        if value is None:
            attrs[key] = "None"
        else:
            attrs[key] = value

    @staticmethod
    def read_nullable_attrs(key: str, attrs: h5py.AttributeManager):
        if attrs[key] == "None":
            return None

        return attrs[key]


class Protein:
    def __init__(self, raw: ProteinRaw, props: ProteinProps, key: str):
        self._raw = raw
        self._representations = raw["representations"]
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
        _representations = self._representations
        if _representations is None:
            raise Exception

        return _representations

    @property
    def key(self):
        return self._key

    @property
    def props(self):
        return self._props

    def set_representations(self, representations: torch.Tensor):
        self._representations = representations


class ProteinList:
    proteins_dir = "proteins"

    def __init__(self, proteins: list[Protein]):
        self._proteins = proteins

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
    def from_dataset_csv_as_batch(self, path: str, size=32):
        df = pl.read_csv(path)

        protein_lists: list[ProteinList] = []
        for i in range(0, len(df), size):
            proteins: list[Protein] = []
            for j, row in enumerate(df[i : i + size].iter_rows(named=True)):
                raw: ProteinRaw = {"seq": row["seq"], "representations": None}
                props: ProteinProps = {
                    "ccs": row["ccs"],
                    "rt": row["rt"],
                    "mass": row["mass"],
                    "length": len(row["seq"]),
                }
                protein = Protein(raw=raw, props=props, key=f"{i}-{j}")
                proteins.append(protein)

            protein_list = ProteinList(proteins=proteins)
            protein_lists.append(protein_list)

        return protein_lists

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
                    "ccs": NullableHDF.read_nullable_attrs("ccs", attrs),
                    "rt": NullableHDF.read_nullable_attrs("rt", attrs),
                    "mass": NullableHDF.read_nullable_attrs("mass", attrs),
                    "length": NullableHDF.read_nullable_attrs("length", attrs),
                }

                protein = Protein(raw=raw, props=props, key=key)
                proteins.append(protein)

        return ProteinList(proteins=proteins)

    def save_as_hdf(self, path: str):
        with h5py.File(path, mode="w") as f:
            f.create_group(self.proteins_dir)
            for protein in self.proteins:
                dataset = f.create_dataset(
                    f"{self.proteins_dir}/{protein.key}", data=protein.representations, dtype="f"
                )
                attrs = dataset.attrs

                attrs["seq"] = protein.seq

                NullableHDF.set_nullable_attrs("length", protein.props["length"], attrs)
                NullableHDF.set_nullable_attrs("rt", protein.props["rt"], attrs)
                NullableHDF.set_nullable_attrs("ccs", protein.props["ccs"], attrs)
                NullableHDF.set_nullable_attrs("mass", protein.props["mass"], attrs)
