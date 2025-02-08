import os

import h5py
import polars as pl
import pytest
import torch

from src.lib.config.dir import Dir
from src.modules.data.hdf.hdf5 import HDF5
from src.modules.protein.exceptions import ProteinPipedUnavailableException
from src.modules.protein.protein_list import ProteinList


class TestProteinList:
    def test_from_csv(self) -> None:
        path = os.path.join(Dir.test_sources_dir, "data.csv")
        df = pl.read_csv(path)
        protein_list = ProteinList.from_csv(path)

        assert len(protein_list) == len(df)

        for row in df.iter_rows(named=True):
            protein = protein_list.find_by_key(row["index"])

            assert protein.seq == row["seq"]
            assert protein.read_props("ccs") == row.get("ccs")
            assert protein.read_props("rt") == row.get("rt")
            assert protein.read_props("length") == row.get("length")
            assert protein.read_props("charge") == row.get("charge")
            assert protein.read_props("mass") == row.get("mass")

    def test_from_hdf5(self) -> None:
        path = os.path.join(Dir.test_sources_dir, "data.h5")
        protein_list = ProteinList.from_hdf5(os.path.join(Dir.test_sources_dir, "data.h5"))

        with h5py.File(path, mode="r") as f:
            keys = f["proteins"].keys()

            for key in keys:
                data = f[f"{ProteinList.proteins_dir}/{key}"]
                attrs = data.attrs

                protein = protein_list.find_by_key(key)

                assert protein.seq == attrs["seq"]
                assert torch.equal(protein.representations, torch.Tensor(data[:]))
                with pytest.raises(ProteinPipedUnavailableException):
                    protein.piped

                assert protein.read_props("ccs") == HDF5.read_nullable_attrs("ccs", attrs)
                assert protein.read_props("rt") == HDF5.read_nullable_attrs("rt", attrs)
                assert protein.read_props("mass") == HDF5.read_nullable_attrs("mass", attrs)
                assert protein.read_props("length") == HDF5.read_nullable_attrs("length", attrs)
                assert protein.read_props("charge") == HDF5.read_nullable_attrs("charge", attrs)
