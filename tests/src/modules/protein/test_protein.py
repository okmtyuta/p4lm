import os

import polars as pl
import pytest
import torch

from src.lib.config.dir import Dir
from src.modules.protein.exceptions import (
    ProteinPipedUnavailableException,
    ProteinPropsUnreadableException,
    ProteinRepresentationsUnavailableException,
)
from src.modules.protein.protein import Protein, ProteinSource, protein_prop_names


class TestProtein:
    def test_init(self) -> None:
        df = pl.read_csv(os.path.join(Dir.test_sources_dir, "data.csv"))
        for row in df.iter_rows(named=True):
            representations = torch.Tensor([0, 0, 0])
            piped = torch.Tensor([1, 1, 1])

            source: ProteinSource = {
                "raw": {"seq": row["seq"], "representations": representations, "piped": piped},
                "props": {
                    "ccs": row.get("ccs"),
                    "rt": row.get("rt"),
                    "length": row.get("length"),
                    "charge": row.get("charge"),
                    "mass": row.get("mass"),
                },
                "key": row["index"],
            }
            protein = Protein(source=source)

            assert protein.seq == source["raw"]["seq"]
            assert protein.read_props("ccs") == source["props"]["ccs"]
            assert protein.read_props("rt") == source["props"]["rt"]
            assert protein.read_props("length") == source["props"]["length"]
            assert protein.read_props("charge") == source["props"]["charge"]
            assert protein.read_props("mass") == source["props"]["mass"]
            assert protein.key == row["index"]

            assert torch.equal(protein.representations, representations)
            assert torch.equal(protein.piped, piped)

    def test_raise_exception(self) -> None:
        source: ProteinSource = {
            "raw": {"seq": "DSGSDALRSGLTVPTSPKGRLL", "representations": None, "piped": None},
            "props": {"ccs": None, "rt": None, "length": None, "charge": None, "mass": None},
            "key": "test",
        }
        protein = Protein(source=source)
        for name in protein_prop_names:
            with pytest.raises(ProteinPropsUnreadableException):
                protein.read_props(name)
        with pytest.raises(ProteinRepresentationsUnavailableException):
            protein.representations
        with pytest.raises(ProteinPipedUnavailableException):
            protein.piped
