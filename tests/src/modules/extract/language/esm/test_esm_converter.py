import os

import polars as pl

from src.lib.config.dir import Dir
from src.modules.extract.language.esm.esm_converter import ESMConverter


class TestESMConverter:
    def test_esm2_converter(self) -> None:
        esm2_converter = ESMConverter("esm2")
        df = pl.read_csv(os.path.join(Dir.test_sources_dir, "data.csv"))
        for row in df.iter_rows(named=True):
            length = row["length"]
            seq = row["seq"]
            representations = esm2_converter(seqs=list(seq))
            assert representations.shape == (length, 1280)

    def test_esm1b_converter(self) -> None:
        esm1b_converter = ESMConverter("esm1b")
        df = pl.read_csv(os.path.join(Dir.test_sources_dir, "data.csv"))
        for row in df.iter_rows(named=True):
            length = row["length"]
            seq = row["seq"]
            representations = esm1b_converter(seqs=list(seq))
            assert representations.shape == (length, 1280)
