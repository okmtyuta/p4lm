import polars as pl

import os

from src.lib.config.dir import Dir

df = pl.read_csv(os.path.join(Dir.root, "data", "ishihama", "data.csv")).with_columns(
    pl.col("seq").str.len_chars().alias("length")
)

df.write_csv(os.path.join(Dir.root, "data", "ishihama", "data.csv"))
