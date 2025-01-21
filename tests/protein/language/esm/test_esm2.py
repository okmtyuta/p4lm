from src.protein.language.esm.esm2 import ESM2Language
from src.protein.protein import Protein


def test_call():
    esm2_language = ESM2Language()
    proteins = esm2_language([Protein("AAA")])
    assert len(proteins) == 1
    assert len(proteins[0].representations) == 3
