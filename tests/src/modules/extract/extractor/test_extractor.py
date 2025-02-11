import os

import pytest

from src.lib.config.dir import Dir
from src.modules.extract.extractor.extractor import Extractor
from src.modules.extract.language.esm.esm1b import ESM1bLanguage
from src.modules.extract.language.esm.esm2 import ESM2Language
from src.modules.extract.language.quick_esm.quick_esm1b import QuickESM1bLanguage
from src.modules.extract.language.quick_esm.quick_esm2 import QuickESM2Language
from src.modules.protein.protein_list import ProteinList


class TestExtractor:
    @pytest.fixture
    def setup(self):
        self.protein_list = ProteinList.from_csv(os.path.join(Dir.test_sources_dir, "data.csv"))

    def test_esm2_extractor(self, setup) -> None:
        esm2_language = ESM2Language()
        extractor = Extractor(language=esm2_language)

        protein_list = extractor(protein_list=self.protein_list, batch_size=16)

        for protein in protein_list.proteins:
            assert protein.representations.shape == (protein.read_props("length"), 1280)

    def test_esm1b_extractor(self, setup) -> None:
        esm1b_language = ESM1bLanguage()
        extractor = Extractor(language=esm1b_language)

        protein_list = extractor(protein_list=self.protein_list, batch_size=16)

        for protein in protein_list.proteins:
            assert protein.representations.shape == (protein.read_props("length"), 1280)

    def test_quick_esm2_extractor(self, setup) -> None:
        quick_esm2_language = QuickESM2Language()
        extractor = Extractor(language=quick_esm2_language)

        protein_list = extractor(protein_list=self.protein_list, batch_size=16)

        for protein in protein_list.proteins:
            assert protein.representations.shape == (protein.read_props("length"), 1280)

    def test_quick_esm1b_extractor(self, setup) -> None:
        quick_esm1b_language = QuickESM1bLanguage()
        extractor = Extractor(language=quick_esm1b_language)

        protein_list = extractor(protein_list=self.protein_list, batch_size=16)

        for protein in protein_list.proteins:
            assert protein.representations.shape == (protein.read_props("length"), 1280)
