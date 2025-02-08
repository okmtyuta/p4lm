from tqdm import tqdm

from src.modules.language._language import _Language
from src.modules.protein.protein import ProteinList


class Extractor:
    def __init__(self, language: _Language):
        self._language = language

    def __call__(self, protein_list: ProteinList, batch_size: int):
        protein_lists = protein_list.even_split(unit_size=batch_size)

        for protein_list in tqdm(protein_lists):
            self._language(protein_list=protein_list)

        return ProteinList.join(protein_lists=protein_lists)
