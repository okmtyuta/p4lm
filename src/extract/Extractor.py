import abc

from tqdm import tqdm

from src.language._language import _Language
from src.protein.protein import Protein, ProteinList
from src.timer.Timer import Timer


class Extractor:
    def __init__(self, csv_path: str, hdf5_path: str, language: _Language, dataset_name: str):
        self._language = language
        self._csv_path = csv_path
        self._hdf5_path = hdf5_path
        self._dataset_name = dataset_name
        self._protein_lists = ProteinList.from_dataset_csv_as_batch(path=csv_path)

        self._timer = Timer()

    @property
    def timer(self):
        return self._timer

    def extract(self):
        self._timer.start("extract")

        for protein_list in tqdm(self._protein_lists):
            self._language(protein_list=protein_list)

        ProteinList.join(protein_lists=self._protein_lists).save_as_hdf(self._hdf5_path)

        self._timer.stop("extract")
