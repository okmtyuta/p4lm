import abc

from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


class DataPipe(metaclass=abc.ABCMeta):
    def __call__(self, protein_list: ProteinList) -> ProteinList:
        proteins = [self._act(protein=protein) for protein in protein_list.proteins]
        return protein_list.set_proteins(proteins=proteins)

    @abc.abstractmethod
    def _act(self, protein: Protein) -> Protein:
        raise NotImplementedError


class DataPipeline:
    def __init__(self, pipes: list[DataPipe]):
        self._pipes = pipes

    def __call__(self, protein_list: ProteinList):
        for pipe in self._pipes:
            protein_list = pipe(protein_list=protein_list)

        return protein_list
