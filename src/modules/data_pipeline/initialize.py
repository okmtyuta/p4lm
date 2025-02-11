from src.modules.data_pipeline.data_pipeline import DataPipe
from src.modules.protein.protein import Protein


class Initialize(DataPipe):
    def _act(self, protein: Protein):
        return protein.set_piped(piped=protein.representations)
