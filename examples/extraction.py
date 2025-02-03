import os

from src.modules.extractor.Extractor import Extractor
from src.modules.language.esm.esm2 import ESM2Language
from src.modules.protein.protein import ProteinList

# designate the language you want to use for extraction
language = ESM2Language()
# set the language to extractor
extractor = Extractor(language=language)

# read proteins from csv
protein_list = ProteinList.from_dataset_csv(
    path=os.path.join(os.path.dirname(__file__), "src", "data", "dataset", "ishihama", "data.csv")
)

# execute extraction
extractor(protein_list=protein_list, batch_size=32)

# save extracted features as HDF5 file named `data.h5`
protein_list.save_as_hdf5("data.h5")
