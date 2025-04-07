from typing import TypedDict
from src.modules.data_pipeline.aggregator import Aggregator
from src.modules.data_pipeline.data_pipeline import DataPipeline
from src.modules.data_pipeline.initialize import Initialize
from src.modules.data_pipeline.sinusoidal_positional_encoder import (
    SinusoidalPositionalEncoder,
    BidirectionalSinusoidalPositionalEncoder,
)
from src.modules.dataloader.dataloader import Dataloader, DataloaderState
from src.modules.model.architecture import Architecture
from src.modules.model.configurable_model import ConfigurableModel
from src.modules.protein.protein_list import ProteinList
from src.modules.train.trainer import Trainer

class TrainExperimentConfig(TypedDict):
  protein_list: ProteinList
  data_pipeline: DataPipeline

protein_list = ProteinList.from_hdf5("temp/normalized.h5")


agg = Aggregator("mean")
init = Initialize()
pe = BidirectionalSinusoidalPositionalEncoder(a=1000, b=1, gamma=0)
pipeline = DataPipeline(pipes=[init, pe, agg])
dataloader_state = DataloaderState(
    {
        "protein_list": protein_list,
        "batch_size": 128,
        "input_props": ["mass", "charge", "length"],
        "output_props": ["ccs", "rt"],
        "pipeline": pipeline,
        "cacheable": True,
    }
)
dataloader = Dataloader(state=dataloader_state)


architecture = Architecture(source=(128, 5), input_size=1280 * 2 + 3, output_size=2)
model = ConfigurableModel(architecture=architecture)


trainer = Trainer(config={"data_split_ratio": (0.8, 0.1, 0.1), "epoch": 1000}, model=model, dataloader=dataloader)
trainer.train()
