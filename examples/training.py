import torch
from schedulefree import RAdamScheduleFree
from scipy.stats import pearsonr
from tqdm import tqdm

from src.modules.data_pipeline.aggregator import Aggregator
from src.modules.data_pipeline.data_pipeline import DataPipeline
from src.modules.data_pipeline.initial_pipe import InitialPipe
from src.modules.data_pipeline.trigonometric_positional_encoder import (
    TrigonometricPositionalEncoder,
)
from src.modules.dataloader.dataloader import Dataloader
from src.modules.model.architecture import Architecture
from src.modules.model.configurable_model import ConfigurableModel
from src.modules.protein.protein import ProteinList

TARGETS = [{"name": "rt", "weight": 1}, {"name": "ccs", "weight": 1}]

protein_list = ProteinList.from_hdf("ishihama-100.h5")

positional_encoder = TrigonometricPositionalEncoder(a=1000, b=1, gamma=0)

aggregator = Aggregator("mean")
init = InitialPipe()
pipeline = DataPipeline(pipes=[init, positional_encoder, aggregator])
dataloader = Dataloader(
    protein_list=protein_list,
    batch_size=32,
    input_props=["length"],
    output_props=["rt", "ccs"],
    pipeline=pipeline,
    cacheable=True,
)


architecture = Architecture(source=(128, 5), input_size=1280 + 1, output_size=2)
model = ConfigurableModel(architecture=architecture)
model.train()

criterion = torch.nn.MSELoss()
optimizer = RAdamScheduleFree(model.parameters(), lr=0.01)

optimizer.train()

train_loader, evaluate_loader = dataloader.rational_split([0.8, 0.2])


for i in range(3000):
    for j, batch in tqdm(enumerate(train_loader.batches)):
        optimizer.zero_grad()

        input, label, protein = batch.use()

        output = model(input)

        loss = None

        for ith, target in enumerate(TARGETS):
            weighted = target["weight"] * criterion(output[:, ith], label[:, ith])
            if loss is None:
                loss = weighted
            else:
                loss += weighted

        if loss is None:
            raise Exception

        loss.backward()
        optimizer.step()

    print(f"({i}) END")

    if i % 1 == 0:
        outputs: dict = {}
        labels: dict = {}
        for target in TARGETS:
            outputs[target["name"]] = []
            labels[target["name"]] = []

        for batch in tqdm(evaluate_loader.batches):
            input, label, protein = batch.use()

            output = model(input)

            for ith, target in enumerate(TARGETS):
                outputs[target["name"]] += output[:, ith].tolist()
                labels[target["name"]] += label[:, ith].tolist()

        for target in TARGETS:
            print(
                f'{i}: pearsonr of {target["name"]}: {pearsonr(outputs[target["name"]], labels[target["name"]]).correlation}'
            )
