import json
import os

import torch
from tqdm import tqdm

from src.lib.config.dir import Dir
from src.lib.utils.utils import Utils
from src.modules.data_pipeline.aggregator import Aggregator
from src.modules.data_pipeline.data_pipeline import DataPipeline
from src.modules.data_pipeline.initialize import Initialize
from src.modules.dataloader.dataloader import DataBatch, Dataloader, DataloaderState
from src.modules.model.architecture import Architecture
from src.modules.model.configurable_model import ConfigurableModel
from src.modules.protein.protein_list import ProteinList
from src.modules.train.criterion import Criterion

# dynamics = Dynamics()
# dynamics.train()

# class Trainer:
#     def __init__(self, dataloader: Dataloader, model: ConfigurableModel):
#         self._dataloader = dataloader
#         self._model = model

#     def _epoch_train(self):
#         pass

# protein_list = ProteinList.from_hdf5("ishihama-100.h5")
protein_list = ProteinList.from_hdf5("normalized.h5")

# dynamic_positional_encoder = DynamicPositionalEncoder(dynamics=dynamics)


agg = Aggregator("mean")
init = Initialize()
pipeline = DataPipeline(pipes=[init, agg])
dataloader_state = DataloaderState(
    {
        "protein_list": protein_list,
        "batch_size": 128,
        "input_props": ["mass", "charge", "length"],
        "output_props": ["ccs"],
        "pipeline": pipeline,
        "cacheable": True,
    }
)
dataloader = Dataloader(state=dataloader_state)


architecture = Architecture(source=(128, 5), input_size=1280 + 3, output_size=1)
model = ConfigurableModel(architecture=architecture)
# model.train()

# criterion = torch.nn.MSELoss()
# model_optimizer = RAdamScheduleFree(model.parameters(), lr=0.01)
# # dynamics_optimizer = RAdamScheduleFree(dynamics.parameters(), lr=0.1)

# model_optimizer.train()
# # dynamics_optimizer.train()

# train_loader, evaluate_loader = dataloader.rational_split([0.8, 0.2])


class Trainer:
    def __init__(self, model: ConfigurableModel, dataloader: Dataloader):
        self._model = model
        self._train_loader, self._evaluate_loader, self._validate_loader = dataloader.rational_split([0.8, 0.1, 0.1])
        self._criterion = Criterion()

        self._model.train()
        self._model.optimizer.train()

    def _batch_train(self, batch: DataBatch):
        self._model.optimizer.zero_grad()
        input, label, protein = batch.use()

        output = model(input=input)
        # losses: list[torch.Tensor] = []
        # for i in range(len(batch.output_props)):
        #     loss = self._criterion.mse(output[:, i], label[:, i])
        #     losses.append(loss)

        # loss_items = [float(loss.item()) for loss in losses]
        # weights = torch.ones(len(batch.output_props)) - torch.Tensor(Utils.normalize(loss_items))

        # loss = torch.sum(weights * torch.stack(losses))
        loss = self._criterion.mse(output, label)
        loss.backward()
        self._model.optimizer.step()

    def _batch_evaluate(self, batch: DataBatch):
        self._model.optimizer.zero_grad()
        input, label, protein = batch.use()

        output = model(input=input)

        loss = self._criterion.mse(output=output, target=label)
        pearsonr = self._criterion.pearsonr(output=output, target=label)

        print(f"loss is {loss} and pearsonr is {pearsonr}")

    def _epoch_train(self):
        for batch in self._train_loader.batches:
            self._batch_train(batch=batch)

    def _evaluate(self):
        labels: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        for batch in self._train_loader.batches:
            self._model.optimizer.zero_grad()
            input, label, protein = batch.use()

            output = model(input=input)

            # loss = self._criterion.mse(output=output, target=label)

            labels.append(label.squeeze(dim=1))
            outputs.append(output.squeeze(dim=1))

        pearsonrs = self._criterion.pearsonr(torch.cat(labels), torch.cat(outputs))
        return pearsonrs

    def train(self) -> None:
        pearsonrs_list: list[torch.Tensor] = []
        for i in tqdm(range(1000)):
            self._epoch_train()
            pearsonrs = self._evaluate()
            print(pearsonrs)
            pearsonrs_list.append(pearsonrs)

            if i % 10 == 0:
                with open(os.path.join(Dir.root, "ccs.json"), mode="w") as f:
                    f.write(json.dumps(torch.stack(pearsonrs_list).tolist()))


trainer = Trainer(model=model, dataloader=dataloader)
trainer.train()
# for i in range(3000):
#     # outputs = []
#     # labels = []
#     for j, batch in tqdm(enumerate(train_loader.batches)):
#         model_optimizer.zero_grad()
#         # dynamics_optimizer.zero_grad()
#         # dynamics_opt.zero_grad()
#         # pred = positional_encoder._generate_positions()

#         # loss = criterion(pred, p)
#         # print(loss)

#         input, label, protein = batch.use()

#         output = model(input)

#         # full = None

#         # loss = criterion(output[:, 0], label[:, 0]) + criterion(output[:, 1], label[:, 1])
#         # loss = criterion(output[:, 0], label[:, 0])
#         loss = criterion(output, label)

#         # outputs += output.squeeze(dim=1).tolist()
#         # labels += label.squeeze(dim=1).tolist()

#         # loss = criterion(output, label)
#         # print(list(dynamics.parameters())[0])

#         loss.backward()
#         model_optimizer.step()
#         # dynamics_optimizer.step()
#         # dynamics_opt.step()
#         # print(positional_encoder._generate_positions())
#         # print("@@@@@@")
#         # print(
#         #     f"{i}, {j}: {torch.sqrt(loss)} pearsonr: {pearsonr(output.squeeze().tolist(), label.squeeze().tolist()).correlation}"
#         # )
#         # print("@@@@@@")
#         # print("positions: ", positional_encoder._generate_positions().squeeze(dim=1).tolist())
#         # print(
#         #     f"({i}, {j}) END: {torch.sqrt(loss)} pearsonr: {pearsonr(output.squeeze().tolist(), label.squeeze().tolist()).correlation}"
#         # )
#     print(f"({i}) END")
#     # print(f"{i} end")

#     if i % 1 == 0:
#         ort = []
#         occs = []

#         lrt = []
#         lccs = []
#         for batch in tqdm(evaluate_loader.batches):
#             input, label, protein = batch.use()

#             output = model(input)

#             # ort += output[:, 0].tolist()
#             # lrt += label[:, 0].tolist()
#             # occs += output[:, 1].tolist()
#             # lccs += label[:, 1].tolist()
#             # occs += output[:, 1].tolist()
#             # lccs += label[:, 1].tolist()
#             occs += output.squeeze(dim=1).tolist()
#             lccs += label.squeeze(dim=1).tolist()

#         # print(f"{i}: pearsonr of rt: {pearsonr(ort, lrt).correlation}")
#         print(f"{i}: pearsonr of ccs: {pearsonr(occs, lccs).correlation}")
