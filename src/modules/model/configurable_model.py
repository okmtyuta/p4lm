import torch
from schedulefree import RAdamScheduleFree

from src.modules.model.architecture import Architecture


class ConfigurableModel(torch.nn.Module):
    def __init__(self, architecture: Architecture) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential(*architecture.raw_layers)

        self._optimizer = RAdamScheduleFree(self.parameters(), lr=1e-3)

    @property
    def optimizer(self):
        return self._optimizer

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return super().__call__(input)

    def forward(self, x):
        return self.layers(x)
