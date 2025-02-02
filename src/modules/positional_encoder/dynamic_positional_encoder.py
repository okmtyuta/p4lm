import torch
from torchdiffeq import odeint_adjoint as odeint

from src.modules.data_pipeline.data_pipeline import DataPipe
from src.modules.positional_encoder.dynamics import Dynamics


class DynamicPositionalEncoder(DataPipe):
    def __init__(self, dynamics: Dynamics):
        self._p0 = self._generate_initial_position()
        self._dynamics = dynamics
        self._t_eval = torch.linspace(0, 1, steps=50).to(torch.float32)

    def _act(self, protein):
        length = protein.length
        positions = self._generate_positions()
        encoder = self._sample_positions(positions, length)
        protein.set_piped(encoder * protein.piped)
        return protein

    def _generate_initial_position(self) -> torch.Tensor:
        p0 = torch.rand(1)

        return p0

    def _generate_positions(self) -> torch.Tensor:
        positions: torch.Tensor = odeint(self._dynamics, self._p0, self._t_eval, method="rk4")

        return positions

    def _sample_positions(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        length = positions.size(0)
        indices = torch.linspace(0, length - 1, steps=k).round().long()

        return positions[indices]
