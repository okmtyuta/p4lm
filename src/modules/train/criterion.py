import numpy.typing as npt
import torch
from scipy import stats


class Criterion:
    def __init__(self):
        self._mse_loss = torch.nn.MSELoss()
        self._l1_loss = torch.nn.L1Loss()

    def mse(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self._mse_loss(output, target)
        return loss

    def l1(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self._l1_loss(output, target)
        return loss

    def pearsonr(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        correlation = stats.pearsonr(output.detach(), target.detach()).correlation
        return torch.Tensor([correlation])

    # def delta(self, output: torch.Tensor, target: torch.Tensor, alpha: float = 0.95):
    #     output_mean = torch.mean(output)
    #     target_mean = torch.mean(target)
    #     output_var = torch.var(output)
    #     target_var = torch.var(target)

    #     diff = output_mean - target_mean
    #     upper = diff + 1.96 * torch.sqrt((output_var / len(output)) + (target_var / len(target)))
    #     lower = diff - 1.96 * torch.sqrt((output_var / len(output)) + (target_var / len(target)))

    #     print(lower, upper)
