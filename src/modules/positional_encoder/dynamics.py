import torch


class Dynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
        )

    def forward(self, t: torch.Tensor, p: torch.Tensor):
        input = torch.cat([p, t.unsqueeze(0)])
        return self.seq(input)
