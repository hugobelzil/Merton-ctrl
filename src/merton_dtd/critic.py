from __future__ import annotations

import torch
from torch import nn

from .config import MertonParams


class ScalarCRRACritic(nn.Module):
    """
    Structured critic with the exact wealth homogeneity built in.

    V(w) = A * w^(1-gamma) / (1-gamma), with A = exp(log_A).

    This is the exact form of the fixed-policy value function in the
    infinite-horizon CRRA Merton problem.
    """

    def __init__(self, params: MertonParams, init_log_A: float = 0.0) -> None:
        super().__init__()
        self.params = params
        self.log_A = nn.Parameter(torch.tensor(float(init_log_A), dtype=torch.float32))

    def value(self, wealth: torch.Tensor) -> torch.Tensor:
        g = self.params.gamma
        w = torch.clamp(wealth, min=1e-12)
        A = torch.exp(self.log_A)
        return A * torch.pow(w, 1.0 - g) / (1.0 - g)

    def value_and_derivatives(self, wealth: torch.Tensor):
        g = self.params.gamma
        w = torch.clamp(wealth, min=1e-12)
        A = torch.exp(self.log_A)

        V = A * torch.pow(w, 1.0 - g) / (1.0 - g)
        Vw = A * torch.pow(w, -g)
        Vww = -g * A * torch.pow(w, -g - 1.0)
        return V, Vw, Vww

    def forward(self, wealth: torch.Tensor) -> torch.Tensor:
        return self.value(wealth)


class MLPCRRACritic(nn.Module):
    """
    Structured MLP critic that preserves the CRRA sign and approximate scaling.

    V(w) = base_gamma(w) * exp(mlp(log w))
    where base_gamma(w) = w^(1-gamma)/(1-gamma).

    If the mlp output is constant, this reduces to the exact CRRA form.
    """

    def __init__(
        self,
        params: MertonParams,
        hidden_dim: int = 64,
        depth: int = 2,
        init_log_A: float = 0.0,
    ) -> None:
        super().__init__()
        self.params = params

        layers: list[nn.Module] = []
        input_dim = 1
        last_dim = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim

        final = nn.Linear(last_dim, 1)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, float(init_log_A))
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def _raw_multiplier(self, wealth: torch.Tensor) -> torch.Tensor:
        x = torch.log(torch.clamp(wealth, min=1e-12)).unsqueeze(-1)
        return self.net(x).squeeze(-1)

    def value(self, wealth: torch.Tensor) -> torch.Tensor:
        g = self.params.gamma
        w = torch.clamp(wealth, min=1e-12)
        base = torch.pow(w, 1.0 - g) / (1.0 - g)
        multiplier = torch.exp(self._raw_multiplier(w))
        return base * multiplier

    def value_and_derivatives(self, wealth: torch.Tensor):
        w = wealth.clone().detach().requires_grad_(True)
        V = self.value(w)
        ones = torch.ones_like(V)
        Vw = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=True)[0]
        Vww = torch.autograd.grad(
            Vw,
            w,
            grad_outputs=torch.ones_like(Vw),
            create_graph=True,
        )[0]
        return V, Vw, Vww

    def forward(self, wealth: torch.Tensor) -> torch.Tensor:
        return self.value(wealth)


class VanillaMLPCritic(nn.Module):
    """
    Fully unstructured MLP critic.

    - No CRRA base factor
    - No exact closed-form coefficient initialization
    - Outputs V(w) directly

    By default it uses log-wealth as input because wealth is positive and can
    vary over orders of magnitude. That is an input scaling choice, not an
    assumption on the output form.

    If you want ultra-vanilla behavior, set use_log_input=False.
    """

    def __init__(
        self,
        params: MertonParams,
        hidden_dim: int = 64,
        depth: int = 2,
        activation: str = "tanh",
        use_log_input: bool = True,
    ) -> None:
        super().__init__()
        self.params = params
        self.use_log_input = use_log_input

        if activation == "tanh":
            act_factory = nn.Tanh
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "gelu":
            act_factory = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: list[nn.Module] = []
        input_dim = 1
        last_dim = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_factory())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def _features(self, wealth: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(wealth, min=1e-12)
        if self.use_log_input:
            x = torch.log(w)
        else:
            x = w
        return x.unsqueeze(-1)

    def value(self, wealth: torch.Tensor) -> torch.Tensor:
        x = self._features(wealth)
        return self.net(x).squeeze(-1)

    def value_and_derivatives(self, wealth: torch.Tensor):
        w = wealth.clone().detach().requires_grad_(True)
        V = self.value(w)
        ones = torch.ones_like(V)
        Vw = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=True)[0]
        Vww = torch.autograd.grad(
            Vw,
            w,
            grad_outputs=torch.ones_like(Vw),
            create_graph=True,
        )[0]
        return V, Vw, Vww

    def forward(self, wealth: torch.Tensor) -> torch.Tensor:
        return self.value(wealth)