from __future__ import annotations

import torch
from torch import nn

from .config import MertonParams


class VanillaMLPCritic(nn.Module):
    """
    Fully unstructured MLP critic.

    - No CRRA base factor
    - No exact closed-form coefficient initialization
    - Outputs V(w) directly, or V(t, w) when constructed with `time_horizon`.
      In finite-horizon mode the input features are (t / T, log w) and
      `value_and_derivatives` additionally returns V_t.

    By default it uses log-wealth as input because wealth is positive and can
    vary over orders of magnitude. That is an input scaling choice, not an
    assumption on the output form. If you want ultra-vanilla behavior, set
    use_log_input=False.
    """

    def __init__(
        self,
        params: MertonParams,
        hidden_dim: int = 64,
        depth: int = 2,
        activation: str = "tanh",
        use_log_input: bool = True,
        time_horizon: float | None = None,
    ) -> None:
        super().__init__()
        self.params = params
        self.use_log_input = use_log_input
        self.time_horizon = float(time_horizon) if time_horizon is not None else None

        if activation == "tanh":
            act_factory = nn.Tanh
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "gelu":
            act_factory = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: list[nn.Module] = []
        input_dim = 1 if self.time_horizon is None else 2
        last_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_factory())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    @property
    def is_time_aware(self) -> bool:
        return self.time_horizon is not None

    def _features(
        self, wealth: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        w = torch.clamp(wealth, min=1e-12)
        x_w = torch.log(w) if self.use_log_input else w
        if not self.is_time_aware:
            return x_w.unsqueeze(-1)
        if t is None:
            raise ValueError("Critic was built with time_horizon set; pass `t`.")
        return torch.stack([t / self.time_horizon, x_w], dim=-1)

    def value(self, wealth: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(self._features(wealth, t)).squeeze(-1)

    def value_and_derivatives(
        self, wealth: torch.Tensor, t: torch.Tensor | None = None
    ):
        """Returns (V, V_w, V_ww) when stationary, (V, V_t, V_w, V_ww) when finite-horizon."""
        w_in = wealth.clone().detach().requires_grad_(True)
        if not self.is_time_aware:
            V = self.value(w_in)
            ones = torch.ones_like(V)
            Vw = torch.autograd.grad(V, w_in, grad_outputs=ones, create_graph=True)[0]
            Vww = torch.autograd.grad(
                Vw, w_in, grad_outputs=torch.ones_like(Vw), create_graph=True
            )[0]
            return V, Vw, Vww

        if t is None:
            raise ValueError("Critic was built with time_horizon set; pass `t`.")

        t_in = t.clone().detach().requires_grad_(True)
        V = self.value(w_in, t_in)
        ones = torch.ones_like(V)
        Vt = torch.autograd.grad(V, t_in, grad_outputs=ones, create_graph=True)[0]
        Vw = torch.autograd.grad(V, w_in, grad_outputs=ones, create_graph=True)[0]
        Vww = torch.autograd.grad(
            Vw, w_in, grad_outputs=torch.ones_like(Vw), create_graph=True
        )[0]
        return V, Vt, Vw, Vww

    def forward(self, wealth: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.value(wealth, t)
