from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class VAEAllSNP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        proj_dim: int = 64,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        latent_dim: int = 16,
        dropout: float = 0.0,
        activation: str = "leakyrelu",
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        act = self._make_activation(activation)

        self.enc = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim, eps=ln_eps),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(proj_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1, eps=ln_eps),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2, eps=ln_eps),
            act,
        )
        self.mu = nn.Linear(hidden_dim2, latent_dim)
        self.logvar = nn.Linear(hidden_dim2, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.LayerNorm(hidden_dim2, eps=ln_eps),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LayerNorm(hidden_dim1, eps=ln_eps),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim1, proj_dim),
            nn.LayerNorm(proj_dim, eps=ln_eps),
            act,
            nn.Linear(proj_dim, input_dim),
        )

        self._init_weights()

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        if name == "leakyrelu":
            return nn.LeakyReLU(0.01, inplace=True)
        raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), min=-5.0, max=5.0)
        return mu, logvar

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar, z
