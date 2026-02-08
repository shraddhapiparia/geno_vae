from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-5.0, max=5.0)
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True).t()
    return torch.clamp(a2 + b2 - 2.0 * (a @ b.t()), min=0.0)


def mmd_rbf_median(z: torch.Tensor, z_prior: torch.Tensor, mults=(0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    with torch.no_grad():
        d = pairwise_sq_dists(z, z)
        n = d.size(0)
        if n > 1:
            med = d[~torch.eye(n, dtype=torch.bool, device=d.device)].median()
        else:
            med = d.median()
        med = torch.clamp(med, min=1e-6)

    sigmas2 = [med * (m ** 2) for m in mults]
    zz = pairwise_sq_dists(z, z)
    pp = pairwise_sq_dists(z_prior, z_prior)
    zp = pairwise_sq_dists(z, z_prior)

    Kzz = 0.0; Kpp = 0.0; Kzp = 0.0
    for s2 in sigmas2:
        gamma = 1.0 / (2.0 * s2)
        Kzz += torch.exp(-gamma * zz)
        Kpp += torch.exp(-gamma * pp)
        Kzp += torch.exp(-gamma * zp)

    n = z.size(0)
    if n > 1:
        Kzz = (Kzz.sum() - Kzz.diag().sum()) / (n * (n - 1))
        Kpp = (Kpp.sum() - Kpp.diag().sum()) / (n * (n - 1))
    else:
        Kzz = Kzz.mean()
        Kpp = Kpp.mean()

    Kzp = Kzp.mean()
    return Kzz + Kpp - 2.0 * Kzp


def beta_anneal(step: int, warmup_steps: int, beta_max: float, burnin_steps: int = 0) -> float:
    if beta_max <= 0:
        return 0.0
    if burnin_steps > 0 and step < burnin_steps:
        return 0.0
    if warmup_steps <= 0:
        return float(beta_max)
    t = step - burnin_steps
    if t <= 0:
        return 0.0
    return float(beta_max) * min(1.0, t / warmup_steps)


def batch_latent_stats(mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor) -> Dict[str, float]:
    return {
        "mu_abs_mean": float(mu.abs().mean().item()),
        "mu_std_mean": float(mu.std(dim=0).mean().item()),
        "logvar_mean": float(logvar.mean().item()),
        "logvar_std": float(logvar.std().item()),
        "z_norm_mean": float(z.norm(dim=1).mean().item()),
    }


def vae_loss_mse(
    x: torch.Tensor,
    xhat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    beta: float,
    reg_type: str,
    mmd_mults=(0.5, 1.0, 2.0, 4.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.mse_loss(xhat, x, reduction="mean")

    if beta <= 0.0:
        reg = torch.zeros((), device=x.device, dtype=x.dtype)
    else:
        if reg_type == "kl":
            reg = kl_divergence(mu, logvar)
        elif reg_type == "mmd":
            z_prior = torch.randn_like(z)
            reg = mmd_rbf_median(z, z_prior, mults=mmd_mults)
        else:
            raise ValueError("reg_type must be 'kl' or 'mmd'")

    loss = recon + beta * reg
    return loss, recon, reg


@dataclass(frozen=True)
class SweepItem:
    reg_type: str
    latent_dim: int
    lr: float
    beta_max: float
    warmup_steps: int
    proj_dim: int = 64
    h1: int = 64
    h2: int = 32
    weight_decay: float = 1e-5
    activation: str = "leakyrelu"


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def make_sweep_from_yaml(cfg: Dict[str, Any]) -> List[SweepItem]:
    sw = cfg.get("sweep", {})
    if not isinstance(sw, dict) or not sw:
        raise ValueError("Missing or empty 'sweep' section in YAML config.")

    reg_types   = [str(r).lower() for r in _as_list(sw.get("reg_type", ["mmd", "kl"]))]
    latent_dims = [int(z) for z in _as_list(sw.get("latent_dim", [16]))]
    lrs         = [float(lr) for lr in _as_list(sw.get("lr", [3e-4]))]
    warmups     = [int(w) for w in _as_list(sw.get("warmup_steps", [200]))]

    proj_dims   = [int(p) for p in _as_list(sw.get("proj_dim", [64]))]
    h1s         = [int(h) for h in _as_list(sw.get("h1", [64]))]
    h2s         = [int(h) for h in _as_list(sw.get("h2", [32]))]
    weight_decays = [float(wd) for wd in _as_list(sw.get("weight_decay", [1e-5]))]
    activations = [str(a) for a in _as_list(sw.get("activation", ["leakyrelu"]))]

    beta_block = sw.get("beta_max", {})
    if not isinstance(beta_block, dict):
        raise ValueError("sweep.beta_max must be a dict with keys 'kl' and/or 'mmd'.")

    for rt in reg_types:
        if rt not in ("kl", "mmd"):
            raise ValueError(f"Unknown reg_type: {rt}")
        if rt not in beta_block:
            raise ValueError(f"sweep.beta_max missing key for reg_type='{rt}' (add beta_max.{rt}: [...])")

    items: List[SweepItem] = []
    for rt in reg_types:
        betas = [float(b) for b in _as_list(beta_block[rt])]
        for z in latent_dims:
            for lr in lrs:
                for w in warmups:
                    for b in betas:
                        for p in proj_dims:
                            for h1 in h1s:
                                for h2 in h2s:
                                    for wd in weight_decays:
                                        for act in activations:
                                            items.append(SweepItem(
                                                reg_type=rt, latent_dim=z, lr=lr, beta_max=b, warmup_steps=w,
                                                proj_dim=p, h1=h1, h2=h2, weight_decay=wd, activation=act
                                            ))
    return items


def run_name(item: SweepItem, seed: int) -> str:
    lr_s = f"{item.lr:.0e}".replace("+0", "").replace("+", "")
    wd_s = f"{item.weight_decay:.0e}".replace("+0", "").replace("+", "")
    b_s  = f"{item.beta_max:g}"
    act  = item.activation.lower()
    return (
        f"z{item.latent_dim:02d}_lr{lr_s}_b{b_s}_w{item.warmup_steps}"
        f"_p{item.proj_dim}_h{item.h1}_h{item.h2}_wd{wd_s}_{act}_seed{seed}"
    )
