from __future__ import annotations

import os
import json
import time
import math
import argparse
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from .config import load_yaml, deep_get
from .data import inspect_plink_raw, convert_plink_raw_to_memmap, GenotypeMemmapDataset
from .model import VAEAllSNP
from .metrics import (
    batch_latent_stats,
    beta_anneal,
    make_sweep_from_yaml,
    run_name,
    vae_loss_mse,
)
from .plotting import plot_training_curves, plot_sweep_bars


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    reg_type: str,
    beta: float,
    mmd_mults: Tuple[float, ...],
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    tot_loss = tot_recon = tot_reg = 0.0
    n = 0

    diag_acc = {"mu_abs_mean": 0.0, "mu_std_mean": 0.0, "logvar_mean": 0.0, "logvar_std": 0.0, "z_norm_mean": 0.0}
    diag_batches = 0

    amp_enabled = bool(use_amp and device.type in ("cuda", "mps"))
    amp_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        if amp_enabled:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                xhat, mu, logvar, z = model(x)
                loss, recon, reg = vae_loss_mse(x, xhat, mu, logvar, z, beta, reg_type, mmd_mults)
        else:
            xhat, mu, logvar, z = model(x)
            loss, recon, reg = vae_loss_mse(x, xhat, mu, logvar, z, beta, reg_type, mmd_mults)

        bs = x.size(0)
        tot_loss += float(loss.item()) * bs
        tot_recon += float(recon.item()) * bs
        tot_reg += float(reg.item()) * bs
        n += bs

        d = batch_latent_stats(mu, logvar, z)
        for k in diag_acc:
            diag_acc[k] += d[k]
        diag_batches += 1

    out = {"loss": tot_loss / max(1, n), "recon": tot_recon / max(1, n), "reg": tot_reg / max(1, n)}
    if diag_batches > 0:
        for k in diag_acc:
            out[k] = diag_acc[k] / diag_batches
    return out


def train_one_run(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    reg_type: str,
    beta_max: float,
    warmup_steps: int,
    burnin_epochs: int,
    mmd_mults: Tuple[float, ...],
    out_dir: str,
    use_amp: bool,
    patience: int,
    min_delta: float,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ckpt_best = os.path.join(out_dir, "ckpt_best.pt")
    history_out = os.path.join(out_dir, "history.csv")
    meta_out = os.path.join(out_dir, "meta.json")
    test_out = os.path.join(out_dir, "test_metrics.json")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    amp_enabled = bool(use_amp and device.type in ("cuda", "mps"))
    amp_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val = float("inf")
    best_epoch = None
    patience_counter = 0
    global_step = 0
    rows: List[Dict[str, Any]] = []

    steps_per_epoch = max(1, len(train_loader))
    burnin_steps = burnin_epochs * steps_per_epoch
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = run_recon = run_reg = 0.0
        n_seen = 0
        diag_acc = {"mu_abs_mean": 0.0, "mu_std_mean": 0.0, "logvar_mean": 0.0, "logvar_std": 0.0, "z_norm_mean": 0.0}
        diag_batches = 0

        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            beta = beta_anneal(global_step, warmup_steps, beta_max, burnin_steps=burnin_steps)
            opt.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    xhat, mu, logvar, z = model(x)
                    loss, recon, reg = vae_loss_mse(x, xhat, mu, logvar, z, beta, reg_type, mmd_mults)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
            else:
                xhat, mu, logvar, z = model(x)
                loss, recon, reg = vae_loss_mse(x, xhat, mu, logvar, z, beta, reg_type, mmd_mults)

                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            bs = x.size(0)
            run_loss += float(loss.item()) * bs
            run_recon += float(recon.item()) * bs
            run_reg += float(reg.item()) * bs
            n_seen += bs

            d = batch_latent_stats(mu.detach(), logvar.detach(), z.detach())
            for k in diag_acc:
                diag_acc[k] += d[k]
            diag_batches += 1

            global_step += 1

        beta_eval = beta_anneal(global_step, warmup_steps, beta_max, burnin_steps=burnin_steps)

        train_metrics = {
            "loss": run_loss / max(1, n_seen),
            "recon": run_recon / max(1, n_seen),
            "reg": run_reg / max(1, n_seen),
        }
        if diag_batches > 0:
            for k in diag_acc:
                train_metrics[k] = diag_acc[k] / diag_batches

        val_metrics = eval_epoch(model, val_loader, device, reg_type, beta_eval, mmd_mults, use_amp=use_amp)

        row = {
            "epoch": ep,
            "current_beta": beta_eval,
            "train_loss": train_metrics["loss"],
            "train_recon": train_metrics["recon"],
            "train_reg": train_metrics["reg"],
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon"],
            "val_reg": val_metrics["reg"],
        }
        rows.append(row)

        print(
            f"[epoch {ep:03d}] "
            f"train loss={row['train_loss']:.4f} recon={row['train_recon']:.4f} reg={row['train_reg']:.4f} | "
            f"val loss={row['val_loss']:.4f} recon={row['val_recon']:.4f} reg={row['val_reg']:.4f} | "
            f"β={beta_eval:.3f}"
        )

        improved = row["val_loss"] < (best_val - min_delta)
        if improved:
            best_val = row["val_loss"]
            best_epoch = ep
            patience_counter = 0
            torch.save({"model": model.state_dict()}, ckpt_best)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early-stop] best_epoch={best_epoch} best_val={best_val:.6f}")
                break

    hist_df = pd.DataFrame(rows)
    hist_df.to_csv(history_out, index=False)
    plot_training_curves(history_out, fig_dir)

    if not os.path.exists(ckpt_best):
        raise RuntimeError("Best checkpoint not found; training may have failed early.")
    state = torch.load(ckpt_best, map_location=device)
    model.load_state_dict(state["model"])

    test_metrics = eval_epoch(model, test_loader, device, reg_type, beta_max, mmd_mults, use_amp=use_amp)
    with open(test_out, "w") as f:
        json.dump(test_metrics, f, indent=2)

    meta = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "train_time_sec": float(time.time() - t0),
        "device": str(device),
        "opt": {"lr": lr, "weight_decay": weight_decay},
        "reg": {"type": reg_type, "beta_max": beta_max, "warmup_steps": warmup_steps, "burnin_epochs": burnin_epochs},
        "early_stopping": {"patience": patience, "min_delta": min_delta, "metric": "val_loss"},
    }
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "test_loss": float(test_metrics["loss"]),
        "test_recon": float(test_metrics["recon"]),
        "test_reg": float(test_metrics["reg"]),
        "ckpt_best": ckpt_best,
    }


@torch.no_grad()
def export_embeddings(model, dataset_with_iid, device, out_csv: str, batch_size: int):
    model.eval()
    loader = DataLoader(dataset_with_iid, batch_size=batch_size, shuffle=False, num_workers=0)

    mus = []
    iids = []
    for x, iid in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        mus.append(mu.cpu().numpy())
        iids.extend(list(iid))

    MU = np.vstack(mus)
    cols = [f"z{i+1}" for i in range(MU.shape[1])]
    df = pd.DataFrame(MU, columns=cols)
    df["IID"] = np.array(iids, dtype=object)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[save] embeddings: {out_csv} shape={MU.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    dataset = cfg.get("dataset", "dataset")
    ld_tag = cfg.get("ld_tag", "ld_tag")

    raw = deep_get(cfg, ["paths", "raw"])
    memmap_path = deep_get(cfg, ["paths", "memmap"])
    iids_path = deep_get(cfg, ["paths", "iids"])
    results_root = deep_get(cfg, ["paths", "results_root"])
    if not all([raw, memmap_path, iids_path, results_root]):
        raise ValueError("Config missing required paths: paths.raw/memmap/iids/results_root")

    training = cfg.get("training", {})
    batch_size = args.batch_size if args.batch_size is not None else int(training.get("batch_size", 64))
    epochs = args.epochs if args.epochs is not None else int(training.get("epochs", 20))

    val_frac = float(training.get("val_frac", 0.1))
    test_frac = float(training.get("test_frac", 0.1))
    num_workers = int(training.get("num_workers", 2))
    grad_clip = float(training.get("grad_clip", 0.5))
    dropout = float(training.get("dropout", 0.0))
    seed = int(training.get("seed", 42))
    deterministic = bool(training.get("deterministic", False))
    use_amp = bool(training.get("use_amp", False) or args.use_amp)

    burnin_epochs = int(training.get("burnin_epochs", 10))
    patience = int(training.get("patience", 10))
    min_delta = float(training.get("min_delta", 1e-6))

    io = cfg.get("io", {})
    chunksize = int(io.get("chunksize", 256))
    export_batch_size = int(io.get("export_batch_size", batch_size))

    seed_everything(seed, deterministic=deterministic)

    if not (os.path.exists(memmap_path) and os.path.exists(iids_path)):
        convert_plink_raw_to_memmap(raw, memmap_path, iids_path, chunksize=chunksize)
    else:
        print("[skip] memmap/iids already exist")

    N, D, _ = inspect_plink_raw(raw)
    print(f"[shape] N={N}, D={D}")

    device = torch.device(detect_device())
    print("[device]", device)

    base_dir = os.path.join(results_root, dataset, ld_tag, "mse")
    os.makedirs(base_dir, exist_ok=True)

    full_ds = GenotypeMemmapDataset(memmap_path, (N, D), return_iid=False)

    n_test = max(1, int(N * test_frac))
    n_val = max(1, int(N * val_frac))
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise ValueError("Train fraction too small; adjust val/test fractions.")

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    ds_with_iid = GenotypeMemmapDataset(memmap_path, (N, D), iid_npy_path=iids_path, return_iid=True)

    mmd_mults = tuple(cfg.get("mmd", {}).get("mults", [0.5, 1.0, 2.0, 4.0]))

    sweep = make_sweep_from_yaml(cfg)
    print(f"[sweep] runs={len(sweep)}")

    summary_rows = []
    for item in sweep:
        reg_dir = os.path.join(base_dir, item.reg_type)
        rid = run_name(item, seed=seed)
        out_dir = os.path.join(reg_dir, rid)

        done_flag = os.path.join(out_dir, "test_metrics.json")
        if os.path.exists(done_flag):
            print(f"[skip existing] {item.reg_type}/{rid}")
            continue

        model = VAEAllSNP(
            input_dim=D,
            proj_dim=item.proj_dim,
            hidden_dim1=item.h1,
            hidden_dim2=item.h2,
            latent_dim=item.latent_dim,
            dropout=dropout,
            activation=item.activation,
        ).to(device)

        run_info = train_one_run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=item.lr,
            weight_decay=item.weight_decay,
            grad_clip=grad_clip,
            reg_type=item.reg_type,
            beta_max=item.beta_max,
            warmup_steps=item.warmup_steps,
            burnin_epochs=burnin_epochs,
            mmd_mults=mmd_mults,
            out_dir=out_dir,
            use_amp=use_amp,
            patience=patience,
            min_delta=min_delta,
        )

        state = torch.load(run_info["ckpt_best"], map_location=device)
        model.load_state_dict(state["model"])
        emb_path = os.path.join(out_dir, "embeddings_best.csv")
        export_embeddings(model, ds_with_iid, device, emb_path, batch_size=export_batch_size)

        summary_rows.append({
            "run_id": f"{item.reg_type}/{rid}",
            "reg_type": item.reg_type,
            "latent_dim": item.latent_dim,
            "lr": item.lr,
            "beta_max": item.beta_max,
            "warmup_steps": item.warmup_steps,
            "proj_dim": item.proj_dim,
            "hidden_dim1": item.h1,
            "hidden_dim2": item.h2,
            "weight_decay": item.weight_decay,
            "activation": item.activation,
            "best_epoch": run_info["best_epoch"],
            "best_val_loss": run_info["best_val_loss"],
            "test_loss": run_info["test_loss"],
            "test_recon": run_info["test_recon"],
            "test_reg": run_info["test_reg"],
            "out_dir": out_dir,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(results_root, dataset, ld_tag, "sweep_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    sweep_fig_dir = os.path.join(results_root, dataset, ld_tag, "sweep_figures")
    plot_sweep_bars(summary_csv, sweep_fig_dir)

    print(f"[OK] sweep summary: {summary_csv}")
    print(f"[OK] sweep figures: {sweep_fig_dir}")


if __name__ == "__main__":
    main()
