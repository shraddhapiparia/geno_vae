from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(history_csv: str, fig_dir: str) -> None:
    df = pd.read_csv(history_csv)
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train", marker="o", markersize=3)
    plt.plot(df["epoch"], df["val_loss"], label="val", marker="s", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Total loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "curves_total.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_recon"], label="train", marker="o", markersize=3)
    plt.plot(df["epoch"], df["val_recon"], label="val", marker="s", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.title("Reconstruction")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "curves_recon.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_reg"], label="train", marker="o", markersize=3)
    plt.plot(df["epoch"], df["val_reg"], label="val", marker="s", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Reg")
    plt.title("Regularization (KL or MMD)")
    plt.legend(); plttight_layout()
    plt.savefig(os.path.join(fig_dir, "curves_reg.png"), dpi=200)
    plt.close()


def plot_sweep_bars(summary_csv: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)

    def _bar(metric: str, fname: str, title: str):
        d = df.sort_values(metric, ascending=True).copy()
        labels = d["run_id"].tolist()
        vals = d[metric].tolist()

        plt.figure(figsize=(max(8, 0.35 * len(labels)), 4))
        plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel(metric); plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    _bar("test_loss",  "compare_test_total.png", "Test total loss (MSE+reg)")
    _bar("test_recon", "compare_test_recon.png", "Test recon (MSE)")
    _bar("test_reg",   "compare_test_reg.png",   "Test reg term")
