# geno-vae

Work-in-progress research code for training a VAE on genotype matrices (PLINK `.raw`) using a memmap-backed pipeline.

This repo currently includes:
- `.raw → memmap` conversion (one-time) and IID tracking
- VAE training with MSE reconstruction on 0/1/2 genotypes
- Regularization options: **KL** or **MMD**
- β-annealing + early stopping
- YAML-driven sweeps with per-run logs, plots, and embedding export (`μ`)

## Why this exists
A baseline VAE on genotype data can strongly reflect ancestry structure. 

> **Note:** Results and conclusions may change as experiments evolve.

## Quickstart

```bash
pip install -r requirements.txt

python scripts/train_vae.py --config configs/example_gacrs_r2p2.yaml
```
