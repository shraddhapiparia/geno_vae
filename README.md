# geno-vae

Work-in-progress research code for training a VAE on genotype matrices (PLINK `.raw`) using a memmap-backed pipeline.

This repo currently includes:
- Efficient genotype handling via memory-mapped arrays
- VAE models with KL or MMD regularization
- β-annealing schedules and early stopping
- YAML-driven hyperparameter sweeps
- Exporting latent embeddings for downstream analysis

## Why this exists
A baseline VAE on genotype data can strongly reflect ancestry structure. 

> **Note:** Results and conclusions may change as experiments evolve.

## Quickstart

```bash
git clone https://github.com/shraddhapiparia/geno-vae.git
cd geno-vae
pip install -e .
python -m geno_vae.train --config configs/example.yaml
```
