from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def inspect_plink_raw(raw_path: str) -> Tuple[int, int, List[str]]:
    with open(raw_path, "r") as f:
        header = f.readline().strip()
        if not header:
            raise ValueError("Empty .raw file (no header).")

    cols = header.split()
    required = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    if cols[:6] != required:
        raise ValueError(f"Unexpected first 6 columns. Found: {cols[:6]}")

    snp_cols = cols[6:]
    if not snp_cols:
        raise ValueError("No SNP columns found after PHENOTYPE.")

    n_samples = 0
    with open(raw_path, "r") as f:
        _ = f.readline()
        for _ in f:
            n_samples += 1

    return n_samples, len(snp_cols), snp_cols


def convert_plink_raw_to_memmap(
    raw_path: str,
    out_memmap_path: str,
    out_iids_path: str,
    *,
    dtype_out: np.dtype = np.float32,
    chunksize: int = 256,
) -> Tuple[int, int]:
    n_samples, n_snps, snp_cols = inspect_plink_raw(raw_path)
    print(f"[inspect] N={n_samples} samples, D={n_snps} SNPs")

    os.makedirs(os.path.dirname(out_memmap_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_iids_path) or ".", exist_ok=True)

    mm = np.memmap(out_memmap_path, mode="w+", dtype=dtype_out, shape=(n_samples, n_snps))
    iids = np.empty((n_samples,), dtype=object)

    usecols = ["IID"] + snp_cols
    row0 = 0

    for chunk in pd.read_csv(
        raw_path,
        sep=r"\s+",
        usecols=usecols,
        chunksize=chunksize,
        dtype=str,
        engine="c",
    ):
        iid_chunk = chunk["IID"].astype(str).to_numpy()
        n = len(iid_chunk)

        snp_chunk = chunk.drop(columns=["IID"], errors="ignore")
        snp_chunk = snp_chunk.apply(pd.to_numeric, errors="coerce")

        if snp_chunk.isna().values.any():
            raise ValueError(
                "Found NaNs in SNP chunk; expected no missing genotypes. "
                "Add masking/imputation if needed."
            )

        X = snp_chunk.to_numpy(dtype=np.float32, copy=False)
        mm[row0 : row0 + n, :] = X
        iids[row0 : row0 + n] = iid_chunk
        row0 += n

        if row0 % (chunksize * 20) == 0:
            print(f"[convert] wrote {row0}/{n_samples} rows...")

    if row0 != n_samples:
        raise RuntimeError(f"Row count mismatch: wrote {row0}, expected {n_samples}")

    mm.flush()
    np.save(out_iids_path, iids, allow_pickle=True)
    print(f"[OK] memmap: {out_memmap_path}")
    print(f"[OK] iids:   {out_iids_path}")
    return n_samples, n_snps


class GenotypeMemmapDataset(Dataset):
    """Memmap-backed genotype dataset (0/1/2) -> float32 torch tensor."""
    def __init__(
        self,
        memmap_path: str,
        shape: Tuple[int, int],
        *,
        iid_npy_path: Optional[str] = None,
        dtype: np.dtype = np.float32,
        return_iid: bool = False,
    ):
        self.shape = shape
        self.X = np.memmap(memmap_path, mode="r", dtype=dtype, shape=shape)

        self.iids = None
        if iid_npy_path is not None:
            self.iids = np.load(iid_npy_path, allow_pickle=True)
            if len(self.iids) != shape[0]:
                raise ValueError(f"IID length {len(self.iids)} != n_samples {shape[0]}")

        self.return_iid = return_iid
        if self.return_iid and self.iids is None:
            raise ValueError("return_iid=True requires iid_npy_path.")

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int):
        row = np.array(self.X[idx], copy=True)
        x = torch.from_numpy(row)
        if self.return_iid:
            return x, str(self.iids[idx])
        return x
