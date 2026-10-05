"""Reading and writing AnnData / tables for the command-line pipeline."""
from __future__ import annotations

import json
import os

import anndata as ad
import numpy as np
import pandas as pd


def read_h5ad(path: str) -> ad.AnnData:
    a = ad.read_h5ad(path)
    a.var_names_make_unique()
    return a


def write_table(df: pd.DataFrame, path: str, index: bool = False) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, sep="\t", index=index)
    return path


def write_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return path


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Series, pd.DataFrame)):
        return o.to_dict()
    return str(o)


def sanitize_obs(a: ad.AnnData) -> None:
    """Make obs columns h5ad-serialisable (object columns of mixed types -> str)."""
    for c in a.obs.columns:
        if a.obs[c].dtype == object:
            a.obs[c] = a.obs[c].astype(str)
