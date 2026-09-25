"""Connectome loading helpers for the Shiu et al. model (FlyWire v783)."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from lif import W_SYN


def make_W(n, src, dst, w):
    """W[post, pre] = w_syn x signed synapse count (float32, as in the original re-implementation)."""
    return csr_matrix(((W_SYN * np.asarray(w, np.float64)).astype(np.float32),
                       (np.asarray(dst), np.asarray(src))), shape=(n, n))


def taste_root_ids():
    """Root ids of sugar/water GRNs, bitter GRNs and MN9 (FlyWire annotations)."""
    cls = pd.read_csv("data/classification.csv.gz",
                      usecols=["root_id", "super_class", "class", "sub_class"])
    lb = pd.read_csv("data/labels.csv.gz", usecols=["root_id", "label"])
    gus = (cls.super_class == "sensory") & (cls["class"] == "gustatory")
    sugar = cls.root_id[gus & (cls.sub_class == "sugar/water")].values
    bitter = cls.root_id[gus & (cls.sub_class == "bitter")].values
    mn9 = lb[lb.label.str.contains(r"\bMN9\b", regex=True, na=False)].root_id.unique()
    return sugar, bitter, mn9


def load_shiu():
    """Shiu et al.'s v783 graph: all connections (no 5-synapse threshold) and their sign column.
    Returns n, src, dst, w, root_ids (index order)."""
    df = pd.read_parquet("data/shiu/Connectivity_783.parquet",
                         columns=["Presynaptic_Index", "Postsynaptic_Index",
                                  "Excitatory x Connectivity"])
    ids = pd.read_csv("data/shiu/Completeness_783.csv").iloc[:, 0].values
    return (len(ids), df.Presynaptic_Index.values, df.Postsynaptic_Index.values,
            df["Excitatory x Connectivity"].values, ids)


def to_index(root_ids, all_ids):
    pos = pd.Series(np.arange(len(all_ids)), index=all_ids)
    keep = [r for r in root_ids if r in pos.index]
    return pos.loc[keep].values, len(root_ids) - len(keep)
