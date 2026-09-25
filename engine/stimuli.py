"""Stimulation groups and readout for the Shiu et al. model.
Stimulation groups = FlyWire gustatory receptor-neuron types with >= 5 cells; readout = motor neurons, with
MN9 identified from FlyWire labels. Constants: contact Poisson rate 100 Hz (x a per-type multiplier), 0.1-ms step."""
import numpy as np
import pandas as pd

from lifcore import to_index

COMP = "data/shiu/Completeness_783.csv"
CON = "data/shiu/Connectivity_783.parquet"
R_CONTACT, DT_MS = 100.0, 0.1


class Channels:
    def __init__(self):
        ids = pd.read_csv(COMP).iloc[:, 0].values
        cls = pd.read_csv("data/classification.csv.gz")
        ct = pd.read_csv("data/consolidated_cell_types.csv.gz").drop_duplicates("root_id")
        lb = pd.read_csv("data/labels.csv.gz", usecols=["root_id", "label"])
        m = cls.merge(ct, on="root_id", how="left")
        gus = m[(m.super_class == "sensory") & (m["class"] == "gustatory")]
        vc = gus.primary_type.value_counts()
        self.stim_names = sorted(vc[vc >= 5].index)
        self.stim = {t: to_index(gus.root_id[gus.primary_type == t].values, ids)[0]
                     for t in self.stim_names}
        self.modality = {t: gus.sub_class[gus.primary_type == t].mode().iat[0]
                         for t in self.stim_names}
        motor = m[m.super_class == "motor"]
        self.rec, _ = to_index(motor.root_id.values, ids)
        mn9 = set(lb[lb.label.str.contains(r"\bMN9\b", regex=True, na=False)].root_id)
        kept = [r for r in motor.root_id.values if r in set(ids)]
        self.mn9_channels = [j for j, r in enumerate(kept) if r in mn9]
        self.all_stim = np.concatenate(list(self.stim.values()))
