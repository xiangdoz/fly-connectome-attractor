"""E9 (logged before running): do other validation-type stimuli ignite the self-sustained state?
Stimulus sets (all cells together): all bitter GRNs, JO-C + JO-E, JO-F, all low-salt GRNs, all gustatory neurons;
100 and 200 Hz for 1 s, then 1 s silence; fresh seeds 8271-8273. Ignited = > 1,000 active (> 1 Hz) in the last 250 ms."""
import time

import pandas as pd

from common import np, save
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate

SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
SEEDS, RATES = (8271, 8272, 8273), (100, 200)


def main():
    t0 = time.time()
    ch = Channels()
    ids = pd.read_csv(COMP).iloc[:, 0].values
    cls = pd.read_csv("data/classification.csv.gz")
    ct = pd.read_csv("data/consolidated_cell_types.csv.gz").drop_duplicates("root_id")
    m = cls.merge(ct, on="root_id", how="left")
    gus = (m.super_class == "sensory") & (m["class"] == "gustatory")
    sets = {"ALL_BITTER": m.root_id[gus & (m.sub_class == "bitter")].values,
            "JO_CE": m.root_id[m.primary_type.isin(["JO-C", "JO-E"])].values,
            "JO_F": m.root_id[m.primary_type == "JO-F"].values,
            "ALL_LOWSALT": m.root_id[gus & (m.sub_class == "low-salt")].values,
            "ALL_GUSTATORY": m.root_id[gus].values}
    extra = {}
    for k, v in sets.items():
        idx, miss = to_index(v, ids)
        extra[k] = idx
        print("%s: %d cells (%d not in v783 list)" % (k, len(idx), miss), flush=True)
    cols, tags = [], []
    for k in extra:
        for hz in RATES:
            for s in SEEDS:
                cols.append((s, {"stim": {k: hz / 100.0}}))
                tags.append((k, hz, s))
    cnt, mcnt = simulate(ch, SCHED, cols, extra_groups=extra)
    dur = SCHED[-1][1] / 1000.0
    rows = []
    for b, (k, hz, s) in enumerate(tags):
        last = int((cnt[-1, :, b] / dur > 1).sum())
        rows.append(dict(set=k, n=len(extra[k]), hz=hz, seed=s, active_last=last, ignited=last > 1000,
                         mn9_stim_hz=float(mcnt[1, b] / 1.0)))
    save("e9_other_stimuli.json", dict(sched=SCHED, rows=rows, minutes=(time.time() - t0) / 60))
    for k in extra:
        for hz in RATES:
            sub = [r for r in rows if r["set"] == k and r["hz"] == hz]
            print("%-14s n=%3d %3d Hz | ignited %d/3 | active in last window %s" % (
                k, len(extra[k]), hz, sum(r["ignited"] for r in sub), [r["active_last"] for r in sub]), flush=True)
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
