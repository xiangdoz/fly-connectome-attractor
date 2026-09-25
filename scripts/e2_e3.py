"""E2: does the ignited state end on its own? LB3 1.2 for 500 ms, then 10 s silence in 1-s windows; seeds 8211-8215.
E3: which sub-networks are necessary? From the start, remove one cell class (W[:, j] = 0 for its neurons) or
remove all synaptic INPUT to sensory neurons (W[i, :] = 0); LB3 1.2 for 500 ms then 1 s silence; seeds 8216-8220.
Necessary = ignition in <= 1/5 while the intact control ignites in >= 4/5 (PLAN_v2.md).
  python e2_e3.py e2 | e3"""
import sys
import time

import pandas as pd

from common import np, save
from stimuli import Channels, COMP
from sim import simulate, weights

THRESH = 1000


def active(cnt, sched):
    dur = np.array([d / 1000.0 for _, d in sched])
    return [[int((cnt[k, :, b] / dur[k] > 1).sum()) for k in range(len(sched))] for b in range(cnt.shape[2])]


def e2():
    t0 = time.time()
    ch = Channels()
    sched = [("empty", 100), ("p1", 500)] + [("s%d" % i, 1000) for i in range(10)]
    seeds = list(range(8211, 8216))
    cnt, mcnt = simulate(ch, sched, [(s, {"p1": {"LB3": 1.2}}) for s in seeds])
    act = active(cnt, sched)
    for s, a in zip(seeds, act):
        print("seed %d | p1 %d | active per second of silence: %s" % (s, a[1], " ".join(str(v) for v in a[2:])), flush=True)
    persistent = sum(a[-1] > THRESH for a in act)
    print("persistent at 10 s: %d/5 -> %s" % (persistent, "PERSISTENT" if persistent >= 4 else "not persistent (criterion >= 4/5)"))
    save("e2_persistence.json", dict(sched=sched, seeds=seeds, active=act, mn9=mcnt.tolist(), minutes=(time.time() - t0) / 60))


def classes():
    ids = pd.read_csv(COMP).iloc[:, 0].values
    cls = pd.read_csv("data/classification.csv.gz")
    ct = pd.read_csv("data/consolidated_cell_types.csv.gz").drop_duplicates("root_id")
    m = pd.DataFrame({"root_id": ids}).merge(cls, on="root_id", how="left").merge(ct, on="root_id", how="left")
    sets = {c: np.nonzero((m["class"] == c).values)[0] for c in ("Kenyon_Cell", "ALPN", "olfactory", "ALLN", "LHLN", "DAN", "MBON")}
    sets["APL"] = np.nonzero((m.primary_type == "APL").values)[0]
    sensory = np.nonzero((m.super_class == "sensory").values)[0]
    return sets, sensory


def e3():
    t0 = time.time()
    ch = Channels()
    sets, sensory = classes()
    sched = [("empty", 100), ("p1", 500)] + [("s%d" % i, 250) for i in range(4)]
    seeds = list(range(8216, 8221))
    cols = [(s, {"p1": {"LB3": 1.2}}) for s in seeds]
    conds = [("intact", None)] + [("minus " + k, v) for k, v in sets.items()] + [("no input to sensory", "SENS")]
    rows = []
    for name, v in conds:
        if v is None:
            W = weights()
        elif isinstance(v, str):
            W = weights().tolil() if False else weights()
            W = W.tocsr(copy=True)
            mask = np.ones(W.shape[0])
            mask[sensory] = 0.0
            W = W.multiply(mask[:, None]).tocsr()                    # zero rows = no synaptic input to sensory cells
            W.eliminate_zeros()
        else:
            W = weights(ablate=v)
        cnt, mcnt = simulate(ch, sched, cols, W=W)
        act = active(cnt, sched)
        ign = sum(a[-1] > THRESH for a in act)
        n = len(v) if v is not None and not isinstance(v, str) else (len(sensory) if isinstance(v, str) else 0)
        rows.append(dict(cond=name, n_removed=n, ignited=ign, active=act, mn9_p1=(mcnt[1] / 0.5).tolist()))
        print("%-22s removed %6d | ignited %d/5 | active in last silent window %s | MN9 during pulse %s"
              % (name, n, ign, [a[-1] for a in act], np.round(mcnt[1] / 0.5, 0).tolist()), flush=True)
        save("e3_ablation.json", dict(sched=sched, seeds=seeds, rows=rows, minutes=(time.time() - t0) / 60))
    base = rows[0]["ignited"]
    print("intact ignited %d/5; NECESSARY (<= 1/5): %s" % (base, [r["cond"] for r in rows[1:] if r["ignited"] <= 1] if base >= 4 else "n/a"))


if __name__ == "__main__":
    {"e2": e2, "e3": e3}[sys.argv[1]]()
