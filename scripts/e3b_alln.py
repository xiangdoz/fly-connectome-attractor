"""E3b (exploratory refinement of E3, logged as such): which antennal-lobe local neurons (ALLN) are necessary?
Remove outputs of ALLN subsets split by predicted transmitter and by primary type; LB3 1.2 for 500 ms then 1 s
silence; fresh seeds 8221-8225. Also: shortest excitatory paths from LB3 to ALLN (wiring only)."""
import time

import pandas as pd

from common import np, save
from stimuli import Channels, COMP
from batch import get_W
from sim import simulate, weights

THRESH = 1000
SEEDS = list(range(8221, 8226))


def meta():
    ids = pd.read_csv(COMP).iloc[:, 0].values
    cls = pd.read_csv("data/classification.csv.gz")
    ct = pd.read_csv("data/consolidated_cell_types.csv.gz").drop_duplicates("root_id")
    nt = pd.read_csv("data/neurons.csv.gz", usecols=["root_id", "nt_type"])
    return pd.DataFrame({"root_id": ids}).merge(cls, on="root_id", how="left").merge(ct, on="root_id", how="left") \
        .merge(nt, on="root_id", how="left")


def main():
    t0 = time.time()
    ch = Channels()
    m = meta()
    alln = m["class"] == "ALLN"
    print("ALLN by transmitter:", m[alln].nt_type.value_counts(dropna=False).to_dict())
    print("ALLN top types:", m[alln].primary_type.value_counts().head(12).to_dict(), flush=True)
    conds = {}
    for nt in ("ACH", "GABA", "GLUT"):
        conds["ALLN %s" % nt] = np.nonzero((alln & (m.nt_type == nt)).values)[0]
    conds["ALLN other/unknown nt"] = np.nonzero((alln & ~m.nt_type.isin(["ACH", "GABA", "GLUT"])).values)[0]
    for t in m[alln].primary_type.value_counts().head(8).index:
        conds["ALLN type %s" % t] = np.nonzero((alln & (m.primary_type == t)).values)[0]
    sched = [("empty", 100), ("p1", 500)] + [("s%d" % i, 250) for i in range(4)]
    cols = [(s, {"p1": {"LB3": 1.2}}) for s in SEEDS]
    dur = np.array([d / 1000.0 for _, d in sched])
    rows = []
    for name, idx in [("intact", np.array([], int))] + list(conds.items()):
        cnt, mcnt = simulate(ch, sched, cols, W=weights(ablate=idx))
        act = [int((cnt[-1, :, b] / dur[-1] > 1).sum()) for b in range(len(SEEDS))]
        ign = sum(a > THRESH for a in act)
        rows.append(dict(cond=name, n=len(idx), ignited=ign, active_last=act))
        print("%-28s removed %4d | ignited %d/5 | active last window %s" % (name, len(idx), ign, act), flush=True)
        save("e3b_alln.json", dict(seeds=SEEDS, rows=rows, minutes=(time.time() - t0) / 60))
    # wiring: strongest excitatory route LB3 -> ALLN (max-product of normalized positive weights, 3 hops)
    W = get_W().tocsr()
    Wp = W.multiply(W > 0).tocsr()
    colsum = np.asarray(Wp.sum(0)).ravel() + 1e-12
    lb3 = ch.stim["LB3"]
    x = np.zeros(W.shape[0]); x[lb3] = 1.0
    reach = []
    for hop in range(1, 5):
        x = Wp @ x
        a = x[np.nonzero(alln.values)[0]]
        reach.append(dict(hop=hop, alln_reached=int((a > 0).sum()), total_reached=int((x > 0).sum())))
        print("excitatory hops from LB3: %d -> ALLN reached %d / %d, all cells reached %d" % (hop, (a > 0).sum(), alln.sum(), (x > 0).sum()))
    save("e3b_alln.json", dict(seeds=SEEDS, rows=rows, reach=reach, minutes=(time.time() - t0) / 60))
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
