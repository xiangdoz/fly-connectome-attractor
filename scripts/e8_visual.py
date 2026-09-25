"""E8 (logged before running): does visual stimulation (LC4 / LPLC2, the looming-sensitive types used to trigger escape
in the Shiu model by Xi & Chen 2025) ignite the same self-sustained state?
Each type alone, all its FlyWire v783 cells, Poisson 50/100/150/200 Hz for 1 s, then 1 s silence; fresh seeds 8261-8265.
Ignited = > 1,000 neurons active (> 1 Hz) in the last 250 ms of silence. For ignited runs: Jaccard with the E1 attractor
consensus set (neurons active in >= 50% of the 19 ignited E1 runs)."""
import os
import time

import pandas as pd

from common import np, save, RESULTS
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate

SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
SEEDS = tuple(range(8261, 8266))
RATES = (50, 100, 150, 200)


def main():
    t0 = time.time()
    ch = Channels()
    ids = pd.read_csv(COMP).iloc[:, 0].values
    ct = pd.read_csv("data/consolidated_cell_types.csv.gz").drop_duplicates("root_id")
    extra = {}
    for t in ("LC4", "LPLC2"):
        idx, miss = to_index(ct.root_id[ct.primary_type == t].values, ids)
        extra[t] = idx
        print("%s: %d cells (%d not in v783 completeness list)" % (t, len(idx), miss), flush=True)
    cols, tags = [], []
    for t in extra:
        for hz in RATES:
            for s in SEEDS:
                cols.append((s, {"stim": {t: hz / 100.0}}))
                tags.append((t, hz, s))
    cnt, mcnt = simulate(ch, SCHED, cols, extra_groups=extra)
    dur = SCHED[-1][1] / 1000.0
    e = np.load(os.path.join(RESULTS, "e1_last_active.npz"))
    freq = np.zeros(cnt.shape[1])
    for k in e.keys():
        freq[e[k]] += 1
    attractor = set(np.nonzero(freq >= 0.5 * len(list(e.keys())))[0].tolist())
    rows = []
    for b, (t, hz, s) in enumerate(tags):
        act = np.nonzero(cnt[-1, :, b] / dur > 1)[0]
        ign = len(act) > 1000
        jac = len(attractor & set(act.tolist())) / max(len(attractor | set(act.tolist())), 1) if ign else None
        rows.append(dict(type=t, hz=hz, seed=s, active_last=int(len(act)), ignited=bool(ign), jaccard_vs_E1=jac,
                         active_during=int((cnt[1, :, b] > 0).sum())))
    save("e8_visual.json", dict(sched=SCHED, rows=rows, minutes=(time.time() - t0) / 60))
    for t in extra:
        for hz in RATES:
            sub = [r for r in rows if r["type"] == t and r["hz"] == hz]
            j = [r["jaccard_vs_E1"] for r in sub if r["ignited"]]
            print("%-6s %3d Hz | ignited %d/5 | active in last window %s | Jaccard vs E1 attractor %s"
                  % (t, hz, sum(r["ignited"] for r in sub), [r["active_last"] for r in sub],
                     np.round(j, 2).tolist() if j else "-"), flush=True)
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
