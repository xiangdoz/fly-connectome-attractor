"""E10 (logged before running): stress test of the corrected model (150 unknown-nt ALLN re-signed inhibitory).
LB3 120 / 200 Hz, all sugar GRNs 200 Hz, all gustatory neurons 200 Hz, JO-C+JO-E 200 Hz; 1 s then 1 s silence;
fresh seeds 8281-8285. Ignited = > 1,000 active (> 1 Hz) in the last 250 ms."""
import time

import pandas as pd

from common import np, save
from e3b_alln import meta
from e5c_sign_fix import reweighted
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate

SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
SEEDS = tuple(range(8281, 8286))


def main():
    t0 = time.time()
    ch = Channels()
    m = meta()
    ids = pd.read_csv(COMP).iloc[:, 0].values
    c2 = np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0]
    W = reweighted(c2, "inhibitory")
    gus = (m.super_class == "sensory") & (m["class"] == "gustatory")
    extra = {"ALL_GUSTATORY": to_index(m.root_id[gus].values, ids)[0],
             "JO_CE": to_index(m.root_id[m.primary_type.isin(["JO-C", "JO-E"])].values, ids)[0]}
    sugar = [t for t in ch.stim_names if ch.modality[t] == "sugar/water"]
    stims = {"LB3 120 Hz": {"LB3": 1.2}, "LB3 200 Hz": {"LB3": 2.0}, "all sugar GRNs 200 Hz": {t: 2.0 for t in sugar},
             "all gustatory 200 Hz": {"ALL_GUSTATORY": 2.0}, "JO-C+JO-E 200 Hz": {"JO_CE": 2.0}}
    cols, tags = [], []
    for name, aff in stims.items():
        for s in SEEDS:
            cols.append((s, {"stim": aff})); tags.append((name, s))
    cnt, mcnt = simulate(ch, SCHED, cols, W=W, extra_groups=extra)
    dur = SCHED[-1][1] / 1000.0
    rows = [dict(stim=n, seed=s, active_last=int((cnt[-1, :, b] / dur > 1).sum()), active_stim=int((cnt[1, :, b] > 0).sum()))
            for b, (n, s) in enumerate(tags)]
    for r in rows:
        r["ignited"] = r["active_last"] > 1000
    save("e10_fix_stress.json", dict(rows=rows, minutes=(time.time() - t0) / 60))
    for name in stims:
        sub = [r for r in rows if r["stim"] == name]
        print("%-24s | ignited %d/5 | active in last window %s | active during stimulus %s" % (
            name, sum(r["ignited"] for r in sub), [r["active_last"] for r in sub], [r["active_stim"] for r in sub]), flush=True)
    print("TOTAL ignited: %d / %d | minutes %.1f" % (sum(r["ignited"] for r in rows), len(rows), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
