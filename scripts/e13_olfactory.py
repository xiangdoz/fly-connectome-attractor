"""E13 (logged before running): olfactory stimulation, original vs corrected model.
Single ORNs (20 and 100 Hz), single ORN types (30 and 100 Hz), all ORNs (30 Hz); 1 s then 1 s silence; seeds 8301-8302.
Ignited = > 1,000 active (> 1 Hz) in the last 250 ms."""
import time

import pandas as pd

from common import np, save
from e3b_alln import meta
from e5c_sign_fix import reweighted
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate, weights

SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
SEEDS = (8301, 8302)


def main():
    t0 = time.time()
    ch = Channels()
    m = meta()
    ids = pd.read_csv(COMP).iloc[:, 0].values
    orn = (m.super_class == "sensory") & (m["class"] == "olfactory")
    rng = np.random.default_rng(13)
    types = sorted(m[orn].primary_type.dropna().unique())
    pick_types = list(rng.choice(types, size=5, replace=False))
    extra, stims = {}, {}
    for t in pick_types:
        cells = np.nonzero((orn & (m.primary_type == t)).values)[0]
        one = int(rng.choice(cells))
        extra["ONE_" + t] = np.array([one])
        extra["TYPE_" + t] = cells
    extra["ALL_ORN"] = np.nonzero(orn.values)[0]
    for t in pick_types:
        stims["single %s 20 Hz" % t] = {"ONE_" + t: 0.2}
    for t in pick_types:
        stims["single %s 100 Hz" % t] = {"ONE_" + t: 1.0}
    for t in pick_types:
        stims["type %s (%d) 30 Hz" % (t, len(extra["TYPE_" + t]))] = {"TYPE_" + t: 0.3}
    for t in pick_types:
        stims["type %s (%d) 100 Hz" % (t, len(extra["TYPE_" + t]))] = {"TYPE_" + t: 1.0}
    stims["all %d ORNs 30 Hz" % len(extra["ALL_ORN"])] = {"ALL_ORN": 0.3}
    cols, tags = [], []
    for name, aff in stims.items():
        for s in SEEDS:
            cols.append((s, {"stim": aff})); tags.append((name, s))
    c2 = np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0]
    models = {"original": weights(), "corrected": reweighted(c2, "inhibitory")}
    dur_stim, dur_last = SCHED[1][1] / 1000.0, SCHED[-1][1] / 1000.0
    out = {}
    for mname, W in models.items():
        cnt, _ = simulate(ch, SCHED, cols, W=W, extra_groups=extra)
        rows = [dict(stim=n, seed=s, active_stim=int((cnt[1, :, b] / dur_stim > 1).sum()),
                     active_last=int((cnt[-1, :, b] / dur_last > 1).sum())) for b, (n, s) in enumerate(tags)]
        for r in rows:
            r["ignited"] = r["active_last"] > 1000
        out[mname] = rows
        save("e13_olfactory.json", dict(models=out, pick_types=pick_types, minutes=(time.time() - t0) / 60))
        print("== %s model" % mname, flush=True)
        for name in stims:
            sub = [r for r in rows if r["stim"] == name]
            print("  %-28s | ignited %d/2 | active during stim %s | active in last window %s" % (
                name, sum(r["ignited"] for r in sub), [r["active_stim"] for r in sub], [r["active_last"] for r in sub]),
                flush=True)
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
