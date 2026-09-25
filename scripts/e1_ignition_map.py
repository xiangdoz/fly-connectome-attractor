"""E1 (PLAN_v2.md + LOG clarification): which inputs ignite the self-sustained state?
(a) SHIU20 (Shiu et al.'s 20 LB3-left sugar GRNs) and SUGAR_ALL (all sugar/water GRNs: LB2d + LB3) at
    50/100/150/200 Hz for 1 s, then 1 s silence; seeds 8201-8205.
(b) every gustatory type alone at 100 and 200 Hz; seeds 8206-8208.
Ignited = > 1,000 neurons active (> 1 Hz) in the LAST 250 ms of silence."""
import os
import time

import pandas as pd

from common import np, save, RESULTS
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate

SHIU21 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
          720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
          720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
          720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
          720575940611875570]
SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
THRESH = 1000


def main():
    t0 = time.time()
    ch = Channels()
    ids = pd.read_csv(COMP).iloc[:, 0].values
    shiu20, missing = to_index(SHIU21, ids)
    extra = {"SHIU20": shiu20}
    sugar_types = [t for t in ch.stim_names if ch.modality[t] == "sugar/water"]
    cols, tags = [], []
    for name in ("SHIU20", "SUGAR_ALL"):
        for hz in (50, 100, 150, 200):
            for s in range(8201, 8206):
                aff = {"stim": ({"SHIU20": hz / 100.0} if name == "SHIU20" else {t: hz / 100.0 for t in sugar_types})}
                cols.append((s, aff)); tags.append(("a", name, hz, s))
    for t in ch.stim_names:
        for hz in (100, 200):
            for s in range(8206, 8209):
                cols.append((s, {"stim": {t: hz / 100.0}})); tags.append(("b", t, hz, s))
    print("%d columns; SHIU20 mapped %d cells (missing %d)" % (len(cols), len(shiu20), missing), flush=True)
    cnt, mcnt = simulate(ch, SCHED, cols, extra_groups=extra)
    dur = np.array([d / 1000.0 for _, d in SCHED])
    rows, last_active = [], {}
    for b, tag in enumerate(tags):
        act = [int((cnt[k, :, b] / dur[k] > 1).sum()) for k in range(len(SCHED))]
        rows.append(dict(tag=list(tag), active=act, mn9_hz=(mcnt[:, b] / dur).round(2).tolist(),
                         ignited=act[-1] > THRESH))
        if act[-1] > THRESH:
            last_active["%s_%s_%d_%d" % tag] = np.nonzero(cnt[-1, :, b] > 0)[0].astype(np.int32)
    np.savez_compressed(os.path.join(RESULTS, "e1_last_active.npz"), **last_active)
    save("e1_ignition_map.json", dict(sched=SCHED, rows=rows, shiu20=shiu20.tolist(), minutes=(time.time() - t0) / 60))
    print("\n(a) ignited / 5 seeds  [mean MN9 Hz during stimulus]")
    for name in ("SHIU20", "SUGAR_ALL"):
        line = []
        for hz in (50, 100, 150, 200):
            sub = [r for r in rows if r["tag"][:3] == ["a", name, hz]]
            line.append("%3d Hz: %d/5 [%.0f]" % (hz, sum(r["ignited"] for r in sub), np.mean([r["mn9_hz"][1] for r in sub])))
        print("   %-9s %s" % (name, " | ".join(line)))
    print("(b) single types, ignited / 3 seeds at 100 Hz | 200 Hz")
    for t in ch.stim_names:
        a = [sum(r["ignited"] for r in rows if r["tag"][:3] == ["b", t, hz]) for hz in (100, 200)]
        print("   %-13s %-14s n=%3d | %d/3 | %d/3" % (t, ch.modality[t][:14], len(ch.stim[t]), a[0], a[1]))
    print("minutes %.1f" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
