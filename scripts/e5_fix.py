"""E5 (PLAN_v2.md): a minimal model change that removes the bistability but keeps validated behaviour.
(a) "no synaptic input to sensory neurons" was tested in E3: it does NOT remove ignition -> (a) fails.
(b) spike-frequency adaptation (lif_sfa.py), tau_a = 100 ms, d_a in {0.25, 0.5, 1, 2} mV.
Self-check first: d_a = 0 must reproduce the unmodified engine exactly.
Per d_a: ignition tests (LB3 at 120 Hz and SUGAR_ALL at 200 Hz, 1 s, then 1 s silence; seeds 8241-8245) and the
validated behaviour = Shiu et al.'s protocol (SHIU20 at 50/100/150/200 Hz, 1 s; seeds 8201-8205, paired with E1).
Fix works = no ignition in 5/5 for both ignition tests AND SHIU20 MN9 within 20% of the original at every rate."""
import json
import os
import time

import pandas as pd

from common import np, save, RESULTS
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate

SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
SHIU21 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
          720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
          720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
          720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
          720575940611875570]


def main():
    t0 = time.time()
    ch = Channels()
    shiu20, _ = to_index(SHIU21, pd.read_csv(COMP).iloc[:, 0].values)
    extra = {"SHIU20": shiu20}
    sugar = [t for t in ch.stim_names if ch.modality[t] == "sugar/water"]
    # ---- self-check: d_a = 0 == unmodified engine ----
    chk = [(8241, {"stim": {"LB3": 1.2}}), (8201, {"stim": {"SHIU20": 1.0}})]
    c0, m0 = simulate(ch, SCHED, chk, extra_groups=extra)
    c1, m1 = simulate(ch, SCHED, chk, extra_groups=extra, sfa=dict(d_a=0.0, tau_a=100.0))
    same = np.array_equal(c0, c1) and np.array_equal(m0, m1)
    print("self-check d_a = 0 vs unmodified engine (all neurons, all blocks):", "IDENTICAL" if same else "DIFFERENT", flush=True)
    if not same:
        raise SystemExit("self-check failed")
    # ---- original SHIU20 MN9 (paired seeds) from E1 ----
    e1 = json.load(open(os.path.join(RESULTS, "e1_ignition_map.json")))["rows"]
    orig = {(r["tag"][2], r["tag"][3]): r["mn9_hz"][1] for r in e1 if r["tag"][:2] == ["a", "SHIU20"]}
    cols, tags = [], []
    for s in range(8241, 8246):
        cols.append((s, {"stim": {"LB3": 1.2}})); tags.append(("ign_LB3_120", s))
        cols.append((s, {"stim": {t: 2.0 for t in sugar}})); tags.append(("ign_SUGAR_ALL_200", s))
    for hz in (50, 100, 150, 200):
        for s in range(8201, 8206):
            cols.append((s, {"stim": {"SHIU20": hz / 100.0}})); tags.append(("shiu_%d" % hz, s))
    dur = np.array([d / 1000.0 for _, d in SCHED])
    out = []
    for d_a in (0.25, 0.5, 1.0, 2.0):
        cnt, mcnt = simulate(ch, SCHED, cols, extra_groups=extra, sfa=dict(d_a=d_a, tau_a=100.0))
        last = [int((cnt[-1, :, b] / dur[-1] > 1).sum()) for b in range(len(cols))]
        mn9 = [float(mcnt[1, b] / dur[1]) for b in range(len(cols))]
        ign = {k: sum(last[b] > 1000 for b, (kk, _) in enumerate(tags) if kk == k) for k in ("ign_LB3_120", "ign_SUGAR_ALL_200")}
        ratio = {}
        for hz in (50, 100, 150, 200):
            new = [mn9[b] for b, (kk, s) in enumerate(tags) if kk == "shiu_%d" % hz]
            old = [orig[(hz, s)] for s in range(8201, 8206)]
            ratio[hz] = float(np.mean(new) / np.mean(old)) if np.mean(old) > 0 else float("nan")
        ok_ign = all(v == 0 for v in ign.values())
        ok_beh = all(abs(r - 1) <= 0.2 for hz, r in ratio.items() if hz >= 100)
        out.append(dict(d_a=d_a, ignited=ign, shiu20_mn9_ratio=ratio, last_active=last, mn9=mn9, works=ok_ign and ok_beh))
        print("d_a %.2f mV | ignited LB3-120: %d/5, SUGAR_ALL-200: %d/5 | SHIU20 MN9 new/orig at 50/100/150/200 Hz: %s | FIX WORKS: %s"
              % (d_a, ign["ign_LB3_120"], ign["ign_SUGAR_ALL_200"], " ".join("%.2f" % ratio[h] for h in (50, 100, 150, 200)),
                 ok_ign and ok_beh), flush=True)
        save("e5_fix.json", dict(self_check=same, rows=out, minutes=(time.time() - t0) / 60))
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
