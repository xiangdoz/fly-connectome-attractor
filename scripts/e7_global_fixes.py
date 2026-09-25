"""E7 (logged before running): global fixes reported for Shiu-style models (Todd, Fly Brain Bridge) under OUR criteria.
 g1: all weights scaled by 0.15 / 0.275 (w_syn 0.15 mV); g2: fast outputs of DA / SER / OCT neurons removed;
 g3: every connection capped at 100 synapses (|w_ij| <= 100 x 0.275 mV).
Tests and criterion identical to e5_fix.py / e5c_sign_fix.py."""
import json
import os
import time

import pandas as pd

from common import np, save, RESULTS
from e3b_alln import meta
from e5_fix import SCHED, SHIU21
from lif import W_SYN
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate, weights


def variant(name, m):
    W = weights().tocsc(copy=True)
    if name == "g1":
        W.data *= 0.15 / 0.275
    elif name == "g2":
        for j in np.nonzero(m.nt_type.isin(["DA", "SER", "OCT"]).values)[0]:
            W.data[W.indptr[j]:W.indptr[j + 1]] = 0.0
    elif name == "g3":
        cap = 100 * W_SYN
        W.data = np.sign(W.data) * np.minimum(np.abs(W.data), cap)
    W = W.tocsr()
    W.eliminate_zeros()
    return W


def main():
    t0 = time.time()
    ch = Channels()
    m = meta()
    shiu20, _ = to_index(SHIU21, pd.read_csv(COMP).iloc[:, 0].values)
    extra = {"SHIU20": shiu20}
    sugar = [t for t in ch.stim_names if ch.modality[t] == "sugar/water"]
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
    names = {"g1": "g1: w_syn 0.15 mV (all weights x0.545)", "g2": "g2: DA/SER/OCT outputs removed (%d cells)"
             % int(m.nt_type.isin(["DA", "SER", "OCT"]).sum()), "g3": "g3: connections capped at 100 synapses"}
    out = []
    for key in ("g1", "g2", "g3"):
        cnt, mcnt = simulate(ch, SCHED, cols, W=variant(key, m), extra_groups=extra)
        last = [int((cnt[-1, :, b] / dur[-1] > 1).sum()) for b in range(len(cols))]
        mn9 = [float(mcnt[1, b] / dur[1]) for b in range(len(cols))]
        ign = {k: sum(last[b] > 1000 for b, (kk, _) in enumerate(tags) if kk == k) for k in ("ign_LB3_120", "ign_SUGAR_ALL_200")}
        ratio = {hz: float(np.mean([mn9[b] for b, (kk, s) in enumerate(tags) if kk == "shiu_%d" % hz]) /
                           np.mean([orig[(hz, s)] for s in range(8201, 8206)])) for hz in (50, 100, 150, 200)}
        works = all(v == 0 for v in ign.values()) and all(abs(ratio[h] - 1) <= 0.2 for h in (100, 150, 200))
        out.append(dict(cond=names[key], ignited=ign, shiu20_mn9_ratio=ratio, last_active=last, mn9=mn9, works=works))
        print("%-46s | ignited LB3-120 %d/5, SUGAR_ALL-200 %d/5 | SHIU20 MN9 new/orig 50/100/150/200: %s | FIX WORKS: %s"
              % (names[key], ign["ign_LB3_120"], ign["ign_SUGAR_ALL_200"], " ".join("%.2f" % ratio[h] for h in (50, 100, 150, 200)),
                 works), flush=True)
        save("e7_global_fixes.json", dict(rows=out, minutes=(time.time() - t0) / 60))
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
