"""E5(c) (logged before running): targeted sign fix. c1: outputs of the 176 unknown/SER/OCT ALLN made inhibitory;
c2: only the 150 unknown-nt ALLN; c3: every neuron without a transmitter prediction gets zero outputs.
Tests / criterion identical to e5_fix.py (ignition LB3-120 and SUGAR_ALL-200 on seeds 8241-8245 -> 0/5 each;
SHIU20 MN9 within 20% of the original at 100/150/200 Hz, seeds 8201-8205 paired with E1)."""
import json
import os
import time

import pandas as pd

from common import np, save, RESULTS
from e3b_alln import meta
from e5_fix import SCHED, SHIU21
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate, weights


def reweighted(idx, mode):
    W = weights().tocsc(copy=True)
    for j in idx:
        s = slice(W.indptr[j], W.indptr[j + 1])
        W.data[s] = -np.abs(W.data[s]) if mode == "inhibitory" else 0.0
    W = W.tocsr()
    W.eliminate_zeros()
    return W


def main():
    t0 = time.time()
    ch = Channels()
    m = meta()
    alln = m["class"] == "ALLN"
    sets = {"c1: 176 unknown/SER/OCT ALLN -> inhibitory": (np.nonzero((alln & ~m.nt_type.isin(["ACH", "GABA", "GLUT"])).values)[0], "inhibitory"),
            "c2: 150 unknown-nt ALLN -> inhibitory": (np.nonzero((alln & m.nt_type.isna()).values)[0], "inhibitory"),
            "c3: all %d unknown-nt neurons -> zero outputs" % int(m.nt_type.isna().sum()): (np.nonzero(m.nt_type.isna().values)[0], "zero")}
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
    out = []
    for name, (idx, mode) in sets.items():
        cnt, mcnt = simulate(ch, SCHED, cols, W=reweighted(idx, mode), extra_groups=extra)
        last = [int((cnt[-1, :, b] / dur[-1] > 1).sum()) for b in range(len(cols))]
        mn9 = [float(mcnt[1, b] / dur[1]) for b in range(len(cols))]
        ign = {k: sum(last[b] > 1000 for b, (kk, _) in enumerate(tags) if kk == k) for k in ("ign_LB3_120", "ign_SUGAR_ALL_200")}
        ratio = {hz: float(np.mean([mn9[b] for b, (kk, s) in enumerate(tags) if kk == "shiu_%d" % hz]) /
                           np.mean([orig[(hz, s)] for s in range(8201, 8206)])) for hz in (50, 100, 150, 200)}
        works = all(v == 0 for v in ign.values()) and all(abs(ratio[h] - 1) <= 0.2 for h in (100, 150, 200))
        out.append(dict(cond=name, n=len(idx), ignited=ign, shiu20_mn9_ratio=ratio, last_active=last, mn9=mn9, works=works))
        print("%-52s | ignited LB3-120 %d/5, SUGAR_ALL-200 %d/5 | SHIU20 MN9 new/orig 50/100/150/200: %s | FIX WORKS: %s"
              % (name, ign["ign_LB3_120"], ign["ign_SUGAR_ALL_200"], " ".join("%.2f" % ratio[h] for h in (50, 100, 150, 200)), works),
              flush=True)
        save("e5c_sign_fix.json", dict(rows=out, minutes=(time.time() - t0) / 60))
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
