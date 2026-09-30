"""E12 (logged before running): how much of the core must be re-signed?
Random partial re-signing of k of the 90 excitatory core cells (k = 23, 45, 68; 3 subsets each, rng seed 12) and two
evidence rules on fast-transmitter synapse votes (fast-majority: re-sign unless ACh > GABA+Glu; fast-plurality: re-sign
unless ACh > max(GABA, Glu)). Tests: LB3 120 Hz and all sugar GRNs 200 Hz, fresh seeds 8291-8293. Also the E11
spectral proxy for each condition."""
import os
import time

import pandas as pd

from common import np, save, RESULTS
from e3b_alln import meta
from e5c_sign_fix import reweighted
from e11_spectrum import lead
from lif import TAU
from stimuli import Channels
from sim import simulate, weights

SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
SEEDS = (8291, 8292, 8293)


def attractor_set(n):
    e = np.load(os.path.join(RESULTS, "e1_last_active.npz"))
    freq = np.zeros(n)
    for k in e.keys():
        freq[e[k]] += 1
    return np.nonzero(freq >= 0.5 * len(e.keys()))[0]


def main():
    t0 = time.time()
    ch = Channels()
    m = meta()
    nt = pd.read_csv("data/neurons.csv.gz")
    c2 = np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0]
    W0 = weights().tocsc()
    sign = np.array([np.sign(W0.data[W0.indptr[j]:W0.indptr[j + 1]].sum()) for j in c2])
    exc = c2[sign > 0]
    d = m.iloc[exc][["root_id"]].merge(nt, on="root_id", how="left")
    assert len(d) == len(exc) == 90
    conds = {}
    rng = np.random.default_rng(12)
    for k in (23, 45, 68):
        for r in range(3):
            conds["random k=%d #%d" % (k, r + 1)] = np.sort(rng.choice(exc, size=k, replace=False))
    conds["fast-majority"] = exc[~(d.ach_avg > d.gaba_avg + d.glut_avg).values]
    conds["fast-plurality"] = exc[~((d.ach_avg > d.gaba_avg) & (d.ach_avg > d.glut_avg)).values]
    conds["all 90 (E5c c2)"] = exc

    S = attractor_set(W0.shape[0])
    tau_s = TAU / 1000.0
    sugar = [t for t in ch.stim_names if ch.modality[t] == "sugar/water"]
    stims = {"LB3 120 Hz": {"LB3": 1.2}, "all sugar GRNs 200 Hz": {t: 2.0 for t in sugar}}
    cols, tags = [], []
    for name, aff in stims.items():
        for s in SEEDS:
            cols.append((s, {"stim": aff})); tags.append((name, s))
    dur = SCHED[-1][1] / 1000.0
    out = []
    for cname, idx in conds.items():
        W = reweighted(idx, "inhibitory")
        eig = lead((W[S][:, S] * tau_s).tocsr())
        if cname.startswith("all 90"):
            rows = None
        else:
            cnt, _ = simulate(ch, SCHED, cols, W=W)
            rows = [dict(stim=n, seed=s, active_last=int((cnt[-1, :, b] / dur > 1).sum())) for b, (n, s) in enumerate(tags)]
            for r in rows:
                r["ignited"] = r["active_last"] > 1000
        out.append(dict(cond=cname, n_resigned=int(len(idx)), eig=eig, rows=rows,
                        ignited=None if rows is None else int(sum(r["ignited"] for r in rows)),
                        cells=[int(i) for i in idx]))
        print("%-18s | re-signed %2d | leading eig %.3f | ignited %s/6 | last-window active %s"
              % (cname, len(idx), eig, "-" if rows is None else out[-1]["ignited"],
                 "-" if rows is None else [r["active_last"] for r in rows]), flush=True)
        save("e12_partial_resign.json", dict(conds=out, minutes=(time.time() - t0) / 60))
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
