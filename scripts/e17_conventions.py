"""E17 (logged before running): the attractor and the screen under alternative sign conventions, brain-wide.
C0 original; C1 fast-majority for neurons without a call; C2 hemilineage imputation for neurons without a call;
C3 flyer-style (ALLN inhibitory unless called ACh; DA/5HT/OA outputs zero). Screen on LB3 200 Hz; simulations on
seeds 8321-8322 (ignition) and 8201-8203 (SHIU20 MN9)."""
import json
import os
import time

import pandas as pd

from common import np, save, RESULTS
from e3b_alln import meta
from e4_meanfield import predict
from e5_fix import SHIU21
from e16_screen_robustness import screen
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate, weights

SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
INH = {"GABA", "GLUT"}


def signed(W0, new_sign):
    """new_sign: array over neurons, +1 / -1 / 0 (0 = zero outputs). Columns of W0 hold the presynaptic neuron."""
    W = W0.tocsc(copy=True)
    for j in np.nonzero(new_sign != np.sign(np.asarray(W0.sum(0)).ravel()))[0]:
        s = slice(W.indptr[j], W.indptr[j + 1])
        W.data[s] = new_sign[j] * np.abs(W.data[s])
    W = W.tocsr()
    W.eliminate_zeros()
    return W


def main():
    t0 = time.time()
    ch = Channels()
    m = meta()
    nt = pd.read_csv("data/neurons.csv.gz")[["root_id", "gaba_avg", "glut_avg", "ach_avg"]]
    m = m.merge(nt, on="root_id", how="left")
    W0 = weights().tocsc()
    sign0 = np.sign(np.asarray(W0.sum(0)).ravel())
    nocall = m.nt_type.isna().values
    # C1 fast-majority
    s1 = sign0.copy()
    fast_inh = (m.gaba_avg.fillna(0) + m.glut_avg.fillna(0) > m.ach_avg.fillna(0)).values
    s1[nocall & (sign0 != 0)] = np.where(fast_inh[nocall & (sign0 != 0)], -1, 1)
    # C2 hemilineage imputation
    s2 = sign0.copy()
    called = m[~m.nt_type.isna() & m.hemilineage.notna()]
    maj = called.groupby("hemilineage").nt_type.agg(lambda x: x.value_counts().index[0]).to_dict()
    hl = m.hemilineage.values
    n_imputed = 0
    for i in np.nonzero(nocall & (sign0 != 0))[0]:
        if isinstance(hl[i], str) and hl[i] in maj:
            s2[i] = -1 if maj[hl[i]] in INH else 1
            n_imputed += 1
    # C3 flyer-style
    s3 = sign0.copy()
    alln = (m["class"] == "ALLN").values
    s3[alln & (m.nt_type != "ACH").values & (sign0 != 0)] = -1
    s3[m.nt_type.isin(["DA", "SER", "OCT"]).values] = 0
    conv = {"C0 original": sign0, "C1 fast-majority (no-call neurons)": s1,
            "C2 hemilineage imputation (no-call neurons)": s2, "C3 flyer-style": s3}
    core = set(np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0].tolist())
    ids = pd.read_csv(COMP).iloc[:, 0].values
    shiu20, _ = to_index(SHIU21, ids)
    orn = (m.super_class == "sensory") & (m["class"] == "olfactory")
    extra = {"SHIU20": shiu20, "VM3": np.nonzero((orn & (m.primary_type == "ORN_VM3")).values)[0],
             "ALL_ORN": np.nonzero(orn.values)[0]}
    sugar = [t for t in ch.stim_names if ch.modality[t] == "sugar/water"]
    stims = {"LB3 120 Hz": {"LB3": 1.2}, "all sugar GRNs 200 Hz": {t: 2.0 for t in sugar},
             "VM3 glomerulus 100 Hz": {"VM3": 1.0}, "all ORNs 30 Hz": {"ALL_ORN": 0.3}}
    e1 = json.load(open(os.path.join(RESULTS, "e1_ignition_map.json")))["rows"]
    orig = {(r["tag"][2], r["tag"][3]): r["mn9_hz"][1] for r in e1 if r["tag"][:2] == ["a", "SHIU20"]}
    dur_last, dur_stim = SCHED[-1][1] / 1000.0, SCHED[1][1] / 1000.0
    out = dict(n_nocall=int(nocall.sum()), n_imputed_c2=int(n_imputed), hemilineage_majority=maj, conventions=[])
    for name, sg in conv.items():
        W = weights() if name.startswith("C0") else signed(W0, sg)
        n_changed = int(np.sum(sg != sign0))
        Wf = W.astype(np.float64).tocsr()
        _, r2, _ = predict(Wf, ch.stim["LB3"], 200.0, len(m))
        P = np.nonzero(r2 > 1.0)[0]
        entry = dict(conv=name, n_changed=n_changed, n_P=int(len(P)))
        if len(P) >= 50:
            lam, ranked, _ = screen(Wf, P, m)
            entry.update(lam=lam, n_cand=int(len(ranked)),
                         top10_classes=m.iloc[ranked[:10]]["class"].fillna("NA").value_counts().to_dict(),
                         top45_core=float(np.mean([g in core for g in ranked[:45]])) if len(ranked) >= 45 else None)
        cols, tags = [], []
        for sn, aff in stims.items():
            for s in (8321, 8322):
                cols.append((s, {"stim": aff})); tags.append((sn, s))
        for hz in (100, 150, 200):
            for s in (8201, 8202, 8203):
                cols.append((s, {"stim": {"SHIU20": hz / 100.0}})); tags.append(("shiu_%d" % hz, s))
        cnt, mcnt = simulate(ch, SCHED, cols, W=W, extra_groups=extra)
        last = [int((cnt[-1, :, b] / dur_last > 1).sum()) for b in range(len(cols))]
        entry["ignition"] = {sn: [last[b] for b, (t, _) in enumerate(tags) if t == sn] for sn in stims}
        entry["n_ignited"] = int(sum(v > 1000 for sn in stims for v in entry["ignition"][sn]))
        entry["mn9_ratio"] = {hz: float(np.mean([mcnt[1, b] / dur_stim for b, (t, s) in enumerate(tags) if t == "shiu_%d" % hz])
                                        / np.mean([orig[(hz, s)] for s in (8201, 8202, 8203)])) for hz in (100, 150, 200)}
        out["conventions"].append(entry)
        save("e17_conventions.json", dict(out, minutes=(time.time() - t0) / 60))
        print("%-46s | changed %5d | |P| %5d | lam %s | ignited %d/8 %s | MN9 ratio %s | top10 %s"
              % (name, n_changed, len(P), "%.2f" % entry["lam"] if "lam" in entry else "-", entry["n_ignited"],
                 entry["ignition"], {h: round(v, 2) for h, v in entry["mn9_ratio"].items()},
                 entry.get("top10_classes")), flush=True)
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
