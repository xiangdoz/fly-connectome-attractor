"""E6: PROSPECTIVE test of the wiring-based ignition predictor (designed on E1c).
  python e6_prospective.py predict   -> freeze thresholds on E1c, draw NEW random LB3 subsets, write predictions
                                        (results/e6_predictions.json + its sha256 in LOG) BEFORE any simulation
  python e6_prospective.py simulate  -> simulate exactly those subsets, score the frozen predictions
Predictor: 2-hop excitatory drive from the stimulated cells onto the 176-cell ALLN core (unknown/SER/OCT nt), x rate.
Baseline: number of cells x rate. Thresholds = the value maximizing accuracy on E1c (midpoint between sorted scores)."""
import hashlib
import json
import os
import sys
import time

from common import np, save, RESULTS
from e3b_alln import meta
from stimuli import Channels
from batch import get_W

KS, SEEDS, HZ = (30, 40, 50, 60, 70, 80, 90), tuple(range(8251, 8257)), 150
SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]


def setup():
    ch = Channels()
    W = get_W().tocsr()
    m = meta()
    alln = m["class"] == "ALLN"
    core = np.nonzero((alln & ~m.nt_type.isin(["ACH", "GABA", "GLUT"])).values)[0]
    Wp = W.multiply(W > 0).tocsr()
    return ch, W.shape[0], core, Wp


def core_drive(idx, n, core, Wp):
    x = np.zeros(n)
    x[idx] = 1.0
    return float((Wp @ (Wp @ x))[core].sum())


def best_threshold(scores, y):
    s = np.sort(np.unique(scores))
    cands = np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]])
    acc = [np.mean((np.asarray(scores) > c) == np.asarray(y)) for c in cands]
    i = int(np.argmax(acc))
    return float(cands[i]), float(acc[i])


def predict():
    ch, n, core, Wp = setup()
    lb3 = ch.stim["LB3"]
    rows = json.load(open(os.path.join(RESULTS, "e1c_phase.json")))["rows"]
    sc, ct, y = [], [], []
    for r in rows:
        idx = np.sort(np.random.default_rng(r["seed"] * 1000 + r["k"]).choice(lb3, size=r["k"], replace=False))
        sc.append(core_drive(idx, n, core, Wp) * r["hz"]); ct.append(r["k"] * r["hz"]); y.append(r["ignited"])
    th_core, acc_core = best_threshold(sc, y)
    th_cnt, acc_cnt = best_threshold(ct, y)
    preds = []
    for k in KS:
        for s in SEEDS:
            idx = np.sort(np.random.default_rng(s * 7919 + k).choice(lb3, size=k, replace=False))   # new draw rule
            d = core_drive(idx, n, core, Wp) * HZ
            preds.append(dict(k=k, seed=s, cells=idx.tolist(), core_score=d, pred_core=d > th_core, pred_count=k * HZ > th_cnt))
    out = dict(frozen=time.strftime("%Y-%m-%d %H:%M:%S"), rate_hz=HZ, threshold_core=th_core, e1c_acc_core=acc_core,
               threshold_count=th_cnt, e1c_acc_count=acc_cnt, predictions=preds)
    path = save("e6_predictions.json", out)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print("frozen thresholds: core %.0f (E1c acc %.2f) | count %.0f (E1c acc %.2f)" % (th_core, acc_core, th_cnt, acc_cnt))
    print("predicted ignited: core %d / %d | count %d / %d" % (sum(p["pred_core"] for p in preds), len(preds),
                                                            sum(p["pred_count"] for p in preds), len(preds)))
    print("sha256(e6_predictions.json) = %s" % h)


def simulate_and_score():
    from sim import simulate
    ch = Channels()
    P = json.load(open(os.path.join(RESULTS, "e6_predictions.json")))
    extra, cols = {}, []
    for p in P["predictions"]:
        name = "E6_%d_%d" % (p["k"], p["seed"])
        extra[name] = np.asarray(p["cells"])
        cols.append((p["seed"], {"stim": {name: P["rate_hz"] / 100.0}}))
    cnt, _ = simulate(ch, SCHED, cols, extra_groups=extra)
    dur = SCHED[-1][1] / 1000.0
    ign = [bool((cnt[-1, :, b] / dur > 1).sum() > 1000) for b in range(len(cols))]
    acc_core = np.mean([p["pred_core"] == g for p, g in zip(P["predictions"], ign)])
    acc_cnt = np.mean([p["pred_count"] == g for p, g in zip(P["predictions"], ign)])
    base = max(np.mean(ign), 1 - np.mean(ign))
    print("E6 prospective: %d / %d ignited | accuracy core-drive predictor %.2f | count predictor %.2f | majority-class %.2f"
          % (sum(ign), len(ign), acc_core, acc_cnt, base))
    for k in KS:
        sub = [(p, g) for p, g in zip(P["predictions"], ign) if p["k"] == k]
        print("   k %2d | ignited %d/%d | core predicted %d | count predicted %d" % (
            k, sum(g for _, g in sub), len(sub), sum(p["pred_core"] for p, _ in sub), sum(p["pred_count"] for p, _ in sub)))
    save("e6_result.json", dict(ignited=ign, acc_core=acc_core, acc_count=acc_cnt, majority=base))


if __name__ == "__main__":
    {"predict": predict, "simulate": simulate_and_score}[sys.argv[1]]()
