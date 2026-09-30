"""E15 (logged before running; post hoc, wiring only): (a) wall time of the end-to-end screen (mean-field persistent
set for LB3 at 200 Hz, then eigenvalue-sensitivity ranking); (b) the E6 drive score with the screen's top 66 as the
target set instead of the ablation-defined 176-cell core, threshold frozen on E1c, scored on the 42 E6 sets."""
import json
import os
import time

from scipy.sparse.linalg import eigs

from common import np, save, RESULTS
from e3b_alln import meta
from e4_meanfield import predict
from e6_prospective import best_threshold, core_drive, KS, SEEDS, HZ
from lif import TAU
from stimuli import Channels
from batch import get_W


def auc(scores, y):
    s, y = np.asarray(scores), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def main():
    ch = Channels()
    m = meta()
    # (a) timing of the end-to-end screen
    t0 = time.time()
    W = get_W().astype(np.float64).tocsr()
    n = W.shape[0]
    t_load = time.time() - t0
    t1 = time.time()
    _, r2, _ = predict(W, ch.stim["LB3"], 200.0, n)
    P = np.nonzero(r2 > 1.0)[0]
    t_mf = time.time() - t1
    t2 = time.time()
    A = (W[P][:, P] * (TAU / 1000.0)).tocsc()
    lam, V = eigs(A.tocsr(), k=1, which="LR", maxiter=20000, tol=1e-10)
    _, U = eigs(A.T.tocsr(), k=1, which="LR", maxiter=20000, tol=1e-10)
    v, u = np.real(V[:, 0]), np.real(U[:, 0])
    v, u = (v if v.sum() > 0 else -v), (u if u.sum() > 0 else -u)
    colsum = np.asarray(A.sum(0)).ravel()
    cand = np.nonzero(m.nt_type.isna().values[P] & (colsum > 0))[0]
    d = np.array([-2.0 * float(u[A.indices[A.indptr[j]:A.indptr[j + 1]]] @ A.data[A.indptr[j]:A.indptr[j + 1]]) * v[j]
                  / float(u @ v) for j in cand])
    ranked = P[cand[np.argsort(d)]]
    t_rank = time.time() - t2
    ref = json.load(open(os.path.join(RESULTS, "e14_rank.json")))["ranked_global"]
    same_top66 = sorted(ranked[:66].tolist()) == sorted(ref[:66])
    print("timing: load W %.1f s | mean-field (both phases) %.1f s | eigen + ranking %.1f s | |P| = %d | top-66 same as E14: %s"
          % (t_load, t_mf, t_rank, len(P), same_top66), flush=True)
    # (b) drive score onto the screen's top 66
    top66 = np.array(ref[:66])
    Wp = get_W().tocsr()
    Wp = Wp.multiply(Wp > 0).tocsr()
    lb3 = ch.stim["LB3"]
    rows = json.load(open(os.path.join(RESULTS, "e1c_phase.json")))["rows"]
    sc, y = [], []
    for r in rows:
        idx = np.sort(np.random.default_rng(r["seed"] * 1000 + r["k"]).choice(lb3, size=r["k"], replace=False))
        sc.append(core_drive(idx, n, top66, Wp) * r["hz"]); y.append(r["ignited"])
    th, acc_fit = best_threshold(sc, y)
    e6p = json.load(open(os.path.join(RESULTS, "e6_predictions.json")))["predictions"]
    ign = json.load(open(os.path.join(RESULTS, "e6_result.json")))["ignited"]
    s6 = [core_drive(np.array(p["cells"]), n, top66, Wp) * HZ for p in e6p]
    pred = [s > th for s in s6]
    acc6 = float(np.mean([p == g for p, g in zip(pred, ign)]))
    acc_176 = float(np.mean([p["pred_core"] == g for p, g in zip(e6p, ign)]))
    out = dict(t_load_s=t_load, t_meanfield_s=t_mf, t_rank_s=t_rank, n_P=int(len(P)), top66_same_as_e14=bool(same_top66),
               e1c_auc_top66=auc(sc, y), e1c_acc_fit=acc_fit, threshold=th, e6_acc_top66=acc6, e6_acc_176core=acc_176,
               e6_auc_top66=auc(s6, ign), e6_auc_176core=auc([p["core_score"] for p in e6p], ign))
    save("e15_pipeline.json", out)
    print("drive onto screen top-66: E1c AUC %.3f (fit acc %.2f) | E6 (post hoc) accuracy %.2f, AUC %.3f | "
          "176-core E6 accuracy %.2f, AUC %.3f" % (out["e1c_auc_top66"], acc_fit, acc6, out["e6_auc_top66"], acc_176,
                                                   out["e6_auc_176core"]))


if __name__ == "__main__":
    main()
