"""E16 (logged before running; wiring only): robustness of the annotation-uncertainty screen to the input used for the
mean-field persistent set and to the membership threshold, and a second-round screen on the corrected wiring."""
import json
import os
import time

from scipy.sparse.linalg import eigs

from common import np, save, RESULTS
from e3b_alln import meta
from e4_meanfield import predict
from e5c_sign_fix import reweighted
from lif import TAU
from stimuli import Channels
from batch import get_W


def screen(W, P, m):
    A = (W[P][:, P] * (TAU / 1000.0)).tocsc()
    lam, V = eigs(A.tocsr(), k=1, which="LR", maxiter=20000, tol=1e-10)
    _, U = eigs(A.T.tocsr(), k=1, which="LR", maxiter=20000, tol=1e-10)
    v, u = np.real(V[:, 0]), np.real(U[:, 0])
    v, u = (v if v.sum() > 0 else -v), (u if u.sum() > 0 else -u)
    colsum = np.asarray(A.sum(0)).ravel()
    cand = np.nonzero(m.nt_type.isna().values[P] & (colsum > 0))[0]
    d = np.array([-2.0 * float(u[A.indices[A.indptr[j]:A.indptr[j + 1]]] @ A.data[A.indptr[j]:A.indptr[j + 1]]) * v[j]
                  / float(u @ v) for j in cand])
    return float(np.real(lam[0])), P[cand[np.argsort(d)]], np.sort(d)


def main():
    t0 = time.time()
    ch = Channels()
    m = meta()
    n = len(m)
    W = get_W().astype(np.float64).tocsr()
    core = set(np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0].tolist())
    ref66 = set(json.load(open(os.path.join(RESULTS, "e14_rank.json")))["ranked_global"][:66])
    orn = np.nonzero(((m.super_class == "sensory") & (m["class"] == "olfactory")).values)[0]
    vm3 = np.nonzero(((m.super_class == "sensory") & (m.primary_type == "ORN_VM3")).values)[0]
    sugar = np.concatenate([ch.stim[t] for t in ch.stim_names if ch.modality[t] == "sugar/water"])
    inputs = {"LB3 200 Hz": (ch.stim["LB3"], 200.0), "all sugar GRNs 200 Hz": (sugar, 200.0),
              "all ORNs 30 Hz": (orn, 30.0), "all ORNs 100 Hz": (orn, 100.0), "ORN_VM3 100 Hz": (vm3, 100.0)}
    rows = []
    for name, (idx, hz) in inputs.items():
        _, r2, it = predict(W, idx, hz, n)
        for th in (1.0, 5.0):
            P = np.nonzero(r2 > th)[0]
            if len(P) < 50:
                rows.append(dict(input=name, thresh=th, n_P=int(len(P)), note="no persistent set"))
                print("%-22s th %.0f Hz | |P| = %d (no persistent set)" % (name, th, len(P)), flush=True)
                continue
            lam, ranked, _ = screen(W, P, m)
            r = dict(input=name, thresh=th, n_P=int(len(P)), lam=lam, n_cand=int(len(ranked)),
                     core_top45=float(np.mean([g in core for g in ranked[:45]])),
                     core_top66=float(np.mean([g in core for g in ranked[:66]])),
                     overlap_ref66=int(len(ref66 & set(ranked[:66].tolist()))))
            rows.append(r)
            print("%-22s th %.0f Hz | |P| %5d | eig %.3f | candidates %3d | core in top45 %.2f, top66 %.2f | top66 overlap with E14 %d/66"
                  % (name, th, r["n_P"], lam, r["n_cand"], r["core_top45"], r["core_top66"], r["overlap_ref66"]), flush=True)
    # second round on the corrected wiring
    Wc = reweighted(np.array(sorted(core)), "inhibitory").astype(np.float64).tocsr()
    second = []
    for name in ("LB3 200 Hz", "all ORNs 100 Hz"):
        idx, hz = inputs[name]
        _, r2, _ = predict(Wc, idx, hz, n)
        P = np.nonzero(r2 > 1.0)[0]
        entry = dict(input=name, n_P=int(len(P)))
        if len(P) >= 50:
            lam, ranked, d = screen(Wc, P, m)
            entry.update(lam=lam, n_cand=int(len(ranked)),
                         top20_classes=m.iloc[ranked[:20]]["class"].fillna("NA").value_counts().to_dict(),
                         top20_types=m.iloc[ranked[:20]]["primary_type"].fillna("NA").value_counts().head(8).to_dict(),
                         P_classes=m.iloc[P]["class"].fillna("NA").value_counts().head(8).to_dict())
        second.append(entry)
        print("second round, corrected wiring, %s: %s" % (name, entry), flush=True)
    save("e16_screen_robustness.json", dict(rows=rows, second_round=second, minutes=(time.time() - t0) / 60))
    print("minutes %.1f" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
