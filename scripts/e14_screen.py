"""E14 (logged before running): eigenvalue-sensitivity screening of uncertain transmitter annotations.
Part 'rank' is wiring only: on the mean-field predicted persistent set P, first-order change of the leading eigenvalue
of tau*W_P when each uncertain excitatory cell is re-signed. Part 'simulate' validates the top-k against random k."""
import json
import os
import sys
import time

import pandas as pd
from scipy.sparse.linalg import eigs

from common import np, save, RESULTS
from e3b_alln import meta
from e5_fix import SHIU21
from e5c_sign_fix import reweighted
from e11_spectrum import lead
from lif import TAU
from lifcore import to_index
from stimuli import Channels, COMP
from sim import simulate, weights

KS = (23, 45, 90)


def leading_pair(A):
    lam, V = eigs(A, k=1, which="LR", maxiter=20000, tol=1e-10)
    mu, U = eigs(A.T.tocsr(), k=1, which="LR", maxiter=20000, tol=1e-10)
    v, u = np.real(V[:, 0]), np.real(U[:, 0])
    if v.sum() < 0:
        v = -v
    if u.sum() < 0:
        u = -u
    return float(np.real(lam[0])), float(np.real(mu[0])), u, v


def rank():
    m = meta()
    W = weights().tocsc()
    P = np.sort(np.load(os.path.join(RESULTS, "e4_pred_set_drive2.0.npy")))
    tau_s = TAU / 1000.0
    A = (W[P][:, P] * tau_s).tocsc()
    lam, lam_T, u, v = leading_pair(A.tocsr())
    uv = float(u @ v)
    colsum = np.asarray(A.sum(0)).ravel()
    unk = m.nt_type.isna().values[P]
    cand = np.nonzero(unk & (colsum > 0))[0]
    d = np.array([-2.0 * float(u[A.indices[A.indptr[j]:A.indptr[j + 1]]] @ A.data[A.indptr[j]:A.indptr[j + 1]]) * v[j] / uv
                  for j in cand])
    order = cand[np.argsort(d)]                       # most negative first = largest predicted drop
    core = set(np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0].tolist())
    glob = P[order]
    out = dict(n_P=len(P), lam=lam, lam_T=lam_T, n_candidates=len(cand), n_core_in_candidates=int(sum(g in core for g in P[cand])),
               ranked_global=[int(g) for g in glob], pred_drop=[float(x) for x in np.sort(d)])
    for k in KS + (100,):
        top = glob[:k]
        out["top%d_core_frac" % k] = float(np.mean([g in core for g in top]))
        out["top%d_classes" % k] = m.iloc[top]["class"].fillna("NA").value_counts().to_dict()
        Ak = reweighted(top, "inhibitory")
        out["top%d_exact_lam" % k] = lead((Ak[P][:, P] * tau_s).tocsr())
        out["top%d_first_order_lam" % k] = lam + float(np.sort(d)[:k].sum())
    save("e14_rank.json", out)
    print("P = %d cells; leading eig %.3f (transpose %.3f); uncertain excitatory candidates in P: %d (core cells among them: %d)"
          % (len(P), lam, lam_T, len(cand), out["n_core_in_candidates"]))
    for k in KS + (100,):
        print("top-%-3d | core fraction %.2f | exact eig after re-signing %.3f (first-order %.3f) | classes %s"
              % (k, out["top%d_core_frac" % k], out["top%d_exact_lam" % k], out["top%d_first_order_lam" % k],
                 dict(list(out["top%d_classes" % k].items())[:5])), flush=True)


SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]


def simulate_part():
    t0 = time.time()
    r = json.load(open(os.path.join(RESULTS, "e14_rank.json")))
    m = meta()
    ch = Channels()
    P = np.sort(np.load(os.path.join(RESULTS, "e4_pred_set_drive2.0.npy")))
    W0 = weights().tocsc()
    colsum = np.array([W0.data[W0.indptr[j]:W0.indptr[j + 1]].sum() for j in P])
    pool = P[m.nt_type.isna().values[P] & (colsum > 0)]
    rng = np.random.default_rng(14)
    conds = {}
    for k in KS:
        conds["top-%d" % k] = np.array(r["ranked_global"][:k])
        for i in range(3):
            conds["random-%d #%d" % (k, i + 1)] = rng.choice(pool, size=k, replace=False)
    sugar = [t for t in ch.stim_names if ch.modality[t] == "sugar/water"]
    shiu20, _ = to_index(SHIU21, pd.read_csv(COMP).iloc[:, 0].values)
    e1 = json.load(open(os.path.join(RESULTS, "e1_ignition_map.json")))["rows"]
    orig = {(rr["tag"][2], rr["tag"][3]): rr["mn9_hz"][1] for rr in e1 if rr["tag"][:2] == ["a", "SHIU20"]}
    dur_last, dur_stim = SCHED[-1][1] / 1000.0, SCHED[1][1] / 1000.0
    out = []
    for name, idx in conds.items():
        cols, tags = [], []
        for s in (8311, 8312, 8313):
            cols.append((s, {"stim": {"LB3": 1.2}})); tags.append(("ign", s))
            cols.append((s, {"stim": {t: 2.0 for t in sugar}})); tags.append(("ign", s))
        if name.startswith("top"):
            for hz in (100, 150, 200):
                for s in (8201, 8202, 8203):
                    cols.append((s, {"stim": {"SHIU20": hz / 100.0}})); tags.append(("shiu_%d" % hz, s))
        cnt, mcnt = simulate(ch, SCHED, cols, W=reweighted(idx, "inhibitory"), extra_groups={"SHIU20": shiu20})
        last = [int((cnt[-1, :, b] / dur_last > 1).sum()) for b in range(len(cols))]
        ign = sum(last[b] > 1000 for b, (t, _) in enumerate(tags) if t == "ign")
        ratio = {}
        if name.startswith("top"):
            for hz in (100, 150, 200):
                new = [mcnt[1, b] / dur_stim for b, (t, s) in enumerate(tags) if t == "shiu_%d" % hz]
                ratio[hz] = float(np.mean(new) / np.mean([orig[(hz, s)] for s in (8201, 8202, 8203)]))
        core = set(np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0].tolist())
        out.append(dict(cond=name, k=len(idx), core_frac=float(np.mean([int(i) in core for i in idx])), ignited=int(ign),
                        last_active=[last[b] for b, (t, _) in enumerate(tags) if t == "ign"], mn9_ratio=ratio))
        print("%-14s | core frac %.2f | ignited %d/6 | last-window %s | SHIU20 MN9 ratio %s"
              % (name, out[-1]["core_frac"], ign, out[-1]["last_active"],
                 " ".join("%d:%.2f" % (h, x) for h, x in ratio.items()) or "-"), flush=True)
        save("e14_simulate.json", dict(conds=out, minutes=(time.time() - t0) / 60))
    print("minutes %.1f" % ((time.time() - t0) / 60))


def extras():
    """Added after the first ranking result (logged): baseline rankings, exact eigenvalue for the random-k sets used
    in 'simulate' (same generator and draw order), and core flags of the ranked list (for Fig. 5a)."""
    m = meta()
    W = weights().tocsc()
    P = np.sort(np.load(os.path.join(RESULTS, "e4_pred_set_drive2.0.npy")))
    tau_s = TAU / 1000.0
    A = (W[P][:, P] * tau_s).tocsc()
    colsum = np.asarray(A.sum(0)).ravel()
    rowsum = np.asarray(np.abs(A).sum(1)).ravel()
    cand = np.nonzero(m.nt_type.isna().values[P] & (colsum > 0))[0]
    core = set(np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0].tolist())
    out_all = np.asarray(W.sum(0)).ravel()
    ranks = {"out-weight within P": -colsum[cand], "in*out within P": -(colsum[cand] * rowsum[cand]),
             "out-weight whole brain": -out_all[P[cand]]}
    res = {}
    for name, key in ranks.items():
        glob = P[cand[np.argsort(key)]]
        res[name] = {str(k): (float(np.mean([g in core for g in glob[:k]])),
                              lead((reweighted(glob[:k], "inhibitory")[P][:, P] * tau_s).tocsr())) for k in KS}
        print("%-24s %s" % (name, res[name]), flush=True)
    save("e14_baselines.json", res)
    W0 = weights().tocsc()
    cs = np.array([W0.data[W0.indptr[j]:W0.indptr[j + 1]].sum() for j in P])
    pool = P[m.nt_type.isna().values[P] & (cs > 0)]
    rng = np.random.default_rng(14)
    rnd = {}
    for k in KS:
        rnd[str(k)] = [lead((reweighted(rng.choice(pool, size=k, replace=False), "inhibitory")[P][:, P] * tau_s).tocsr())
                       for _ in range(3)]
    save("e14_random_eig.json", rnd)
    print("random-k exact eigenvalues", rnd)
    r = json.load(open(os.path.join(RESULTS, "e14_rank.json")))
    flags = [g in core for g in r["ranked_global"]]
    save("e14_core_flags.json", dict(is_core=flags))
    print("first non-core candidate at rank", flags.index(False) + 1)


if __name__ == "__main__":
    {"rank": rank, "simulate": simulate_part, "extras": extras}[sys.argv[1]]()
