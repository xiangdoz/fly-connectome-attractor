"""Data for the overview figure (wiring only, no new experiment): a class-sorted block of tau*W on the mean-field set P,
the leading right eigenvector on that block, the screen's ranked drops, and the mean synapse-level votes of lLN1_bc."""
import json
import os

import pandas as pd
from scipy.sparse.linalg import eigs

from common import np, RESULTS
from e3b_alln import meta
from lif import TAU
from sim import weights


def main():
    m = meta()
    nt = pd.read_csv("data/neurons.csv.gz")
    m = m.merge(nt[["root_id", "gaba_avg", "glut_avg", "ach_avg", "ser_avg", "da_avg", "oct_avg"]], on="root_id", how="left")
    W = weights().astype(np.float64).tocsr()
    P = np.sort(np.load(os.path.join(RESULTS, "e4_pred_set_drive2.0.npy")))
    A = (W[P][:, P] * (TAU / 1000.0)).tocsr()
    lam, V = eigs(A, k=1, which="LR", maxiter=20000, tol=1e-10)
    v = np.real(V[:, 0]); v = v if v.sum() > 0 else -v
    core = ((m["class"] == "ALLN") & m.nt_type.isna()).values
    cls = m["class"].fillna("NA").values[P]
    blocks = []
    for c, n in (("ALLN", 50), ("ALPN", 40), ("Kenyon_Cell", 30)):
        idx = np.nonzero(cls == c)[0]
        idx = idx[np.argsort(-v[idx])][:n]
        blocks.append(idx)
    sel = np.concatenate(blocks)
    sub = A[sel][:, sel].toarray()
    r = json.load(open(os.path.join(RESULTS, "e14_rank.json")))
    lln = m[m.primary_type == "lLN1_bc"]
    votes = {k: float(lln[k + "_avg"].mean()) for k in ("da", "ser", "ach", "gaba", "glut", "oct")}
    np.savez(os.path.join(RESULTS, "fig_overview_data.npz"), sub=sub, v_sub=v[sel], is_core=core[P][sel],
             cls=np.array([c for c, n in (("ALLN", 50), ("ALPN", 40), ("Kenyon_Cell", 30)) for _ in range(n)]),
             drop=-np.array(r["pred_drop"][:40]), lam=float(np.real(lam[0])))
    json.dump(dict(lLN1_bc_votes=votes, n_lLN1_bc=int(len(lln))), open(os.path.join(RESULTS, "fig_overview_votes.json"), "w"),
              indent=1)
    print("lam %.3f | block %s | core in ALLN block %d | votes %s" % (np.real(lam[0]), sub.shape, core[P][sel].sum(),
                                                                       {k: round(x, 3) for k, x in votes.items()}))


if __name__ == "__main__":
    main()
