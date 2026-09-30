"""E11 (descriptive, wiring only; added after E5c, logged): linear-stability proxy for the attractor.
On the sub-network S of neurons in the self-sustained state (E1 consensus, 8,083 cells), compute the eigenvalue with the
largest real part of A = tau * W_S (W in mV per spike, tau = 5 ms; units mV per Hz) for: the original signs, the corrected
signs (150 unknown-nt ALLN inhibitory), and 20 controls in which 150 random excitatory cells of S (matched count) are
re-signed. Also: the same for S without the 150 cells (removed). Larger leading eigenvalue = stronger recurrent
excitation available to sustain activity. This is a proxy, not a stability proof (the LIF gain is not included)."""
import os

import numpy as np
from scipy.sparse.linalg import eigs

from common import save, RESULTS
from e3b_alln import meta
from lif import TAU
from batch import get_W


def lead(A):
    vals = eigs(A, k=6, which="LR", return_eigenvectors=False, maxiter=5000, tol=1e-6)
    return float(np.max(vals.real))


def main():
    m = meta()
    W = get_W().astype(np.float64).tocsr()
    e = np.load(os.path.join(RESULTS, "e1_last_active.npz"))
    keys = list(e.keys())
    freq = np.zeros(W.shape[0])
    for k in keys:
        freq[e[k]] += 1
    S = np.nonzero(freq >= 0.5 * len(keys))[0]
    c2 = set(np.nonzero(((m["class"] == "ALLN") & m.nt_type.isna()).values)[0].tolist())
    tau_s = TAU / 1000.0
    WS = (W[S][:, S] * tau_s).tocsc()
    pos = {j: i for i, j in enumerate(S)}
    core_in_S = [pos[j] for j in c2 if j in pos]

    def resign(cols_local):
        A = WS.copy()
        for j in cols_local:
            A.data[A.indptr[j]:A.indptr[j + 1]] = -np.abs(A.data[A.indptr[j]:A.indptr[j + 1]])
        return A.tocsr()

    out = dict(n_S=len(S), core_in_S=len(core_in_S))
    out["original"] = lead(WS.tocsr())
    out["core_resigned"] = lead(resign(core_in_S))
    keep = np.array([i for i in range(len(S)) if i not in set(core_in_S)])
    out["core_removed"] = lead(WS[keep][:, keep].tocsr())
    colsum = np.asarray(WS.sum(0)).ravel()
    exc_local = np.nonzero(colsum > 0)[0]
    rng = np.random.default_rng(0)
    ctrl = []
    for _ in range(20):
        pick = rng.choice(np.setdiff1d(exc_local, core_in_S), size=len(core_in_S), replace=False)
        ctrl.append(lead(resign(pick)))
    out["random_resign_controls"] = ctrl
    save("e11_spectrum.json", out)
    print("S = %d cells; core cells in S = %d" % (len(S), len(core_in_S)))
    print("leading Re(eig) of tau*W_S: original %.3f | 150 core re-signed %.3f | core removed %.3f"
          % (out["original"], out["core_resigned"], out["core_removed"]))
    print("random matched re-signing (20 draws): mean %.3f, min %.3f, max %.3f"
          % (np.mean(ctrl), np.min(ctrl), np.max(ctrl)))


if __name__ == "__main__":
    main()
