"""E4b (descriptive, no simulation): class composition of the attractor -- simulated (cells active in >= 50% of the 19
ignited E1 runs) vs the wiring-only mean-field prediction (drive 2.0) -- plus per-class recall of the prediction."""
import os

import numpy as np

from common import save, RESULTS
from e3b_alln import meta


def main():
    m = meta()
    cls = m["class"].fillna(m.super_class).fillna("unannotated").values
    e = np.load(os.path.join(RESULTS, "e1_last_active.npz"))
    keys = list(e.keys())
    freq = np.zeros(len(m))
    for k in keys:
        freq[e[k]] += 1
    sim = np.nonzero(freq >= 0.5 * len(keys))[0]
    pred = np.load(os.path.join(RESULTS, "e4_pred_set_drive2.0.npy"))
    S, P = set(sim.tolist()), set(pred.tolist())
    names, counts = np.unique(cls[list(S | P)], return_counts=True)
    order = names[np.argsort(-counts)]
    rows = []
    for c in order:
        idx = set(np.nonzero(cls == c)[0].tolist())
        s, p = len(S & idx), len(P & idx)
        rows.append(dict(cls=str(c), simulated=s, predicted=p, both=len(S & P & idx),
                         recall=(len(S & P & idx) / s) if s else None))
    save("e4b_composition.json", dict(n_sim=len(S), n_pred=len(P), rows=rows))
    for r in rows[:14]:
        print("%-24s simulated %5d | predicted %5d | both %5d | recall %s" % (
            r["cls"], r["simulated"], r["predicted"], r["both"], "%.2f" % r["recall"] if r["recall"] is not None else "-"))


if __name__ == "__main__":
    main()
