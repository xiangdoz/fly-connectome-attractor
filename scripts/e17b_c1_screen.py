"""E17b (logged before running; post hoc, wiring only): screen the C1 (fast-majority) wiring with P computed from all
ORNs at 30 Hz, the input that left a residual persistent state in E17, and from LB3 at 200 Hz."""
import json
import os

import pandas as pd

from common import np, save, RESULTS
from e3b_alln import meta
from e4_meanfield import predict
from e16_screen_robustness import screen
from e17_conventions import signed
from stimuli import Channels
from sim import weights


def main():
    ch = Channels()
    m = meta()
    nt = pd.read_csv("data/neurons.csv.gz")[["root_id", "gaba_avg", "glut_avg", "ach_avg"]]
    m = m.merge(nt, on="root_id", how="left")
    W0 = weights().tocsc()
    sign0 = np.sign(np.asarray(W0.sum(0)).ravel())
    nocall = m.nt_type.isna().values
    s1 = sign0.copy()
    fast_inh = (m.gaba_avg.fillna(0) + m.glut_avg.fillna(0) > m.ach_avg.fillna(0)).values
    s1[nocall & (sign0 != 0)] = np.where(fast_inh[nocall & (sign0 != 0)], -1, 1)
    W = signed(W0, s1).astype(np.float64).tocsr()
    orn = np.nonzero(((m.super_class == "sensory") & (m["class"] == "olfactory")).values)[0]
    out = []
    for name, (idx, hz) in {"all ORNs 30 Hz": (orn, 30.0), "LB3 200 Hz": (ch.stim["LB3"], 200.0)}.items():
        _, r2, _ = predict(W, idx, hz, len(m))
        P = np.nonzero(r2 > 1.0)[0]
        e = dict(input=name, n_P=int(len(P)), P_classes=m.iloc[P]["class"].fillna("NA").value_counts().head(8).to_dict())
        if len(P) >= 50:
            lam, ranked, _ = screen(W, P, m)
            e.update(lam=lam, n_cand=int(len(ranked)),
                     top20_classes=m.iloc[ranked[:20]]["class"].fillna("NA").value_counts().to_dict(),
                     top20_types=m.iloc[ranked[:20]]["primary_type"].fillna("NA").value_counts().head(8).to_dict())
        out.append(e)
        print(e, flush=True)
    sim = json.load(open(os.path.join(RESULTS, "e17_conventions.json")))
    save("e17b_c1_screen.json", dict(rows=out))


if __name__ == "__main__":
    main()
