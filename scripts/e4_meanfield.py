"""E4: can WIRING ALONE predict the self-sustained state? Deterministic mean-field of Shiu's LIF, no fitted
parameters: mean synaptic drive g_i = tau * sum_j W_ij r_j (W in mV per spike, signed), LIF rate for a constant
drive mu above rest: f(mu) = 0 if mu <= (v_th - v_0) else 1 / (t_rfc + t_mbr * ln(mu / (mu - (v_th - v_0)))).
Phase 1: stimulated neurons clamped at the stimulus rate, others iterate r <- (1-a) r + a f(g) to a fixed point.
Phase 2: stimulus removed (stimulated neurons now also follow f), iterate from phase 1's state.
Predicted persistent set = neurons with r > 1 Hz at the end of phase 2.
Outputs: predicted ignition threshold for LB3 drive, and overlap of the predicted set with simulated ignited sets
(precision / recall / Jaccard) vs a random baseline of the same size drawn proportional to in-degree."""
import glob
import json
import os

from common import np, save, RESULTS
from lif import V0, VTH, T_MBR, TAU, T_RFC
from stimuli import Channels
from batch import get_W

DTH = VTH - V0                         # 7 mV
TAU_S, TMBR_S, TRFC_S = TAU / 1000.0, T_MBR / 1000.0, T_RFC / 1000.0


def f(mu):
    out = np.zeros_like(mu)
    m = mu > DTH + 1e-9
    out[m] = 1.0 / (TRFC_S + TMBR_S * np.log(mu[m] / (mu[m] - DTH)))
    return out


def fixed_point(W, r0, clamp_idx=None, clamp_rate=0.0, a=0.2, iters=400, tol=1e-3):
    r = r0.copy()
    for it in range(iters):
        g = TAU_S * (W @ r)
        new = (1 - a) * r + a * f(g)
        if clamp_idx is not None:
            new[clamp_idx] = clamp_rate
        if np.max(np.abs(new - r)) < tol:
            return new, it
        r = new
    return r, iters


def predict(W, stim_idx, rate_hz, n):
    r1, _ = fixed_point(W, np.zeros(n), stim_idx, rate_hz)
    r2, it = fixed_point(W, r1)
    return r1, r2, it


def overlap(pred, true, n, indeg, rng, reps=200):
    P, T = set(pred.tolist()), set(true.tolist())
    inter = len(P & T)
    prec = inter / max(len(P), 1)
    rec = inter / max(len(T), 1)
    jac = inter / max(len(P | T), 1)
    p = indeg / indeg.sum()
    base = []
    for _ in range(reps):
        R = set(rng.choice(n, size=len(P), replace=False, p=p).tolist()) if len(P) else set()
        base.append(len(R & T) / max(len(R | T), 1))
    return dict(precision=prec, recall=rec, jaccard=jac, n_pred=len(P), n_true=len(T),
                random_jaccard_mean=float(np.mean(base)), random_jaccard_p99=float(np.percentile(base, 99)))


def main():
    ch = Channels()
    W = get_W().astype(np.float64).tocsr()
    n = W.shape[0]
    lb3 = ch.stim["LB3"]
    print("LB3 drive sweep (100 Hz x drive, all 122 LB3): predicted persistent-set size after the stimulus is removed")
    sweep = []
    for d in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0):
        r1, r2, it = predict(W, lb3, 100.0 * d, n)
        sweep.append(dict(drive=d, active_during=int((r1 > 1).sum()), persistent=int((r2 > 1).sum()), iters=it))
        print("   drive %.1f | active during stimulus %6d | persistent after %6d | iters %d" % (d, (r1 > 1).sum(), (r2 > 1).sum(), it), flush=True)
    # overlap with simulated ignited sets
    _, r2, _ = predict(W, lb3, 120.0, n)
    pred = np.nonzero(r2 > 1)[0]
    indeg = np.asarray((W != 0).sum(1)).ravel().astype(float) + 1.0
    rng = np.random.default_rng(0)
    res = dict(sweep=sweep, pred_size=len(pred))
    sim_sets = {}
    z = np.load(os.path.join(RESULTS, "g4_screen.npz"))
    sim_sets["gate_screen_gap(LB3 1.2, seeds 8001-8005, exploratory)"] = np.nonzero(z["gap_strong"] > 1)[0]
    e1 = os.path.join(RESULTS, "e1_last_active.npz")
    if os.path.exists(e1):
        e = np.load(e1)
        keys = list(e.keys())
        if keys:
            freq = np.zeros(n)
            for k in keys:
                freq[e[k]] += 1
            sim_sets["E1 ignited runs: active in >= 50%% of %d runs" % len(keys)] = np.nonzero(freq >= 0.5 * len(keys))[0]
    res["overlap"] = {}
    for name, true in sim_sets.items():
        o = overlap(pred, true, n, indeg, rng)
        res["overlap"][name] = o
        print("overlap with %s: precision %.2f recall %.2f Jaccard %.2f (pred %d, sim %d) | random same-size Jaccard %.3f (p99 %.3f)"
              % (name, o["precision"], o["recall"], o["jaccard"], o["n_pred"], o["n_true"], o["random_jaccard_mean"], o["random_jaccard_p99"]))
    np.save(os.path.join(RESULTS, "e4_pred_set.npy"), pred)
    save("e4_meanfield.json", res)


if __name__ == "__main__":
    main()
