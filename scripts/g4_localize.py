"""G4: localize the circuit behind the lasting MN9 suppression (GATE_PLAN.md).
  python g4_localize.py screen            -> per-neuron spike counts per block, s1 in {0, 1.2}, seeds 8001-8005
                                             (self-check: MN9 counts must equal run_batch's in g0_g2.json)
  python g4_localize.py ablate <set> <k>  -> zero the OUTPUT weights (W[:, j] = 0) of the top-k neurons of a
                                             candidate ranking <set>, re-run s1 in {0, 1.2}; report SI
Simulation = engine/lifcore2.run with a drive that reproduces engine/batch.run_batch's
RNG order exactly (open loop, contact rate 100 Hz x multiplier)."""
import json
import os
import sys
import time

from common import np, two_pulse, save, RESULTS
import lifcore2
from stimuli import Channels, R_CONTACT, DT_MS
from batch import ENGINE, get_W

SEEDS = (8001, 8002, 8003, 8004, 8005)
S1 = (0.0, 1.2)


def simulate(ch, W, s1_list, seeds, gap=150):
    """All columns share one schedule. Returns (sched, counts[nblk, n, B], column tags)."""
    tags = [(s1, s) for s1 in s1_list for s in seeds]
    B = len(tags)
    sched, _ = two_pulse(1.0, gap=gap)
    names = ch.stim_names
    gi = {nm: i for i, nm in enumerate(names)}
    groups = [ch.stim[nm] for nm in names]
    steps_per = [int(round(d / DT_MS)) for _, d in sched]
    blk_of = np.concatenate([[k] * L for k, L in enumerate(steps_per)])
    T = len(blk_of)
    col_rng, per_blk = [], []
    for s1, s in tags:
        rng = np.random.default_rng(s)
        for t_, g_ in ch.stim.items():                       # same dead-cell draw as run_batch (dropout 0)
            rng.random(len(g_))
        col_rng.append(rng)
        _, aff = two_pulse(s1, gap=gap)
        per_blk.append([[(gi[c], float(m)) for c, m in aff.get(state, {}).items()] for state, _ in sched])
    dt_s = DT_MS * 1e-3

    def drive(t):
        k = blk_of[t]
        hits = {}
        for b in range(B):
            for g, m in per_blk[b][k]:
                h = col_rng[b].random(len(groups[g])) < R_CONTACT * 1.0 * m * dt_s
                if h.any():
                    hits.setdefault(g, np.zeros((len(groups[g]), B)))[:, b] += h
        return [(groups[g], np.minimum(hm, 1.0)) for g, hm in hits.items()]

    n = W.shape[0]
    cnt = np.zeros((len(sched), n, B), np.int32)

    def on_step(t, spk):
        cnt[blk_of[t]] += spk

    lifcore2.run(W, B, T, drive, ch.all_stim, on_step, seed=1, log_every=0, label="g4",
                 dtype=getattr(np, ENGINE["dtype"]), store=ENGINE["store"], kernel=ENGINE.get("kernel", "numpy"),
                 prop=ENGINE.get("prop", "W"))
    return sched, cnt, tags


def metadata(ch):
    import pandas as pd
    from stimuli import COMP
    ids = pd.read_csv(COMP).iloc[:, 0].values
    cls = pd.read_csv("data/classification.csv.gz")
    ct = pd.read_csv("data/consolidated_cell_types.csv.gz").drop_duplicates("root_id")
    nt = pd.read_csv("data/neurons.csv.gz", usecols=["root_id", "nt_type"])
    m = pd.DataFrame({"root_id": ids}).merge(cls, on="root_id", how="left").merge(ct, on="root_id", how="left") \
        .merge(nt, on="root_id", how="left")
    return m


def screen():
    t0 = time.time()
    ch = Channels()
    W = get_W().astype(getattr(np, ENGINE["dtype"]))
    sched, cnt, tags = simulate(ch, W, S1, SEEDS)
    mn9 = ch.rec[ch.mn9_channels[0]]
    # self-check against run_batch (g0_g2.json, same seeds, same schedule)
    g = json.load(open(os.path.join(RESULTS, "g0_g2.json")))["rows"]
    ref = {(x["tag"][1], x["tag"][2]): x["mn9_hz"] for x in g if x["tag"][0] == "G2"}
    ok = True
    for b, (s1, s) in enumerate(tags):
        mine = {st: cnt[k, mn9, b] / (d / 1000.0) for k, (st, d) in enumerate(sched) if st in ("p1", "p2")}
        ok &= all(abs(mine[st] - ref[(s1, s)][st]) < 1e-9 for st in mine)
    print("self-check vs run_batch (MN9 per pulse, 10 columns):", "IDENTICAL" if ok else "MISMATCH", flush=True)
    names = [st for st, _ in sched]
    ig, ip2 = names.index("gap"), names.index("p2")
    strong = [b for b, (s1, _) in enumerate(tags) if s1 == 1.2]
    ctrl = [b for b, (s1, _) in enumerate(tags) if s1 == 0.0]
    dur = {st: d / 1000.0 for st, d in sched}
    rate = lambda k, cols: cnt[k][:, cols].mean(1) / dur[names[k]]            # Hz, mean over seeds
    d_gap = rate(ig, strong) - rate(ig, ctrl)
    d_p2 = rate(ip2, strong) - rate(ip2, ctrl)
    W_ = get_W()
    out_sign = np.sign(np.asarray(W_.sum(0)).ravel())                         # net sign of each neuron's outputs
    np.savez_compressed(os.path.join(RESULTS, "g4_screen.npz"), d_gap=d_gap, d_p2=d_p2, out_sign=out_sign,
                        gap_strong=rate(ig, strong), gap_ctrl=rate(ig, ctrl), p2_strong=rate(ip2, strong),
                        p2_ctrl=rate(ip2, ctrl), mn9=mn9)
    meta = metadata(ch)
    print("neurons active in the GAP: strong %d, control %d (mean rate > 1 Hz)"
          % ((rate(ig, strong) > 1).sum(), (rate(ig, ctrl) > 1).sum()))
    for label, d in (("gap", d_gap), ("pulse 2", d_p2)):
        inh = np.where(out_sign < 0)[0]
        top = inh[np.argsort(-d[inh])][:15]
        print("top INHIBITORY neurons by rate increase (strong - control) during %s:" % label)
        for j in top:
            r = meta.iloc[j]
            print("   idx %6d  +%6.1f Hz  %-10s %-14s %-12s %s" % (j, d[j], r.super_class, str(r["class"])[:14],
                                                                str(r.primary_type)[:12], r.nt_type))
    print("minutes %.1f" % ((time.time() - t0) / 60), flush=True)


def ranking(name):
    z = np.load(os.path.join(RESULTS, "g4_screen.npz"))
    inh = np.where(z["out_sign"] < 0)[0]
    key = {"inh_gap": z["d_gap"], "inh_p2": z["d_p2"]}[name]
    return inh[np.argsort(-key[inh])]


def ablate(set_name, k):
    t0 = time.time()
    ch = Channels()
    top = ranking(set_name)[:k]
    W = get_W().astype(getattr(np, ENGINE["dtype"])).tocsc(copy=True)
    for j in top:                                          # zero column j = neuron j's outgoing synapses
        W.data[W.indptr[j]:W.indptr[j + 1]] = 0.0
    W = W.tocsr()
    W.eliminate_zeros()
    sched, cnt, tags = simulate(ch, W, S1, SEEDS)
    mn9 = ch.rec[ch.mn9_channels[0]]
    names = [st for st, _ in sched]
    p2 = {tag: cnt[names.index("p2"), mn9, b] / 0.5 for b, tag in enumerate(tags)}
    si = [1 - p2[(1.2, s)] / p2[(0.0, s)] if p2[(0.0, s)] > 0 else np.nan for s in SEEDS]
    z = json.load(open(os.path.join(RESULTS, "g0_g2.json")))["rows"]
    intact_ctrl = np.mean([x["mn9_hz"]["p2"] for x in z if x["tag"][0] == "G2" and x["tag"][1] == 0.0
                           and x["tag"][2] in SEEDS])
    ctrl = np.mean([p2[(0.0, s)] for s in SEEDS])
    res = dict(set=set_name, k=k, neurons=top.tolist(), si=si, p2=[[list(t), v] for t, v in p2.items()],
               ctrl_p2=ctrl, intact_ctrl_p2=intact_ctrl, minutes=(time.time() - t0) / 60)
    save("g4_ablate_%s_%d.json" % (set_name, k), res)
    print("ablate %s top-%d: SI per seed %s (mean %.2f) | control pulse-2 MN9 %.1f Hz vs intact %.1f Hz (%+.0f%%)"
          % (set_name, k, np.round(si, 2).tolist(), np.nanmean(si), ctrl, intact_ctrl, 100 * (ctrl / intact_ctrl - 1)),
          flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "screen":
        screen()
    else:
        ablate(sys.argv[2], int(sys.argv[3]))
