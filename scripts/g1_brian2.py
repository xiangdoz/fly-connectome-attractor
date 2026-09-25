"""G1: is the suppression an artefact of the fast engine? Runs Shiu et al.'s OWN Brian2 model (shiu_code/
model.py) with an open-loop driver that mirrors engine/batch.py (same RNG
order): same dead-cell draw, then per 0.1-ms step one rng.random(len(group)) per driven type, hit if
< 100 Hz x multiplier x dt, kick = w_syn * f_poi added to the membrane voltage.
  python g1_brian2.py b2 <seed> <s1>   -> one Brian2 run, appends to results/g1_brian2.jsonl
  python g1_brian2.py fast             -> the same jobs on the fast engine -> results/g1_fast.json
  python g1_brian2.py report           -> comparison + gate verdict"""
import json
import os
import sys
import time

from common import np, two_pulse, job, rate, save, RESULTS
from stimuli import Channels, COMP, CON, R_CONTACT, DT_MS

SEEDS = (8101, 8102, 8103)
S1 = (1.2, 0.6, 0.0)


def brian2_run(ch, sched, aff, seed):
    from brian2 import prefs, ms, mV, second, Network, network_operation, defaultclock
    from brian2 import seed as b2seed
    prefs.codegen.target = "numpy"
    import model as sm
    params = dict(sm.default_params)
    neu, syn, _ = sm.create_model(COMP, CON, params)
    neu.rfc[ch.all_stim] = 0 * ms
    kick = float(params["w_syn"] * params["f_poi"] / mV) * 1e-3
    etype = np.array([k for k, d in sched for _ in range(int(round(d / DT_MS)))])
    block = np.concatenate([[i] * int(round(d / DT_MS)) for i, (_, d) in enumerate(sched)])
    steps = len(etype)
    rng = np.random.default_rng(seed)
    b2seed(seed)
    dead = {t: rng.random(len(g)) < 0.0 for t, g in ch.stim.items()}      # same draw order as engine/batch.py
    dt_s = DT_MS * 1e-3
    blk = np.zeros((len(sched), len(ch.rec)), np.int64)

    @network_operation(dt=defaultclock.dt, when="synapses")
    def afferent(t):
        k = int(round(float(t / ms) / DT_MS))
        if k >= steps:
            return
        for tname, mult in aff.get(etype[k], {}).items():                # open loop: contact rate only
            grp = ch.stim[tname]
            hit = (rng.random(len(grp)) < R_CONTACT * mult * dt_s) & ~dead[tname]
            if hit.any():
                idx = grp[hit]
                neu.v_[idx] = neu.v_[idx] + kick

    @network_operation(dt=defaultclock.dt, when="end")
    def efferent(t):
        k = int(round(float(t / ms) / DT_MS))
        if k >= steps:
            return
        now = float(t / second)
        blk[block[k]] += np.abs(neu.lastspike_[ch.rec] - now) < 0.5 * dt_s

    Network(neu, syn, afferent, efferent).run(steps * DT_MS * ms)
    return blk


def main():
    mode = sys.argv[1]
    ch = Channels()
    if mode == "b2":
        seed, s1 = int(sys.argv[2]), float(sys.argv[3])
        sched, aff = two_pulse(s1)
        t0 = time.time()
        blk = brian2_run(ch, sched, aff, seed)
        r = dict(seed=seed, s1=s1, sched=sched, mn9_counts=blk[:, ch.mn9_channels[0]].tolist(),
                 minutes=round((time.time() - t0) / 60, 2))
        with open(os.path.join(RESULTS, "g1_brian2.jsonl"), "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print("BRIAN2", r, flush=True)
    elif mode == "fast":
        from batch import run_batch
        items = [((s, s1),) + two_pulse(s1) for s in SEEDS for s1 in S1]
        res = run_batch(ch, [job(ch, sch, aff, s) for (s, _), sch, aff in items], seed=1, log_every=0)
        rows = [dict(seed=s, s1=s1, sched=sch, mn9_counts=r["blk_counts"][:, ch.mn9_channels[0]].tolist())
                for ((s, s1), sch, aff), r in zip(items, res)]
        save("g1_fast.json", rows)
        print("FAST", [(x["seed"], x["s1"], x["mn9_counts"]) for x in rows], flush=True)
    else:
        b2 = [json.loads(l) for l in open(os.path.join(RESULTS, "g1_brian2.jsonl"))]
        fa = {(x["seed"], x["s1"]): x for x in json.load(open(os.path.join(RESULTS, "g1_fast.json")))}
        print("seed  s1  | Brian2 MN9 counts per block | fast engine | pulse-2 Hz (B2 / fast)")
        ok_match, si_b2 = True, {}
        for x in sorted(b2, key=lambda z: (z["seed"], -z["s1"])):
            f = fa[(x["seed"], x["s1"])]
            b, fc = x["mn9_counts"], f["mn9_counts"]
            p2b, p2f = b[3] / 0.5, fc[3] / 0.5
            rel = abs(p2b - p2f) / max(p2f, 1e-9)
            ok_match &= (b == fc) or rel <= 0.05
            print("%d %.1f | %s | %s | %.1f / %.1f%s" % (x["seed"], x["s1"], b, fc, p2b, p2f, "  EXACT" if b == fc else ""))
            si_b2.setdefault(x["seed"], {})[x["s1"]] = p2b
        si = [1 - v[1.2] / v[0.0] for v in si_b2.values() if 1.2 in v and 0.0 in v and v[0.0] > 0]
        print("Brian2 SI(1.2) per seed:", [round(v, 2) for v in si])
        print("G1 PASS:", ok_match and len(si) == 3 and all(v >= 0.5 for v in si))


if __name__ == "__main__":
    main()
