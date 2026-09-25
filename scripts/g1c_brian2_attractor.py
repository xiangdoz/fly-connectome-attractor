"""G1c (added after the gate, logged): does Shiu et al.'s OWN Brian2 model also enter the self-sustained state?
Open-loop driver as in g1_brian2.py (same RNG draw order), but counting spikes of ALL 139,255 neurons per block.
Schedule: rest 100 ms -> LB3 at 100 Hz x s1 for 500 ms -> four 250-ms silent windows. s1 in {1.2, 0}; seeds 8101-8103.
  python g1c_brian2_attractor.py b2 <seed> <s1>  -> appends results/g1c_brian2.jsonl
  python g1c_brian2_attractor.py fast            -> same columns on the fast engine -> results/g1c_fast.json
  python g1c_brian2_attractor.py report"""
import json
import os
import sys
import time

from common import np, save, RESULTS
from stimuli import Channels, COMP, CON, R_CONTACT, DT_MS

SEEDS, S1 = (8101, 8102, 8103), (1.2, 0.0)
SCHED = [("empty", 100), ("p1", 500)] + [("s%d" % i, 250) for i in range(4)]


def aff_for(s1):
    return {"p1": ({"LB3": s1} if s1 > 0 else {})}


def brian2_run(ch, seed, s1):
    from brian2 import prefs, ms, mV, second, Network, network_operation, defaultclock
    from brian2 import seed as b2seed
    prefs.codegen.target = "numpy"
    import model as sm
    aff = aff_for(s1)
    params = dict(sm.default_params)
    neu, syn, _ = sm.create_model(COMP, CON, params)
    neu.rfc[ch.all_stim] = 0 * ms
    kick = float(params["w_syn"] * params["f_poi"] / mV) * 1e-3
    etype = np.array([k for k, d in SCHED for _ in range(int(round(d / DT_MS)))])
    block = np.concatenate([[i] * int(round(d / DT_MS)) for i, (_, d) in enumerate(SCHED)])
    steps = len(etype)
    rng = np.random.default_rng(seed)
    b2seed(seed)
    dead = {t: rng.random(len(g)) < 0.0 for t, g in ch.stim.items()}
    dt_s = DT_MS * 1e-3
    n = len(neu)
    cnt = np.zeros((len(SCHED), n), np.int32)

    @network_operation(dt=defaultclock.dt, when="synapses")
    def afferent(t):
        k = int(round(float(t / ms) / DT_MS))
        if k >= steps:
            return
        for tname, mult in aff.get(etype[k], {}).items():
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
        cnt[block[k]] += np.abs(neu.lastspike_ - float(t / second)) < 0.5 * dt_s

    Network(neu, syn, afferent, efferent).run(steps * DT_MS * ms)
    return cnt


def active(cnt):
    return [int((cnt[k] / (d / 1000.0) > 1).sum()) for k, (_, d) in enumerate(SCHED)]


def main():
    mode = sys.argv[1]
    ch = Channels()
    if mode == "b2":
        seed, s1 = int(sys.argv[2]), float(sys.argv[3])
        t0 = time.time()
        cnt = brian2_run(ch, seed, s1)
        r = dict(seed=seed, s1=s1, active=active(cnt), mn9=cnt[:, ch.rec[ch.mn9_channels[0]]].tolist(),
                 minutes=round((time.time() - t0) / 60, 1))
        with open(os.path.join(RESULTS, "g1c_brian2.jsonl"), "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print("BRIAN2", r, flush=True)
    elif mode == "fast":
        from sim import simulate
        cols = [(s, aff_for(s1)) for s in SEEDS for s1 in S1]
        cnt, mcnt = simulate(ch, SCHED, cols)
        rows = [dict(seed=s, s1=s1, active=[int((cnt[k, :, b] / (d / 1000.0) > 1).sum()) for k, (_, d) in enumerate(SCHED)],
                     mn9=mcnt[:, b].tolist()) for b, (s, s1) in enumerate((s, s1) for s in SEEDS for s1 in S1)]
        save("g1c_fast.json", rows)
        print("FAST", [(x["seed"], x["s1"], x["active"]) for x in rows], flush=True)
    else:
        b2 = [json.loads(l) for l in open(os.path.join(RESULTS, "g1c_brian2.jsonl"))]
        fa = {(x["seed"], x["s1"]): x for x in json.load(open(os.path.join(RESULTS, "g1c_fast.json")))}
        print("seed  s1 | active neurons per block [rest, pulse, silent 1-4]: Brian2 | fast engine")
        for x in sorted(b2, key=lambda z: (z["seed"], -z["s1"])):
            print("%d %.1f | %s | %s" % (x["seed"], x["s1"], x["active"], fa[(x["seed"], x["s1"])]["active"]))
        ign = [x["active"][-1] > 1000 for x in b2 if x["s1"] == 1.2]
        ctl = [x["active"][-1] for x in b2 if x["s1"] == 0.0]
        print("Brian2 ignited after s1 = 1.2: %d/%d | control last-window active: %s" % (sum(ign), len(ign), ctl))


if __name__ == "__main__":
    main()
