"""G3 (pre-registered, descriptive): recovery of pulse-2 MN9 with the gap, s1 = 1.2 vs 0, gap in {500, 1000, 2000}
(150 ms is in g0_g2.json), seeds 8001-8005.
EXPLORATORY (added after the G4 screen): persistence of the self-sustained state -- after pulse 1
(s1 in {0, 0.8, 1.2}), 2 s of NO input in 250-ms windows: number of active neurons (>= 1 spike in the
window... reported as count of neurons with rate > 1 Hz) and MN9 rate per window, then a 0.6 probe."""
import time

from common import np, two_pulse, job, rate, save
from stimuli import Channels
from batch import run_batch
from sim import simulate

SEEDS = (8001, 8002, 8003, 8004, 8005)


def main():
    t0 = time.time()
    ch = Channels()
    # ---- pre-registered G3 ----
    items = []
    for gap in (500, 1000, 2000):
        for s1 in (0.0, 1.2):
            for s in SEEDS:
                sched, aff = two_pulse(s1, gap=gap)
                items.append(((gap, s1, s), sched, job(ch, sched, aff, s)))
    res = run_batch(ch, [it[2] for it in items], seed=1, log_every=0)
    g3 = [dict(gap=g, s1=s1, seed=s, p2=rate(r, sched, ch, "p2")) for ((g, s1, s), sched, _), r in zip(items, res)]
    print("G3 recovery (pulse-2 MN9 Hz, mean over 5 seeds; SI paired by seed):")
    for gap in (500, 1000, 2000):
        c = {x["seed"]: x["p2"] for x in g3 if x["gap"] == gap and x["s1"] == 0.0}
        st = [x for x in g3 if x["gap"] == gap and x["s1"] == 1.2]
        si = [1 - x["p2"] / c[x["seed"]] for x in st if c[x["seed"]] > 0]
        print("   gap %4d ms | control %.1f Hz | after 1.2: %.1f Hz | SI %.2f (per seed %s)"
              % (gap, np.mean(list(c.values())), np.mean([x["p2"] for x in st]), np.mean(si), np.round(si, 2).tolist()))
    # ---- exploratory persistence time course ----
    sched = [("empty", 100), ("p1", 500)] + [("w%d" % i, 250) for i in range(8)] + [("p2", 500)]
    cols = [(s, {"p1": ({"LB3": s1} if s1 else {}), "p2": {"LB3": 0.6}}) for s1 in (0.0, 0.8, 1.2) for s in SEEDS]
    tags = [(s1, s) for s1 in (0.0, 0.8, 1.2) for s in SEEDS]
    cnt, mcnt = simulate(ch, sched, cols)
    names = [k for k, _ in sched]
    tc = []
    for b, (s1, s) in enumerate(tags):
        act = [int((cnt[k, :, b] / (d / 1000.0) > 1).sum()) for k, (_, d) in enumerate(sched)]
        mhz = [float(mcnt[k, b] / (d / 1000.0)) for k, (_, d) in enumerate(sched)]
        tc.append(dict(s1=s1, seed=s, active=act, mn9_hz=mhz))
    print("\nEXPLORATORY persistence: active neurons per window [p1 | w0..w7 (250 ms each, NO input) | p2 probe 0.6]")
    for b, x in enumerate(tc):
        a = x["active"]
        print("   s1 %.1f seed %d | p1 %5d | %s | p2 %5d | MN9 p2 %.0f Hz" % (
            x["s1"], x["seed"], a[1], " ".join("%5d" % v for v in a[2:10]), a[10], x["mn9_hz"][10]))
    save("g3_recovery.json", dict(g3=g3, persistence=dict(sched=sched, rows=tc), minutes=(time.time() - t0) / 60))
    print("minutes %.1f" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
