"""E1c (exploratory addition after E1, logged): ignition phase diagram -- number of recruited sugar GRNs x rate.
Random subsets of LB3 (both sides; subset drawn with the column's seed) of size k in {20, 40, 60, 80, 100, 122},
Poisson 100/150/200 Hz for 1 s, then 1 s silence; fresh seeds 8231-8233 (a new subset per seed)."""
import time

from common import np, save
from stimuli import Channels
from sim import simulate

SCHED = [("empty", 100), ("stim", 1000)] + [("s%d" % i, 250) for i in range(4)]
KS, RATES, SEEDS = (20, 40, 60, 80, 100, 122), (100, 150, 200), (8231, 8232, 8233)


def main():
    t0 = time.time()
    ch = Channels()
    lb3 = ch.stim["LB3"]
    extra, cols, tags = {}, [], []
    for k in KS:
        for s in SEEDS:
            name = "SUB_%d_%d" % (k, s)
            extra[name] = np.sort(np.random.default_rng(s * 1000 + k).choice(lb3, size=k, replace=False))
            for hz in RATES:
                cols.append((s, {"stim": {name: hz / 100.0}}))
                tags.append((k, hz, s))
    cnt, mcnt = simulate(ch, SCHED, cols, extra_groups=extra)
    dur = SCHED[-1][1] / 1000.0
    rows = []
    for b, (k, hz, s) in enumerate(tags):
        last = int((cnt[-1, :, b] / dur > 1).sum())
        rows.append(dict(k=k, hz=hz, seed=s, active_last=last, ignited=last > 1000))
    save("e1c_phase.json", dict(sched=SCHED, rows=rows, minutes=(time.time() - t0) / 60))
    print("ignited / 3 seeds   (rows: number of LB3 cells stimulated; columns: rate)")
    print("   k   | " + " | ".join("%3d Hz" % h for h in RATES))
    for k in KS:
        print("  %3d  | " % k + " | ".join("  %d/3 " % sum(r["ignited"] for r in rows if r["k"] == k and r["hz"] == h) for h in RATES))
    print("minutes %.1f" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
