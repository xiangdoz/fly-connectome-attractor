"""G0 (copy integrity: re-run the original exploratory protocol, seeds 3903-3905) and
G2 (dose-response of the suppression, fresh seeds 8001-8010), one batch. See GATE_PLAN.md."""
import time

from common import np, two_pulse, job, rate, save   # noqa: F401  (sets sys.path)
from stimuli import Channels
from batch import run_batch

G0_SEEDS = (3903, 3904, 3905)
G2_SEEDS = tuple(range(8001, 8011))
S1_GRID = (0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0)


def g0_jobs(ch):
    """Exactly v5_tasks.explore(): pulses named p<i>_<drive>, 500 ms each, 150 ms empty after each."""
    out = []
    for s in G0_SEEDS:
        for label, drives, lead in (("strong(1.2) then 0.6", [1.2, 0.6], 100), ("0.6 then 0.6", [0.6, 0.6], 100),
                                    ("single 0.6 at 2nd time", [0.6], 750)):
            sched, aff = [("empty", lead)], {}
            for i, d in enumerate(drives):
                st = "p%d_%.1f" % (i, d)
                sched += [(st, 500), ("empty", 150)]
                aff[st] = {"LB3": d}
            out.append((("G0", label, s), sched, job(ch, sched, aff, s)))
    return out


def main():
    t0 = time.time()
    ch = Channels()
    items = g0_jobs(ch)
    for s in G2_SEEDS:
        for s1 in S1_GRID:
            sched, aff = two_pulse(s1)
            items.append((("G2", s1, s), sched, job(ch, sched, aff, s)))
    res = run_batch(ch, [it[2] for it in items], seed=1, label="g0g2", log_every=0)
    rows = []
    for (tag, sched, _), r in zip(items, res):
        pulses = [k for k, _ in sched if k != "empty" and k != "gap"]
        rows.append(dict(tag=list(tag), sched=sched, mn9_hz={p: rate(r, sched, ch, p) for p in pulses}))
    save("g0_g2.json", dict(rows=rows, minutes=(time.time() - t0) / 60))
    # ---- G0 report ----
    print("G0 copy integrity (expected 56.7 -> 10.7 | 47.3 -> 43.3 | 47.3):")
    for label in ("strong(1.2) then 0.6", "0.6 then 0.6", "single 0.6 at 2nd time"):
        hz = [list(x["mn9_hz"].values()) for x in rows if x["tag"][0] == "G0" and x["tag"][1] == label]
        print("   %-24s %s" % (label, np.round(np.mean(hz, 0), 1).tolist()))
    # ---- G2 report ----
    base = {s: x["mn9_hz"]["p2"] for x in rows for s in [x["tag"][2]] if x["tag"][0] == "G2" and x["tag"][1] == 0.0}
    print("G2 dose-response (pulse-2 MN9 Hz mean +- sd; SI = 1 - rate2/rate2(s1=0), paired by seed):")
    for s1 in S1_GRID:
        sub = [x for x in rows if x["tag"][0] == "G2" and x["tag"][1] == s1]
        r2 = np.array([x["mn9_hz"]["p2"] for x in sub])
        r1 = np.array([x["mn9_hz"]["p1"] for x in sub])
        si = np.array([1 - x["mn9_hz"]["p2"] / base[x["tag"][2]] if base[x["tag"][2]] > 0 else np.nan for x in sub])
        print("   s1 %.1f | pulse1 %5.1f Hz | pulse2 %5.1f +- %4.1f Hz | SI %5.2f +- %4.2f | SI>=0.5 in %d/%d"
              % (s1, r1.mean(), r2.mean(), r2.std(), np.nanmean(si), np.nanstd(si), int((si >= 0.5).sum()), len(si)))
    print("minutes %.1f" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
