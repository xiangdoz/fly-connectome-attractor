"""G1b (exploratory, added after G1): are the 2 Brian2/fast mismatches explained by sensitivity of the
ignited state to floating-point rounding? Same engine, same seeds (8101-8103), float64 vs float32,
s1 in {1.2, 0.6, 0.0}. Prediction if yes: float32 == float64 in non-ignited runs, differs in ignited runs."""
from common import np, two_pulse, job, save
from stimuli import Channels
from batch import run_batch

SEEDS, S1 = (8101, 8102, 8103), (1.2, 0.6, 0.0)


def main():
    ch = Channels()
    items = [((s, s1),) + two_pulse(s1) for s in SEEDS for s1 in S1]
    jobs = [job(ch, sch, aff, s) for (s, _), sch, aff in items]
    out = {}
    for dt in ("float64", "float32"):
        res = run_batch(ch, jobs, seed=1, log_every=0, dtype=getattr(np, dt))
        out[dt] = [r["blk_counts"][:, ch.mn9_channels[0]].tolist() for r in res]
    rows = []
    for i, ((s, s1), _, _) in enumerate(items):
        a, b = out["float64"][i], out["float32"][i]
        rows.append(dict(seed=s, s1=s1, f64=a, f32=b, same=a == b))
        print("seed %d s1 %.1f | float64 %s | float32 %s | %s" % (s, s1, a, b, "same" if a == b else "DIFFERENT"))
    save("g1b_precision.json", rows)


if __name__ == "__main__":
    main()
