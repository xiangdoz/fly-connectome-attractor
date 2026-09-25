"""Reproducibility check for this repository's engine (run from engine/ after downloading the data, see data/README.md).
Re-runs a subset of stored results and requires EXACT equality:
  (1) batch.run_batch: the G0 protocol (9 columns) and G2 seed 8001 at every first-pulse drive (8 columns)
      -> MN9 rates must equal results/g0_g2.json;
  (2) sim.simulate: the G1c columns (6) -> active-neuron counts per block must equal results/g1c_fast.json."""
import json
import os

from common import np, two_pulse, job, rate, RESULTS
from stimuli import Channels
from batch import run_batch
from g0_g2 import g0_jobs, S1_GRID
from g1c_brian2_attractor import SEEDS as G1C_SEEDS, S1 as G1C_S1, SCHED as G1C_SCHED, aff_for
from sim import simulate


def main():
    ch = Channels()
    ref = json.load(open(os.path.join(RESULTS, "g0_g2.json")))["rows"]
    items = g0_jobs(ch)
    for s1 in S1_GRID:
        sched, aff = two_pulse(s1)
        items.append((("G2", s1, 8001), sched, job(ch, sched, aff, 8001)))
    res = run_batch(ch, [it[2] for it in items], seed=1)
    bad = 0
    for (tag, sched, _), r in zip(items, res):
        want = [x["mn9_hz"] for x in ref if x["tag"] == list(tag)][0]
        got = {p: rate(r, sched, ch, p) for p in want}
        same = all(abs(got[p] - want[p]) < 1e-9 for p in want)
        bad += not same
        print("%-40s %s" % (tag, "IDENTICAL" if same else "DIFFERENT %s vs %s" % (got, want)))
    fast = {(x["seed"], x["s1"]): x["active"] for x in json.load(open(os.path.join(RESULTS, "g1c_fast.json")))}
    cols = [(s, aff_for(s1)) for s in G1C_SEEDS for s1 in G1C_S1]
    cnt, _ = simulate(ch, G1C_SCHED, cols)
    for b, (s, s1) in enumerate((s, s1) for s in G1C_SEEDS for s1 in G1C_S1):
        act = [int((cnt[k, :, b] / (d / 1000.0) > 1).sum()) for k, (_, d) in enumerate(G1C_SCHED)]
        same = act == fast[(s, s1)]
        bad += not same
        print("sim.simulate seed %d s1 %.1f %s" % (s, s1, "IDENTICAL" if same else "DIFFERENT %s vs %s" % (act, fast[(s, s1)])))
    print("RESULT:", "ALL IDENTICAL" if bad == 0 else "%d MISMATCHES" % bad)


if __name__ == "__main__":
    main()
