"""Shared helpers. Run every script with the working directory = connectome-history/engine
(the copied engine reads engine.json and data/ relative to the cwd)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", "engine"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
if os.path.abspath(os.getcwd()) != ENGINE:
    raise SystemExit("run with cwd = %s" % ENGINE)
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, "shiu_code"))

import numpy as np  # noqa: E402

LEAD, PULSE, PROBE_GAIN = 100, 500, 0.6


def two_pulse(s1, gap=150, channel="LB3", s2=PROBE_GAIN, tail=0):
    """rest LEAD -> p1 (channel at 100 Hz x s1, PULSE ms) -> gap -> p2 (100 Hz x s2, PULSE ms) [-> tail]."""
    sched = [("empty", LEAD), ("p1", PULSE), ("gap", gap), ("p2", PULSE)]
    if tail:
        sched.append(("tail", tail))
    aff = {"p1": ({channel: s1} if s1 > 0 else {}), "p2": {channel: s2}}
    return sched, aff


def job(ch, sched, aff, seed):
    nerve = dict(aff=aff, eff=ch.mn9_channels[0], theta=np.inf)      # open loop: no body feedback
    return dict(nerve=nerve, pert={}, sched=sched, seed=seed)


def rate(result, sched, ch, state):
    i = [k for k, _ in sched].index(state)
    return float(result["blk_counts"][i, ch.mn9_channels[0]]) / (sched[i][1] / 1000.0)


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)
    return path
