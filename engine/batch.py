"""Open-loop batched simulation: many independent trials run as the columns of one lifcore2 batch.
job = dict(nerve=dict(aff={state: {gustatory_type: rate multiplier}}), sched=[(state, ms), ...], seed=int)
During a state, every cell of a listed type receives Poisson input at 100 Hz x multiplier.
Random numbers: each column draws from its own numpy default_rng(job seed) in a fixed order -- one draw per
gustatory type at the start, then, per 0.1-ms step, one draw per driven type -- the same order as the Brian2
drivers in scripts/g1_brian2.py and scripts/g1c_brian2_attractor.py (matched-seed comparisons).
Jobs may differ in length; shorter jobs are padded with no input."""
import json as _json
import os as _os

import numpy as np

import lifcore2
from lifcore import load_shiu
from stimuli import Channels, R_CONTACT, DT_MS  # noqa: F401

_W = None
ENGINE = dict(dtype="float32", store="abs", rng_mode="episode")
if _os.path.exists("engine.json"):
    ENGINE.update(_json.load(open("engine.json")))


def get_W():
    global _W
    if _W is None:
        n, src, dst, w, _ = load_shiu()
        _W = lifcore2.make_W(n, src, dst, w)
    return _W


def run_batch(ch, jobs, seed=0, label="", log_every=0, keep_totals=False, dtype=None, store=None, kernel=None,
              prop=None):
    """Returns, per job, dict(blk_counts=(n_blocks, n_motor) spike counts of every motor neuron per schedule block
    [, totals=(n,) spike counts of every neuron if keep_totals])."""
    dtype = dtype or getattr(np, ENGINE["dtype"])
    store = store or ENGINE["store"]
    kernel = kernel or ENGINE.get("kernel", "numpy")
    prop = prop or ENGINE.get("prop", "W")
    B = len(jobs)
    names = ch.stim_names
    gi = {nm: i for i, nm in enumerate(names)}
    groups = [ch.stim[nm] for nm in names]
    nsteps = [sum(int(round(d / DT_MS)) for _, d in j["sched"]) for j in jobs]
    T = max(nsteps)
    nblk = max(len(j["sched"]) for j in jobs)
    blk = np.full((T, B), nblk, np.int32)           # nblk = padding block
    for b, j in enumerate(jobs):
        t0 = 0
        for k, (_state, d) in enumerate(j["sched"]):
            L = int(round(d / DT_MS))
            blk[t0:t0 + L, b] = k
            t0 += L
    col_rng, col_types = [], []
    for b, j in enumerate(jobs):
        rng = np.random.default_rng(j["seed"])
        for _t, g in ch.stim.items():              # one draw per gustatory type (kept for the fixed RNG order)
            rng.random(len(g))
        col_rng.append(rng)
        aff = j["nerve"].get("aff", {})
        col_types.append([[(gi[c], float(m)) for c, m in aff.get(state, {}).items()] for state, _ in j["sched"]])
    dt_s = DT_MS * 1e-3
    cols = np.arange(B)
    blkc = np.zeros((B, nblk + 1, len(ch.rec)), np.int32)

    def drive(t):
        hits = {}
        for b in range(B):
            k = blk[t, b]
            if k >= len(col_types[b]):
                continue
            for g, m in col_types[b][k]:
                h = col_rng[b].random(len(groups[g])) < R_CONTACT * 1.0 * m * dt_s
                if h.any():
                    hits.setdefault(g, np.zeros((len(groups[g]), B)))[:, b] += h
        return [(groups[g], np.minimum(hm, 1.0)) for g, hm in hits.items()]

    def on_step(t, spk):
        r = spk[ch.rec]
        if r.any():
            blkc[cols, blk[t]] += r.T

    W = get_W() if dtype == np.float32 else get_W().astype(dtype)
    counts = lifcore2.run(W, B, T, drive, ch.all_stim, on_step, seed=seed, log_every=log_every, label=label,
                          dtype=dtype, store=store, kernel=kernel, prop=prop)
    out = []
    for b, j in enumerate(jobs):
        r = dict(blk_counts=blkc[b, :len(j["sched"])].copy())
        if keep_totals:
            r["totals"] = counts[:, b].copy()
        out.append(r)
    return out
