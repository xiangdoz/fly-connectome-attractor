"""Generic open-loop simulation with per-block, per-neuron spike counts (same drive / RNG order as
engine/batch.run_batch; validated through g4_localize.simulate, which it generalizes).
columns: list of (seed, aff) with aff = {state: {gustatory_type: multiplier}}; all columns share `sched`."""
from common import np
import lifcore2
from stimuli import R_CONTACT, DT_MS
from batch import ENGINE, get_W


def weights(ablate=()):
    W = get_W().astype(getattr(np, ENGINE["dtype"]))
    if len(ablate):
        W = W.tocsc(copy=True)
        for j in ablate:
            W.data[W.indptr[j]:W.indptr[j + 1]] = 0.0          # neuron j's outgoing synapses
        W = W.tocsr()
        W.eliminate_zeros()
    return W


def simulate(ch, sched, columns, W=None, counts=True, extra_groups=None, sfa=None):
    """extra_groups: {name: neuron index array} usable in aff like a gustatory type (e.g. Shiu's 20 sugar
    GRNs); they are NOT part of the dead-cell draw, so the standard RNG stream is unchanged."""
    W = weights() if W is None else W
    B = len(columns)
    names = list(ch.stim_names) + sorted(extra_groups or {})
    gi = {nm: i for i, nm in enumerate(names)}
    groups = [ch.stim[nm] if nm in ch.stim else np.asarray(extra_groups[nm]) for nm in names]
    steps_per = [int(round(d / DT_MS)) for _, d in sched]
    blk_of = np.concatenate([[k] * L for k, L in enumerate(steps_per)])
    col_rng, per_blk = [], []
    for seed, aff in columns:
        rng = np.random.default_rng(seed)
        for _t, g_ in ch.stim.items():
            rng.random(len(g_))
        col_rng.append(rng)
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
    mn9 = ch.rec[ch.mn9_channels[0]]
    cnt = np.zeros((len(sched), n, B), np.int32) if counts else None
    mcnt = np.zeros((len(sched), B), np.int32)

    def on_step(t, spk):
        k = blk_of[t]
        if counts:
            cnt[k] += spk
        mcnt[k] += spk[mn9]

    targets = np.unique(np.concatenate([ch.all_stim] + [np.asarray(g) for g in (extra_groups or {}).values()]))
    if sfa is not None:                                  # E5: LIF + spike-frequency adaptation (lif_sfa.py)
        from lif_sfa import run_sfa
        run_sfa(W, B, len(blk_of), drive, targets, on_step, seed=1, d_a=sfa["d_a"], tau_a=sfa["tau_a"],
                dtype=getattr(np, ENGINE["dtype"]))
    else:
        lifcore2.run(W, B, len(blk_of), drive, targets, on_step, seed=1, log_every=0, label="sim",
                     dtype=getattr(np, ENGINE["dtype"]), store=ENGINE["store"], kernel=ENGINE.get("kernel", "numpy"),
                     prop=ENGINE.get("prop", "W"))
    return cnt, mcnt
