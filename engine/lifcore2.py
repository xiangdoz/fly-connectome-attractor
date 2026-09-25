"""Batched re-implementation of Shiu et al.'s Brian2 LIF model (philshiu/Drosophila_brain_model, model.py).
  dv/dt = (v_0 - v + g) / t_mbr, dg/dt = -g / tau (unless refractory); reset v = v_rst, g = 0;
  synapse on_pre g += w with delay t_dly; Poisson input v += w_syn x f_poi, Poisson targets have no refractory period;
  exact (linear) integration; synaptic input arriving during the target's refractory period is discarded (as observed
  in Brian2); Brian2's per-step order: integrate -> threshold -> synapses / Poisson -> reset.
Batched over B independent columns, event-driven; the numba kernel is compiled without fastmath."""
import time

import numpy as np
from scipy.sparse import csr_matrix

from lif import V0, VRST, VTH, T_MBR, TAU, T_RFC, T_DLY, W_SYN, F_POI, DT
from lifcore import make_W, taste_root_ids, load_shiu, to_index  # re-export

E_M = np.float32(np.exp(-DT / T_MBR))
E_G = np.float32(np.exp(-DT / TAU))
A_VG = np.float32(TAU / (TAU - T_MBR) * (np.exp(-DT / TAU) - np.exp(-DT / T_MBR)))


_KERNEL = None


def _get_kernel():
    """Fused per-step update (steps 1-5 + refractory countdown), parallel over neurons.
    Same operation order as the numpy path; compiled WITHOUT fastmath, so results are
    bit-identical (verified by kernel_check.py)."""
    global _KERNEL
    if _KERNEL is None:
        import numba

        @numba.njit(parallel=True, fastmath=False, cache=True)
        def k(v, g, ref, ring_slot, spk, base, e_m, a_vg, e_g, vth):
            n, B = v.shape
            for i in numba.prange(n):
                for b in range(B):
                    r = ref[i, b]
                    if r <= 0:
                        vv = v[i, b]
                        gg = g[i, b]
                        u = vv - base
                        vv = base + u * e_m + gg * a_vg
                        v[i, b] = vv
                        spk[i, b] = vv > vth
                        g[i, b] = gg * e_g + ring_slot[i, b]
                    else:
                        spk[i, b] = False
                        g[i, b] = 0.0
                        ref[i, b] = r - 1
                    ring_slot[i, b] = 0.0
        _KERNEL = k
    return _KERNEL


def run(W, B, steps, drive, poisson_targets, on_step=None, seed=0,
        log_every=2000, label="", dtype=np.float32, trace=None, store="abs", kernel="numpy", prop="W"):
    """drive(t) -> list of (neuron_idx, prob_per_column (B,)): Poisson events
    added to v. poisson_targets: every index that drive() may ever stimulate
    (they get rfc = 0, as in model.py). on_step(t, spk) after each step.
    Returns spike counts (n, B)."""
    n = W.shape[0]
    e_m, e_g, a_vg = dtype(np.exp(-DT / T_MBR)), dtype(np.exp(-DT / TAU)), dtype(
        TAU / (TAU - T_MBR) * (np.exp(-DT / TAU) - np.exp(-DT / T_MBR)))
    # store='dev' keeps u = v - V0 instead of v: algebraically identical, but in
    # float32 small subthreshold increments are not rounded away at |v| ~ 52 mV.
    base = dtype(0.0) if store == "dev" else dtype(V0)
    vth, vrst = dtype(VTH) - (dtype(V0) - base), dtype(VRST) - (dtype(V0) - base)
    v = np.full((n, B), base, dtype)
    g = np.zeros((n, B), dtype)
    ref = np.zeros((n, B), np.int16)
    # Brian2: non-refractory once timestep(t - lastspike) >= timestep(rfc), so
    # a 2.2 ms period blocks 21 steps after the spike step, not 22.
    nref = np.full(n, int(round(T_RFC / DT)) - 1, np.int16)
    nref[np.asarray(poisson_targets, int)] = 0
    D = int(round(T_DLY / DT))
    ring = np.zeros((D, n, B), dtype)
    counts = np.zeros((n, B), np.int32)
    rng = np.random.default_rng(seed)
    kick = dtype(F_POI * W_SYN)
    run.trace = []
    t0 = time.time()
    if kernel == "numba":
        kfun = _get_kernel()
        spk = np.zeros((n, B), bool)
    WT = W.T.tocsr() if prop == "T" else None      # rows = presynaptic neurons
    for t in range(steps):
        if kernel == "numba":
            slot = t % D
            kfun(v, g, ref, ring[slot], spk, base, e_m, a_vg, e_g, vth)
            for idx, p in drive(t):
                pp = np.asarray(p, np.float32)
                if pp.ndim == 1:
                    pp = pp[None, :]
                v[idx] += (rng.random((len(idx), B), dtype=np.float32) < pp) * kick
            if spk.any():
                r, c = np.nonzero(spk)
                v[r, c] = vrst
                g[r, c] = 0.0
                ref[r, c] = nref[r]
                counts[r, c] += 1
                if WT is not None:      # only the out-edges of neurons that spiked
                    ST = csr_matrix((np.ones(len(r), np.float32), (c, r)), shape=(B, n))
                    inc = (ST @ WT).T.tocoo()
                else:
                    S = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, B))
                    inc = (W @ S).tocoo()
                ring[slot][inc.row, inc.col] += inc.data
            if trace is not None:
                run.trace.append((v[trace, 0].copy() + (dtype(V0) - base), g[trace, 0].copy()))
            if on_step is not None:
                on_step(t, spk)
            if log_every and (t + 1) % log_every == 0:
                el = time.time() - t0
                print("  %s %d/%d steps, %.0f ms/step, %.1f min" %
                      (label, t + 1, steps, 1e3 * el / (t + 1), el / 60), flush=True)
            continue
        # groups: exact linear update for non-refractory neurons
        act = ref <= 0
        u = v - base
        v_new = base + u * e_m + g * a_vg
        g_new = g * e_g
        v = np.where(act, v_new, v)
        g = np.where(act, g_new, g)
        # thresholds
        spk = (v > vth) & act
        # synapses: delayed synaptic input, then Poisson input to v
        slot = t % D
        g += ring[slot]
        ring[slot].fill(0)
        # Shiu's Brian2 model: synaptic input arriving while the target is
        # refractory is lost (g stays 0 through the refractory period). Found by
        # a deterministic 2-neuron diff against Brian2 (tiny_diff.py).
        g = np.where(act, g, dtype(0.0))   # input during refractory is discarded (Brian2, empirically; see probe_rfc.py)
        for idx, p in drive(t):
            pp = np.asarray(p, np.float32)
            if pp.ndim == 1:                 # one probability per column
                pp = pp[None, :]             # else (len(idx), B) per neuron
            v[idx] += (rng.random((len(idx), B), dtype=np.float32) < pp) * kick
        # resets
        np.subtract(ref, 1, out=ref, where=ref > 0)
        if spk.any():
            r, c = np.nonzero(spk)
            v[r, c] = vrst
            g[r, c] = 0.0
            ref[r, c] = nref[r]
            counts[r, c] += 1
            S = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, B))
            inc = (W @ S).tocoo()
            ring[slot][inc.row, inc.col] += inc.data   # arrives at t + D
        if trace is not None:                # state at END of step t
            run.trace.append((v[trace, 0].copy() + (dtype(V0) - base), g[trace, 0].copy()))
        if on_step is not None:
            on_step(t, spk)
        if log_every and (t + 1) % log_every == 0:
            el = time.time() - t0
            print("  %s %d/%d steps, %.0f ms/step, %.1f min" %
                  (label, t + 1, steps, 1e3 * el / (t + 1), el / 60), flush=True)
    return counts
