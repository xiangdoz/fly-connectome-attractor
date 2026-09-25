"""E5(b): Shiu's LIF + spike-frequency adaptation (SFA), as a SEPARATE module (the copied engine is untouched).
Model: dv/dt = (v_0 - v + g - a) / t_mbr ; dg/dt = -g / tau ; da/dt = -a / tau_a ; on spike: v = v_rst, g = 0,
a += d_a. v and g are frozen during the refractory period (as in Brian2 / lifcore2); a keeps decaying.
Exact per-step integration: u' = u e_m + g a_vg - a a_va ; a' = a e_a ; with
a_va = tau_a / (tau_a - t_mbr) * (exp(-dt/tau_a) - exp(-dt/t_mbr)).
With d_a = 0 this is the numba path of lifcore2.run step for step (self-check in e5_fix.py)."""
import time

import numpy as np
from scipy.sparse import csr_matrix

from lif import V0, VRST, VTH, T_MBR, TAU, T_RFC, T_DLY, W_SYN, F_POI, DT

_K = None


def _kernel():
    global _K
    if _K is None:
        import numba

        @numba.njit(parallel=True, fastmath=False, cache=False)
        def k(v, g, a, ref, ring_slot, spk, base, e_m, a_vg, e_g, vth, e_a, a_va):
            n, B = v.shape
            for i in numba.prange(n):
                for b in range(B):
                    r = ref[i, b]
                    aa = a[i, b]
                    if r <= 0:
                        vv = v[i, b]
                        gg = g[i, b]
                        u = vv - base
                        vv = base + u * e_m + gg * a_vg - aa * a_va
                        v[i, b] = vv
                        spk[i, b] = vv > vth
                        g[i, b] = gg * e_g + ring_slot[i, b]
                    else:
                        spk[i, b] = False
                        g[i, b] = 0.0
                        ref[i, b] = r - 1
                    a[i, b] = aa * e_a
                    ring_slot[i, b] = 0.0
        _K = k
    return _K


def run_sfa(W, B, steps, drive, poisson_targets, on_step=None, seed=0, d_a=0.0, tau_a=100.0, dtype=np.float64):
    n = W.shape[0]
    e_m, e_g = dtype(np.exp(-DT / T_MBR)), dtype(np.exp(-DT / TAU))
    a_vg = dtype(TAU / (TAU - T_MBR) * (np.exp(-DT / TAU) - np.exp(-DT / T_MBR)))
    e_a = dtype(np.exp(-DT / tau_a))
    a_va = dtype(tau_a / (tau_a - T_MBR) * (np.exp(-DT / tau_a) - np.exp(-DT / T_MBR)))
    base = dtype(V0)
    vth, vrst = dtype(VTH), dtype(VRST)
    v = np.full((n, B), base, dtype)
    g = np.zeros((n, B), dtype)
    a = np.zeros((n, B), dtype)
    ref = np.zeros((n, B), np.int16)
    nref = np.full(n, int(round(T_RFC / DT)) - 1, np.int16)
    nref[np.asarray(poisson_targets, int)] = 0
    D = int(round(T_DLY / DT))
    ring = np.zeros((D, n, B), dtype)
    counts = np.zeros((n, B), np.int32)
    rng = np.random.default_rng(seed)
    kick = dtype(F_POI * W_SYN)
    kfun = _kernel()
    spk = np.zeros((n, B), bool)
    WT = W.T.tocsr()
    for t in range(steps):
        slot = t % D
        kfun(v, g, a, ref, ring[slot], spk, base, e_m, a_vg, e_g, vth, e_a, a_va)
        for idx, p in drive(t):
            pp = np.asarray(p, np.float32)
            if pp.ndim == 1:
                pp = pp[None, :]
            v[idx] += (rng.random((len(idx), B), dtype=np.float32) < pp) * kick
        if spk.any():
            r, c = np.nonzero(spk)
            v[r, c] = vrst
            g[r, c] = 0.0
            a[r, c] += dtype(d_a)
            ref[r, c] = nref[r]
            counts[r, c] += 1
            ST = csr_matrix((np.ones(len(r), np.float32), (c, r)), shape=(B, n))
            inc = (ST @ WT).T.tocoo()
            ring[slot][inc.row, inc.col] += inc.data
        if on_step is not None:
            on_step(t, spk)
    return counts
