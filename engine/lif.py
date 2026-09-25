"""Constants of the Shiu et al. (2024) LIF model (philshiu/Drosophila_brain_model, model.py).
    dv/dt = (v_0 - v + g) / t_mbr        dg/dt = -g / tau
    v_0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms, t_rfc = 2.2 ms, t_dly = 1.8 ms,
    w_syn = 0.275 mV per synapse (signed), Poisson kick = w_syn x f_poi = 68.75 mV, time step 0.1 ms."""
V0, VRST, VTH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, F_POI, R_POI = 0.275, 250.0, 150.0
DT = 0.1
