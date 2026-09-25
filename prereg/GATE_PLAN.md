# Feasibility gate (written 2026-09-24, BEFORE any run in this project)

Question: is the lasting MN9 suppression after a strong gustatory drive (seen in an earlier exploratory
run, n = 3) a robust property of the Shiu et al. model, and can its circuit be localized? Go -> write a paper.
No-go -> stop; do not force a paper.

Protocol (open loop, no body): rest 100 ms -> PULSE 1 (LB3 sugar GRNs, Poisson at 100 Hz x s1, 500 ms)
-> gap (default 150 ms) -> PULSE 2 (LB3 at 100 Hz x 0.6, 500 ms). Readout: MN9_L spike rate in pulse 2.
Control: same schedule with s1 = 0 (no first pulse). Suppression index SI = 1 - rate2(s1) / rate2(s1 = 0),
paired by seed.

| Step | What | Seeds | Pass criterion (fixed now) |
|---|---|---|---|
| G0 copy integrity | re-run the original exploratory run with the copied engine | 3903-3905 (reuse of the original run, identical inputs) | MN9 per pulse identical to v5_explore.txt: 56.7 -> 10.7, 47.3 -> 43.3, single 47.3 |
| G1 not an engine artefact | Shiu's own Brian2 model, custom open-loop driver with the same RNG draw order; s1 in {1.2, 0.6, 0} | 8101-8103 | Brian2 itself shows SI(1.2) >= 0.5 in 3/3 seeds, and pulse-2 MN9 spike counts equal the fast engine's (exact) or differ by <= 5% |
| G2 robust | fast engine, s1 in {0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0} | 8001-8010 (fresh) | SI(1.2) >= 0.5 in >= 9/10 seeds; report the full dose-response |
| G3 recovery (descriptive) | s1 = 1.2, gap in {150, 500, 1000, 2000} ms | 8001-8005 | none (reported as is) |
| G4 localizable (feasibility) | gap-window activity difference strong vs control (same-seed prefix trick), then in-silico ablation of top candidate groups during the gap/pulse 2 | 8001-8005 | at least one candidate set of <= 50 neurons whose ablation raises SI(1.2) to <= 0.2 while leaving control pulse-2 MN9 within 20% |

Decision: GO if G0, G1, G2 pass (G4 strongly preferred: the paper needs a localized
circuit). NO-GO otherwise. Results of every step are reported whether they pass or not.
