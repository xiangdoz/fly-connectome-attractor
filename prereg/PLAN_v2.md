# PLAN v2 — hidden bistability of the whole-brain connectome model (written 2026-09-25, before any v2 run)

Why v2: the feasibility gate (GATE_PLAN.md) failed for the ORIGINAL question (a local feeding-suppression
circuit). The gate data revealed something else: the Shiu et al. (2024) FlyWire LIF model is bistable -- a
gustatory drive above a threshold switches it into a self-sustained state of ~8,100 neurons (mostly
olfactory / mushroom body) that persists >= 2 s without input and silences MN9. All v2 claims are tested on
FRESH seeds 8201-8299 (never used before); gate-phase seeds 8001-8103 are exploratory only.

Framing (a network-biology method with a biological case study):
  "Predicting and dissecting self-sustained activity states from connectome wiring": (i) a simulation screen
  that maps which sensory inputs ignite a whole-brain attractor; (ii) a wiring-only predictor of the attractor's
  membership (mean-field fixed point on the signed connectome), validated against simulation; (iii) a
  perturbation analysis that finds the sub-networks necessary to sustain it; (iv) consequences for users of
  connectome-constrained models, and minimal model changes that remove it.

| ID | Question | Design | Seeds | Pre-set criterion / report |
|---|---|---|---|---|
| E1 | Does Shiu et al.'s OWN protocol ignite? Which inputs ignite? | (a) all sugar/water GRN types together at 20/50/100/150/200 Hz for 1 s, then 1 s silence; (b) each gustatory type alone at 100 and 200 Hz | a: 8201-8205; b: 8206-8208 | ignited = > 1,000 neurons active (> 1 Hz) in the last 250 ms of silence. Report the fraction ignited per condition (no pass/fail) |
| E2 | Does the state end on its own? | ignite (LB3 1.2, 500 ms), then 10 s silence, 1-s windows | 8211-8215 | report active count per second; "persistent" if > 1,000 active at 10 s in >= 4/5 |
| E3 | Which sub-networks are necessary? | ablate (W[:, j] = 0) one cell class at a time from the start: Kenyon_Cell, ALPN, olfactory (ORN), ALLN, LHLN, APL, DAN, MBON; plus "all sensory neurons receive no synapses" (zero rows of sensory neurons); LB3 1.2 | 8216-8220 | necessary = ignition in <= 1/5 seeds while the no-ablation control ignites in >= 4/5 |
| E4 | Can wiring alone predict the attractor? | mean-field rate fixed point r = f(W r + I) on the signed Shiu connectome (f = LIF-like saturating transfer, parameters fixed from the model's constants, NOT fitted to the ignited set), started from high activity; compare the predicted active set with the simulated ignited set | uses E1/E2 simulated sets | report precision / recall / Jaccard vs a degree-matched random-set baseline; no pass/fail |
| E5 | Minimal fix that removes bistability but keeps validated behaviour | (a) no synapses onto sensory neurons; (b) spike-frequency adaptation added to the LIF (if time allows) | 8221-8230 | fix works = no ignition at LB3 1.2 in 5/5 AND the sugar -> MN9 dose-response (20-200 Hz) within 20% of the original model below the ignition threshold |

Rules: all runs on the copied engine in this folder; CPU only; no API. Every result is reported whether it
supports the story or not. Any change to this plan after seeing v2 data is logged as a deviation in LOG.md.
