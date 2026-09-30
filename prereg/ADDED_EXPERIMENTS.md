# Experiments added after PLAN_v2 (plans copied from the lab log; each was written before its runs)

## E12 (written before running): how much of the core must be re-signed? (added after the Eckstein finding)
- Motivation: Eckstein et al. report AL local neurons mispredicted 5HT instead of ACh, so some of the 150 may truly be excitatory.
  Of the 150, 90 are excitatory in the model and 60 already inhibitory. Treating 5HT/DA/OA votes as ACh gives exactly the model's
  90 (rule A = current model). Using only fast-transmitter votes: majority (ACh > GABA+Glu) leaves 16 excitatory (re-sign 74);
  plurality (ACh > max(GABA, Glu)) leaves 55 excitatory (re-sign 35).
- Conditions (outputs of the chosen cells made inhibitory; everything else as in the original model):
  random partial re-signing of k of the 90 excitatory core cells, k = 23, 45, 68 (3 random subsets each, numpy rng seed 12);
  rule "fast-majority" (74 re-signed); rule "fast-plurality" (35 re-signed).
- Tests: LB3 120 Hz and all sugar/water GRNs 200 Hz, 1 s then 1 s silence; fresh seeds 8291-8293. Ignition criterion as before.
  Also the E11 spectral proxy for each condition (wiring only).
- Descriptive, no pass/fail. Reported whatever the outcome. If most partial conditions still ignite, the paper will state
  that the attractor needs most of the 90 re-signed and that the true identities of these cells decide the regime.

## E13 (written before running): olfactory stimulation
- Original model and corrected model (150 unknown-nt ALLN inhibitory). 1 s stimulus, then 1 s silence; fresh seeds 8301-8302.
- Stimuli: (a) single ORN at 20 Hz, 5 ORNs from 5 different ORN types (numpy rng seed 13); (b) the same single ORNs at 100 Hz;
  (c) all ORNs of one type (one glomerulus) at 30 Hz, 5 types (rng seed 13); (d) the same types at 100 Hz; (e) all 2,279 ORNs at 30 Hz.
- Report: neurons active (>1 Hz) during the stimulus, and in the last 250 ms of silence (ignited if > 1,000).
- Pre-set consequences: if any single-ORN or single-glomerulus condition ignites the original model, the paper drops
  "only population-level stimulation reaches the state" and reports olfactory ignition; if olfactory input recruits
  ~10,000 cells only during the stimulus, the paper says so and distinguishes recruitment from persistence.

## E14 (written before running): eigenvalue-sensitivity screening of uncertain annotations (wiring only + validation)
- Motivation: turn the E11 spectral observation into an algorithm that finds the sign-sensitive
  cells WITHOUT simulation or ablation search.
- Algorithm: P = mean-field predicted persistent set (results/e4_pred_set.npy; wiring only, 10,615 cells). A = tau*W_P.
  Leading eigenvalue lambda with right/left eigenvectors v, u. For each excitatory cell j in P without a neuron-level
  transmitter call, first-order change of lambda if j is re-signed inhibitory: d_j = -2 (u^T A[:,j]) v_j / (u^T v).
  Rank candidates by -d_j.
- Report: overlap of the top-k with the 150-cell ALLN core found by ablation (k = 23, 45, 90); class composition of
  the top 100; first-order vs exact lambda after re-signing the top-k.
- Validation (simulation): re-sign top-k (k = 23, 45, 90) vs k random uncertain excitatory cells in P (3 draws each,
  numpy rng seed 14). LB3 120 Hz and all sugar 200 Hz, 1 s + 1 s silence, fresh seeds 8311-8313. For top-k sets also
  SHIU20 MN9 at 100/150/200 Hz on seeds 8201-8203 vs E1 original.
- Pre-set criterion ("screening works"): top-45 includes >= 50% core cells AND re-signing top-45 gives 0/6 ignitions
  AND the random-45 controls ignite in at least half of their runs. Reported either way.
- E14 file fix (before any E14 result was used): P must be results/e4_pred_set_drive2.0.npy (10,615 cells, the set used in the paper); e4_pred_set.npy holds a 35-cell low-drive set. First rank run on the wrong file discarded.

## E15 (written before running; POST HOC, wiring only): end-to-end wiring-only pipeline and its cost
- (a) Timing on this CPU: mean-field fixed point for LB3 at 200 Hz (-> persistent set P) + eigenvalue-sensitivity ranking.
- (b) Replace the ablation-defined 176-cell core in the E6 drive score by the screen's top 66 (wiring only). Freeze the
  threshold on E1c exactly as in E6, apply to the 42 E6 sets, compare with E6 outcomes (already known -> post hoc;
  reported as such, not as a prospective result). Also AUC on E1c.

## E16 (written before running; wiring only): robustness of the screen
- Vary the input used to compute the mean-field persistent set P: LB3 200 Hz (reference), all sugar/water GRNs 200 Hz,
  all ORNs 30 Hz, all ORNs 100 Hz, one glomerulus (ORN_VM3) 100 Hz. Vary the membership threshold: 1 Hz (reference), 5 Hz.
- For each: |P|, number of candidates, fraction of core cells in the top 45 / top 66, overlap of the top 66 with E14's top 66.
- Second round: run the same screen on the corrected wiring (150 unknown-nt ALLN inhibitory), LB3 200 Hz and all ORNs 100 Hz;
  report |P|, leading eigenvalue and the classes of the top 20 candidates (what, if anything, the screen flags next).
- Criterion ("robust"): top 45 >= 90% core cells in every variant of the original wiring. Reported either way.
