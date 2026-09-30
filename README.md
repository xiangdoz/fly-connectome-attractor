# A hidden self-sustained state in the whole-brain *Drosophila* connectome model

Code and results for the paper *Wiring predicts and explains a hidden self-sustained state in a whole-brain
connectome model of Drosophila*.

The leaky integrate-and-fire model of Shiu et al. (2024, *Nature*), built on the FlyWire v783 connectome, is bistable:
driving enough sugar gustatory receptor neurons together switches the network into a state in which about 8,100
neurons keep firing for at least 10 s without input. This repository contains the simulator, every experiment script,
the raw results behind every number and figure in the paper, and the plans with pass criteria that were written
before the corresponding runs (`prereg/`).

## Layout

| folder | contents |
|---|---|
| `engine/` | batched re-implementation of Shiu et al.'s Brian2 model (`lifcore2.py`, float64, numba kernel), stimulation groups (`stimuli.py`), open-loop batch runner (`batch.py`), and Shiu et al.'s original `model.py` (MIT licence, `engine/shiu_code/LICENSE`) used for the Brian2 replications |
| `scripts/` | one script per experiment (see table below) |
| `results/` | raw outputs (JSON / NPZ) |
| `figures/figs.py` | regenerates the paper's figures from `results/` |
| `prereg/` | analysis plans and pass criteria written before running |
| `data/README.md` | where to obtain the connectome and annotation files, with checksums |

## Setup

Python 3.12 with numpy, scipy, pandas, pyarrow, numba (and brian2 for the replication scripts). Put the data files
in `engine/data/` (see `data/README.md`). Every script runs with `engine/` as the working directory, e.g.

```
cd engine
python ../scripts/verify_release.py     # re-runs a subset of stored results; must print ALL IDENTICAL
# note: eigenvalues (e11, e14) use ARPACK with a random start vector and reproduce to ~1e-14, not bit for bit
```

## Experiments

| script | what it does | paper |
|---|---|---|
| `g0_g2.py` | probe response vs first-pulse drive, 10 fresh seeds | Fig. 1a |
| `g1_brian2.py`, `g1c_brian2_attractor.py` | matched-seed replication in Shiu et al.'s own Brian2 model | Methods |
| `g3_recovery.py`, `e2_e3.py e2` | persistence of the state (2 s and 10 s without input) | Fig. 1b |
| `e2_e3.py e3`, `e3b_alln.py` | class-level and antennal-lobe ablations | Fig. 1c |
| `e1_ignition_map.py`, `e1c_phase.py` | which stimuli ignite the network | Fig. 2a |
| `e4_meanfield.py` | wiring-only mean-field prediction of attractor membership | Sec. 4.3 |
| `e14_screen.py rank` / `extras` / `simulate` | annotation-uncertainty screen: eigenvalue-sensitivity ranking of uncertain cells in the mean-field set, baselines, and validation by simulation | Sec. 4.5, Fig. 5 |
| `e4b_composition.py` | membership by cell class, simulated vs predicted | Fig. 4 |
| `e6_prospective.py predict` / `simulate` | prospective test of the core-drive predictor (predictions frozen before simulation) | Sec. 4.6, Fig. 2b |
| `e5_fix.py`, `e5c_sign_fix.py`, `e7_global_fixes.py` | adaptation, targeted re-signing, global changes | Fig. 3 |
| `e8_visual.py` | looming-sensitive visual input (LC4, LPLC2) | Sec. 4.2 |
| `e13_olfactory.py` | single ORNs, single glomeruli and all ORNs, original vs corrected model | Sec. 4.2, 4.7 |
| `e9_other_stimuli.py` | bitter, low-salt, JO-C/E, JO-F and all gustatory neurons | Discussion |
| `e10_fix_stress.py` | stress test of the corrected model | Sec. 4.7 |
| `e12_partial_resign.py` | re-signing only part of the core (random subsets, two evidence rules) | Sec. 4.7, Fig. 5c |
| `e11_spectrum.py` | leading eigenvalue of the attractor sub-network, original vs corrected vs random | Sec. 4.4 |
| `g4_localize.py`, `g1b_precision.py` | exploratory gate analyses (reported in the plans) | -- |

Seeds: gate phase 8001-8103 (exploratory); all confirmatory analyses use fresh seeds from 8201 on.

## Licence

Shiu et al.'s `model.py` / `utils.py`: MIT (see `engine/shiu_code/LICENSE`). Everything else: MIT (see `LICENSE`).
