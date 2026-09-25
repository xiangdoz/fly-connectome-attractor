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
| `e6_prospective.py predict` / `simulate` | prospective test of the core-drive predictor (predictions frozen before simulation) | Fig. 2b |
| `e5_fix.py`, `e5c_sign_fix.py`, `e7_global_fixes.py` | adaptation, targeted re-signing, global changes | Fig. 3 |
| `e8_visual.py` | looming-sensitive visual input (LC4, LPLC2) | Sec. 4.2 |
| `g4_localize.py`, `g1b_precision.py` | exploratory gate analyses (reported in the plans) | -- |

Seeds: gate phase 8001-8103 (exploratory); all confirmatory analyses use fresh seeds from 8201 on.

## Licence

Shiu et al.'s `model.py` / `utils.py`: MIT (see `engine/shiu_code/LICENSE`). Licence for the rest of this repository:
to be added by the author.
