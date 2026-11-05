Instructions to run:
First Setup:

cd ~/Documents/GitHub/N03-front
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn matplotlib joblib

Run FRONT:
cd defenses/front
python main.py ../../data/checkpoint/cells/ -format .cell -c checkpoint

Normalize:
cd ../..
python utils/overhead.py defenses/results/ranpad2_1105_1633/ -format .cell
python utils/norm.py defenses/results/ranpad2_1105_1633/

Run CUMUL:
cd attacks/cumul
mkdir -p results
python extract.py ../../data/checkpoint/cells/
python extract.py ../../defenses/results/ranpad2_1105_1633_norm/
python main.py results/cells.npy
python main.py results/ranpad2_1105_1633_norm.npy

Sanity Check:
head -20 data/checkpoint/cells/0-0.cell
head -20 defenses/results/ranpad2_1105_1633/0-0.cell


## Repeated-trial results (Reah)

Ran `bash run_trials.sh 10` on the checkpoint dataset (10 sites × 100 visits, closed world).
FRONT settings: 1100 client / 1100 server dummy packets, padding window 1–14 s.
The undefended attack was run 3 times; FRONT was run 10 times (new random padding each run).

| Condition | Runs | kFP accuracy (mean ± std) | Bandwidth overhead (mean ± std) |
|---|---|---|---|
| No defense | 3 | 98.7% ± 0.0% | 0% |
| FRONT | 10 | 89.7% ± 0.6% | 32.6% ± 0.4% |

Per-run numbers: `logs/trials.csv`. Summary: `logs/trials_summary.txt`. Raw FRONT output: `logs/front_trial_*.log`.

**Sanity checks:** all 1,010 input traces loaded, and FRONT wrote 1,010 output traces in every run.
Measured overhead (32.6%) matches the ~33% reported in the paper. The undefended accuracy is
identical across runs because the train/test splits are fixed.

**Mac setup note:** needs `numpy pandas scikit-learn matplotlib`. `extract.py`, `new_main.py`
and `overhead.py` were changed to use `mp.get_context("fork")`, the same fix Riya applied
to FRONT, so they run on macOS.