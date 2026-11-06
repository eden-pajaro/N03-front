# Project Midpoint
**Course:** 424\
**Team:** N03\
**Members:** Reah Robinson, Riya Ramakrishnan, Eden Pajaro

## Instructions to run:

### Initial Setup:
Clone the repo and move to the project folder (like in the first command).\
Run the following commands to create a virtual environment and install dependencies.
```
cd ~/Documents/GitHub/N03-front
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn matplotlib joblib
```

### Run FRONT:
```
cd defenses/front
python main.py ../../data/checkpoint/cells/ -format .cell -c checkpoint
```

### Normalize:
```
cd ../..
python utils/overhead.py defenses/results/ranpad2_1105_1633/ -format .cell
python utils/norm.py defenses/results/ranpad2_1105_1633/
```

### Run CUMUL Website Fingerprinting Attack:
```
cd attacks/cumul
mkdir -p results
python extract.py ../../data/checkpoint/cells/
python extract.py ../../defenses/results/ranpad2_1105_1633_norm/
python main.py results/cells.npy
python main.py results/ranpad2_1105_1633_norm.npy
```

### Sanity Check:
```
cd ../..
head -20 data/checkpoint/cells/0-0.cell
head -20 defenses/results/ranpad2_1105_1633/0-0.cell
```

#### Results of Sanity Check:
Run without FRONT:
```
0.0     1
0.2516  1
0.4695  -1
0.4718  1
0.4719  1
0.7277  -1
0.7313  1
0.7314  1
0.975   -1
0.9751  -1
0.9751  -1
0.9751  -1
0.9752  -1
0.9752  -1
0.9752  -1
1.1849  -1
1.1915  1
1.1917  1
1.4026  -1
1.4027  -1
```
Run with FRONT:
```
0.0000  1
0.1847  888
0.2516  1
0.2572  888
0.3036  888
0.3375  888
0.4150  888
0.4695  -1
0.4718  1
0.4719  1
0.5142  888
0.5153  888
0.5629  888
0.6118  888
0.6390  888
0.6780  888
0.7277  -1
0.7313  1
0.7314  1
0.7653  888
```
#### What it Means:
Each row is a packet captured during a website visit. Every packet has two values: timestamp and direction. This sanity check shows that fake packets were successfully added by FRONT.

| Field | Value | Meaning |
|---|---|---|
| timestamp | any number | when the packet was sent, in seconds since the page started loading |
| direction | 1 | the direction of the action is computer --> website (eg. "give me the page") |
| direction | -1 | the direction of the action is website --> computer (eg. page content) |
| direction | 888 | a fake packet FRONT added |

## Team Contributions

| Team member | Bounded contribution | Work completed | Evidence to link |
|---|---|---|---|
| Eden Pajaro | Dataset preparation and documentation | Found the dataset, uploaded the checkpoint traces to `data/checkpoint/cells`, and organized the README | Dataset upload commit and README revision |
| Riya Ramakrishnan | Code compatibility | Updated the code to use the checkpoint data paths and work with the newer Python environment. | Compatibility commit |
| Reah Robinson | FRONT runs and results | Ran the FRONT defense several times and averaged the results across runs. | Averaged results added to README |


## Repeated-Trial Results

Ran `bash run_trials.sh 10` on the checkpoint dataset (10 sites × 100 visits, closed world).\
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

## Bounded Contribution Plan

We will extend the default FRONT baseline with a parameter sensitivity study. We will measure how changing FRONT’s dummy cell limits and padding timing affects k-fingerprinting accuracy and cell overhead.

| Experiment | Parameter change | Settings |
|---|---|---|
| Dummy cell limits | Vary `client_dummy_pkt_num` and `server_dummy_pkt_num` together. Keep the timing window at its default. | 50, 100 (baseline), and 200 per direction |
| Padding timing | Vary `min_wnd` and `max_wnd`. Keep both dummy cell limits at 100. | 1–4, 1–8 (baseline), and 1–12 seconds |

We will use the same trace IDs and classifier settings for every condition. We will run each FRONT setting with three fixed seeds and report the mean and range of attack accuracy and cell overhead. We will calculate cell overhead as total added dummy cells divided by total original cells.

### GLUE Extension

If time permits, we will use the fork’s GLUE implementation to combine pairs of visits into continuous traces. For a small, checkable sample, we will record the source visit IDs, verify that the merged output preserves their cells, and measure added cell overhead.
