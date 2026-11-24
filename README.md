# Experiments 1 and 2 (Eden Pajaro)

**Team N03: Website Fingerprinting**

I tested how FRONT’s dummy-cell budget and timing affect website fingerprinting on recorded Tor visits. Lower k-fingerprinting (kFP) accuracy means the classifier correctly identifies fewer visits. Cell overhead is the percentage of extra traffic added by FRONT.

## Intructions for Running Experiments 1 and 2 

From the repository root on the `eden` branch, create and activate a Python 3.11 virtual environment. I used Git Bash on Windows, so this is what I used.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r experiments/front-sweeps/requirements-used.txt
python experiments/front-sweeps/run_front_sweeps.py . sweep-results
python experiments/front-sweeps/summarize_front_sweeps.py sweep-results
```

The first script runs the undefended control and both FRONT sweeps. The second verifies the results and generates the summary CSV and plots. Both write to `sweep-results/`. Note that `sweep-results/` is not on this GitHub repo. I only saved it locally. The results are summarized in the CSVs in `experiments/front-sweeps`.

## Experimental Setup

- **Data:** The runner processes all 1,010 checkpoint traces in `data/checkpoint/cells/` for overhead measurement. The classifier evaluates the 1,000 monitored visits from 10 sites.
- **Control:** I ran kFP on the original traces without FRONT and confirmed that it matched our checkpoint baseline. Every condition uses the same ten stratified 90/10 train/test splits. kFP is trained separately for each condition and evaluated on its held-out traces.
- **Repetitions:** Each FRONT setting uses three padding seeds (`301`, `302`, and `303`). Both experiments share the 1,100-cell, 1–14 s reference runs.
- **Overhead:** To calculate, divide the total number of inserted cells by the total number of original cells and multiply by 100. Before feature extraction, normalize packet directions to `+1` and `-1` so kFP sees direction without the simulator’s dummy-cell markers.

## Experiment 1: Padding Budget

We varied the dummy-cell limit equally in both directions and held the timing scale bounds at 1 and 14 s. Each limit caps how many dummy cells FRONT can schedule per direction for a visit.

| Dummy-cell limit per direction | Mean kFP accuracy | Mean cell overhead |
|---:|---:|---:|
| Undefended control | 98.7% | 0.0% |
| 550 | 93.7% | 16.4% |
| 1,100 | 89.9% | 33.0% |
| 1,650 | 86.1% | 49.1% |

Higher limits lowered classifier accuracy and increased overhead. Raising the limit from 1,100 to 1,650 reduced accuracy by another **3.8 percentage points** and increased overhead by **16.1 percentage points**.

**Generated plot:** `sweep-results/experiment-1-budget.png`

## Experiment 2: Padding Timing

I held the limit at 1,100 cells per direction and the lower timing scale bound at 1 s then varied the upper bound. A larger upper bound lets FRONT spread the dummy cells over a longer period. The simulator discards cells scheduled after the original visit ends.

| Timing scale bounds | Mean kFP accuracy | Mean cell overhead | Dummy cells in the first quarter of the visit |
|---|---:|---:|---:|
| 1–8 s | 92.7% | 35.0% | 62.3% |
| 1–14 s | 89.9% | 33.0% | 44.7% |
| 1–24 s | 89.0% | 27.4% | 34.4% |

The 1–24 s setting saved **5.5 percentage points of overhead** relative to the 1–14 s reference, using the unrounded means. The accuracy ranges across the three runs overlapped. On these traces, the longer timing range used less extra traffic at similar observed accuracy.

**Generated plot:** `sweep-results/experiment-2-timing.png`

## Evidence

- [Summary CSV](experiments/front-sweeps/summary.csv) contains means and run ranges.
- [Run-level CSV](experiments/front-sweeps/runs.csv) contains seeds, accuracy, and cell counts.
