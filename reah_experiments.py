#!/usr/bin/env python3
"""
Reah's FRONT experiments for Team N03 (project plan experiments #3 and #4).

  #3 Client/server budget split: total 2,200 dummy cells, timing 1-14 s,
     client/server = 550/1650, 1100/1100, 1650/550.
  #4 Shift padding timing later: 1,100/1,100 cells, windows 1-14, 4-17, 8-21 s,
     start_padding_time = 0.

The 1100/1100 @ 1-14 s condition is shared by both experiments and run once
(3 padding runs), as the plan says to reuse it.

Run from the repo root:
    python3 reah_experiments.py            # 3 padding runs per condition
    python3 reah_experiments.py --runs 5   # more runs

Everything is written to results/reah/ (CSV, predictions, splits, plots, README).
Finished runs are skipped if you re-run the script, so it is safe to restart.
"""
import argparse, configparser, csv, glob, os, platform, shutil, subprocess, sys, time
import multiprocessing as mp
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "checkpoint", "cells")
FRONT_DIR = os.path.join(ROOT, "defenses", "front")
FRONT_CFG = os.path.join(FRONT_DIR, "config.ini")
RAW_DIR = os.path.join(ROOT, "defenses", "results")          # git-ignored
OUT = os.path.join(ROOT, "results", "reah")
KFP_DIR = os.path.join(ROOT, "attacks", "kfingerprinting")
sys.path.insert(0, KFP_DIR)

MON_SITES, MON_INST, UNMON = 10, 100, 10

# name, experiment tag, client cells, server cells, min_wnd, max_wnd
CONDITIONS = [
    ("split_c550_s1650",  "3",   550, 1650, 1, 14),
    ("base_c1100_s1100",  "3+4", 1100, 1100, 1, 14),
    ("split_c1650_s550",  "3",  1650,  550, 1, 14),
    ("shift_w4_17",       "4",  1100, 1100, 4, 17),
    ("shift_w8_21",       "4",  1100, 1100, 8, 21),
]
EXP3 = ["split_c550_s1650", "base_c1100_s1100", "split_c1650_s550"]
EXP4 = ["base_c1100_s1100", "shift_w4_17", "shift_w8_21"]

RUN_FIELDS = ["condition", "experiment", "run", "client_cfg", "server_cfg", "min_wnd", "max_wnd",
              "start_padding_time", "processed", "scored", "accuracy", "original_cells",
              "client_dummies", "server_dummies", "inserted_cells", "overhead_pct", "median_dummy_time_s",
              "splits_match", "raw_dir", "timestamp"]

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated categorical slots 1-3


# ---------------------------------------------------------------- helpers
def trace_ids():
    ids = [f"{s}-{i}" for s in range(MON_SITES) for i in range(MON_INST)]
    ids += [str(u) for u in range(UNMON)]
    return ids


def read_trace(path):
    # plain split is much faster than np.loadtxt on some Macs
    with open(path) as f:
        vals = f.read().split()
    return np.array(vals, dtype=float).reshape(-1, 2)


def read_all(folder, ids, label):
    out = {}
    for k, t in enumerate(ids, 1):
        out[t] = read_trace(os.path.join(folder, t + ".cell"))
        if k % 250 == 0 or k == len(ids):
            print(f"  {label}: read {k}/{len(ids)} traces", flush=True)
    return out


def ensure_configs():
    cp = configparser.RawConfigParser()
    cp.optionxform = str
    cp.read(FRONT_CFG)
    added = []
    for name, _, c, s, lo, hi in CONDITIONS:
        sec = "reah_" + name
        if cp.has_section(sec):
            continue
        added.append(sec)
        with open(FRONT_CFG, "a") as f:
            f.write(f"\n[{sec}]\nMON_SITE_NUM = {MON_SITES}\nMON_INST_NUM = {MON_INST}\n"
                    f"UNMON_SITE_NUM = {UNMON}\nclient_dummy_pkt_num = {c}\n"
                    f"server_dummy_pkt_num = {s}\nstart_padding_time = 0\n"
                    f"max_wnd = {hi}\nmin_wnd = {lo}\n")
    if added:
        print("Added config sections:", ", ".join(added))


def run_front(name):
    """Run the team's FRONT simulator and archive its output under a unique name."""
    os.makedirs(RAW_DIR, exist_ok=True)
    before = set(glob.glob(os.path.join(RAW_DIR, "ranpad2_*")))
    log = subprocess.run([sys.executable, "main.py", DATA + "/", "-format", ".cell", "-c", "reah_" + name],
                         cwd=FRONT_DIR, capture_output=True, text=True)
    new = sorted(set(glob.glob(os.path.join(RAW_DIR, "ranpad2_*"))) - before, key=os.path.getmtime)
    if log.returncode != 0 or not new:
        print(log.stdout[-2000:], log.stderr[-3000:])
        sys.exit(f"FRONT failed for {name}. See the error above.")
    return new[-1]


def _features(args):
    tid, lines = args
    import extract  # the team's kFP feature code, unchanged
    if len(lines) < 50:          # same rule as extract.py
        return tid, None
    try:
        return tid, extract.TOTAL_FEATURES(lines)
    except Exception:
        return tid, None


def to_lines(arr):
    """Normalise directions to +1/-1 (as utils/norm.py does) and format as kFP input lines."""
    d = np.sign(arr[:, 1]).astype(int)
    return [f"{t}\t{x}\n" for t, x in zip(arr[:, 0], d)]


def features_for(traces):
    jobs = [(tid, to_lines(arr)) for tid, arr in traces.items()]
    ctx = mp.get_context("fork")
    with ctx.Pool() as pool:
        res = pool.map(_features, jobs, chunksize=8)
    return dict(res)


def kfp_eval(feats, folds=10, trees=1000):
    """Closed-world kFP, same as attacks/kfingerprinting/new_main.py:
    random forest leaves + 3 nearest neighbours, a guess counts only if all 3 agree;
    10 stratified 90/10 splits with random_state=0. Returns accuracy, predictions, split IDs."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedShuffleSplit
    ids = [t for t in trace_ids() if "-" in t and feats.get(t) is not None]
    X = np.array([feats[t] for t in ids], dtype=float)
    y = np.array([int(t.split("-")[0]) for t in ids])
    sss = StratifiedShuffleSplit(n_splits=folds, test_size=0.1, random_state=0)
    preds, splits, correct = [], [], 0
    for fold, (tr, te) in enumerate(sss.split(X, y), 1):
        model = RandomForestClassifier(n_jobs=-1, n_estimators=trees, oob_score=True)
        model.fit(X[tr], y[tr])
        trl, tel = model.apply(X[tr]), model.apply(X[te])
        for j, row in enumerate(tel):
            dists = np.sum(trl != row, axis=1)
            nb = y[tr][np.argsort(dists)[:3]]
            guess = int(nb[0]) if len(set(nb)) == 1 else -1
            ok = guess == y[te][j]
            correct += ok
            preds.append({"fold": fold, "trace_id": ids[te[j]], "true_site": int(y[te][j]),
                          "predicted_site": guess, "neighbours": " ".join(map(str, nb)), "correct": int(ok)})
        splits.append([ids[i] for i in te])
    return correct / len(preds), preds, splits, len(ids)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_runs():
    p = os.path.join(OUT, "runs.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def save_runs(rows):
    write_csv(os.path.join(OUT, "runs.csv"), rows, RUN_FIELDS)


# ---------------------------------------------------------------- main work
def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--test", action="store_true", help="quick smoke test (fewer trees/folds)")
    a = ap.parse_args()
    folds, trees = (2, 100) if a.test else (10, 1000)

    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "timing"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)
    ensure_configs()

    ids = trace_ids()
    missing = [t for t in ids if not os.path.exists(os.path.join(DATA, t + ".cell"))]
    print(f"Input traces found: {len(ids) - len(missing)} of {len(ids)}")
    if missing:
        sys.exit(f"Missing input traces, e.g. {missing[:5]}")
    original = read_all(DATA, ids, "original")
    orig_cells = int(sum(len(v) for v in original.values()))
    shutil.copy(os.path.join(DATA, "0-0.cell"), os.path.join(OUT, "samples", "0-0_original.cell"))

    rows = load_runs()
    done = {(r["condition"], r["run"]) for r in rows}
    split_path = os.path.join(OUT, "splits.csv")

    # Undefended reference (also fixes the split membership every condition must match)
    if ("undefended", "1") not in done:
        print("Undefended reference: extracting features and scoring kFP ...")
        acc, preds, splits, scored = kfp_eval(features_for(original), folds, trees)
        write_csv(split_path, [{"fold": i + 1, "trace_id": t} for i, s in enumerate(splits) for t in s],
                  ["fold", "trace_id"])
        write_csv(os.path.join(OUT, "predictions", "undefended_run1.csv"), preds, list(preds[0]))
        rows.append(dict(condition="undefended", experiment="ref", run=1, client_cfg=0, server_cfg=0,
                         min_wnd="", max_wnd="", start_padding_time="", processed=len(ids), scored=len(preds),
                         accuracy=f"{acc:.4f}", original_cells=orig_cells, client_dummies=0, server_dummies=0,
                         inserted_cells=0, overhead_pct="0.00", median_dummy_time_s="", splits_match="reference", raw_dir="",
                         timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
        save_runs(rows)
        print(f"  undefended accuracy = {acc:.4f}")
    with open(split_path) as f:
        ref_split = {}
        for r in csv.DictReader(f):
            ref_split.setdefault(int(r["fold"]), []).append(r["trace_id"])

    for name, exp, c, s, lo, hi in CONDITIONS:
        for run in range(1, a.runs + 1):
            if (name, str(run)) in done:
                print(f"{name} run {run}: already done, skipping")
                continue
            t0 = time.time()
            raw = run_front(name)
            dest = os.path.join(RAW_DIR, "reah", f"{name}_run{run}")
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(raw, dest)

            defended, cdum, sdum, ocells = {}, 0, 0, 0
            ctimes, stimes = [], []
            print(f"{name} run {run}: FRONT done, measuring and scoring ...", flush=True)
            for t in ids:
                arr = read_trace(os.path.join(dest, t + ".cell"))
                defended[t] = arr
                d = arr[:, 1]
                cdum += int(np.sum(d == 888)); sdum += int(np.sum(d == -888))
                ocells += int(np.sum(np.abs(d) == 1))
                ctimes.append(arr[d == 888, 0]); stimes.append(arr[d == -888, 0])
            if ocells != orig_cells:
                sys.exit(f"Sanity check failed: original cells changed ({ocells} vs {orig_cells})")
            ct, st = np.concatenate(ctimes), np.concatenate(stimes)
            med = float(np.median(np.concatenate([ct, st])))
            if run == 1:
                edges = np.arange(0, 121, 1)
                hc, _ = np.histogram(ct, bins=edges); hs, _ = np.histogram(st, bins=edges)
                write_csv(os.path.join(OUT, "timing", f"{name}_run1_dummy_hist.csv"),
                          [{"second": int(e), "client_dummies": int(a), "server_dummies": int(b)}
                           for e, a, b in zip(edges[:-1], hc, hs)],
                          ["second", "client_dummies", "server_dummies"])
                shutil.copy(os.path.join(dest, "0-0.cell"), os.path.join(OUT, "samples", f"0-0_{name}_run1.cell"))

            acc, preds, splits, _ = kfp_eval(features_for(defended), folds, trees)
            match = all(sorted(splits[i]) == sorted(ref_split[i + 1]) for i in range(len(splits)))
            write_csv(os.path.join(OUT, "predictions", f"{name}_run{run}.csv"), preds, list(preds[0]))
            ins = cdum + sdum
            rows.append(dict(condition=name, experiment=exp, run=run, client_cfg=c, server_cfg=s,
                             min_wnd=lo, max_wnd=hi, start_padding_time=0, processed=len(defended),
                             scored=len(preds), accuracy=f"{acc:.4f}", original_cells=ocells,
                             client_dummies=cdum, server_dummies=sdum, inserted_cells=ins,
                             overhead_pct=f"{100 * ins / ocells:.2f}", median_dummy_time_s=f"{med:.2f}", splits_match=match,
                             raw_dir=os.path.relpath(dest, ROOT), timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
            save_runs(rows)
            print(f"{name} run {run}: accuracy={acc:.4f} overhead={100 * ins / ocells:.2f}% "
                  f"client dummies={cdum} server dummies={sdum} splits match={match} ({time.time() - t0:.0f}s)")

    summarise(rows, a.test)


# ---------------------------------------------------------------- summary, plots, README
def stats(rows, name, key):
    v = [float(r[key]) for r in rows if r["condition"] == name]
    return np.mean(v), min(v), max(v), len(v)


def summarise(rows, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e5e5e5"})
    cond = {c[0]: c for c in CONDITIONS}
    tag = {"split_c550_s1650": "550 / 1,650", "base_c1100_s1100": "1,100 / 1,100",
           "split_c1650_s550": "1,650 / 550", "shift_w4_17": "4–17 s", "shift_w8_21": "8–21 s"}

    def per_run(name, key):
        return [float(r[key]) for r in rows if r["condition"] == name]

    # ---- Experiment 3 plot: accuracy and overhead by client share (two panels, one axis each)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    share = [100 * cond[n][2] / (cond[n][2] + cond[n][3]) for n in EXP3]
    for ax, key, lab in [(axs[0], "accuracy", "kFP accuracy (%)"), (axs[1], "overhead_pct", "Cell overhead (%)")]:
        scale = 100 if key == "accuracy" else 1
        means = [np.mean(per_run(n, key)) * scale for n in EXP3]
        for x, n in zip(share, EXP3):
            ax.scatter([x] * len(per_run(n, key)), [v * scale for v in per_run(n, key)], s=30,
                       color=COLORS[0], alpha=0.45, edgecolor="white", linewidth=1, zorder=3)
        ax.plot(share, means, color=COLORS[0], linewidth=2, marker="o", markersize=8,
                markeredgecolor="white", zorder=4)
        for x, m in zip(share, means):
            ax.annotate(f"{m:.1f}", (x, m), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
        ax.set_xticks(share, [f"{s:.0f}%\n({tag[n]})" for s, n in zip(share, EXP3)])
        ax.set_xlabel("Client share of the 2,200-cell budget (client / server)")
        ax.set_title(lab, loc="left", fontsize=10)
        ax.margins(y=0.25)
    fig.suptitle("Exp. 3 · Client/server budget split (1–14 s; dots = runs, line = mean)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp3_split.png"), dpi=160)
    plt.close(fig)

    # ---- Experiment 4 plot: dummy timing distribution overlay + accuracy/overhead
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    for i, n in enumerate(EXP4):
        p = os.path.join(OUT, "timing", f"{n}_run1_dummy_hist.csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            hr = list(csv.DictReader(f))[:60]
        sec = [int(r["second"]) for r in hr]
        h = np.array([int(r["client_dummies"]) + int(r["server_dummies"]) for r in hr])
        axs[0].step(sec, h / 1010, where="post", color=COLORS[i], linewidth=2,
                    label=f"{cond[n][4]}–{cond[n][5]} s")
    axs[0].set_xlabel("Time in trace (s)")
    axs[0].set_title("Dummy cells per trace per second (run 1)", loc="left", fontsize=10)
    axs[0].legend(frameon=False)
    x = np.arange(len(EXP4))
    for ax, key, lab in [(axs[1], "accuracy", "kFP accuracy (%)"), (axs[2], "overhead_pct", "Cell overhead (%)")]:
        scale = 100 if key == "accuracy" else 1
        for i, n in enumerate(EXP4):
            v = [q * scale for q in per_run(n, key)]
            ax.scatter([i] * len(v), v, s=30, color=COLORS[i], alpha=0.45, edgecolor="white", zorder=3)
            m = np.mean(v)
            ax.scatter([i], [m], s=80, color=COLORS[i], edgecolor="white", linewidth=2, zorder=4)
            ax.annotate(f"{m:.1f}", (i, m), textcoords="offset points", xytext=(12, -3), fontsize=9)
        ax.set_xticks(x, [f"{cond[n][4]}–{cond[n][5]} s" for n in EXP4])
        ax.set_xlim(-0.5, len(EXP4) - 0.4)
        ax.set_title(lab, loc="left", fontsize=10)
        ax.margins(y=0.25)
    fig.suptitle("Exp. 4 · Shifting the padding window later (1,100 / 1,100 cells)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp4_shift.png"), dpi=160)
    plt.close(fig)

    write_readme(rows, tag, test)
    print("\nDone. Results are in results/reah/ (README.md, runs.csv, plots).")


def fmt_range(rows, n, key, scale=1.0, dec=1):
    v = [float(r[key]) * scale for r in rows if r["condition"] == n]
    return f"{np.mean(v):.{dec}f} ({min(v):.{dec}f}–{max(v):.{dec}f})"


def write_readme(rows, tag, test):
    import sklearn, pandas
    cond = {c[0]: c for c in CONDITIONS}
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    n_traces = 1010
    und = [r for r in rows if r["condition"] == "undefended"]
    nruns = max(int(r["run"]) for r in rows if r["condition"] != "undefended")
    allmatch = all(r["splits_match"] in ("True", "reference", True) for r in rows)

    def dummies(n, key):
        v = [int(r[key]) / n_traces for r in rows if r["condition"] == n]
        return f"{np.mean(v):.0f}"

    def kept(n):
        c, s = cond[n][2], cond[n][3]
        v = [(int(r["client_dummies"]) + int(r["server_dummies"])) / (n_traces * (c + s)) * 100
             for r in rows if r["condition"] == n]
        return f"{np.mean(v):.1f}%"

    def median_time(n):
        return fmt_range(rows, n, "median_dummy_time_s") + " s"

    L = []
    L.append("# Reah Robinson: FRONT experiments #3 and #4 (Team N03)\n")
    if test:
        L.append("> **SMOKE TEST ONLY.** These numbers came from `--test` (2 folds, 100 trees). Re-run without `--test`.\n")
    L.append("Research question: on our fixed set of Tor traces, how do FRONT's padding direction "
             "(client/server split) and padding timing affect kFP accuracy and cell overhead?\n")
    L.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M')} by `reah_experiments.py`")
    L.append(f"- Code commit when run: `{sha}` (checkpoint evidence SHA: `0da3610602bb0c3af83ff2a2df9dff822b68e205`)")
    L.append(f"- Software: Python {platform.python_version()}, numpy {np.__version__}, pandas {pandas.__version__}, "
             f"scikit-learn {sklearn.__version__}, {platform.system()} {platform.machine()}")
    L.append(f"- Data: `data/checkpoint/cells/`, {n_traces} traces processed (10 sites × 100 visits + 10 unmonitored); "
             f"closed-world kFP scores the {und[0]['scored'] if und else 1000} monitored traces.")
    L.append(f"- Each FRONT setting was padded **{nruns} time(s)** (new random padding each run). Tables show the mean, "
             "with the min–max range across runs in brackets.\n")

    L.append("## Reference (no defense)\n")
    if und:
        L.append(f"kFP accuracy without FRONT: **{float(und[0]['accuracy']) * 100:.1f}%** "
                 f"({und[0]['scored']} traces scored), 0% overhead.\n")

    L.append("## Experiment 3: client/server budget split\n")
    L.append("Total configured budget held at 2,200 dummy cells; timing 1–14 s; `start_padding_time = 0`.\n")
    L.append("| Client / server (configured) | Client share | Actual client dummies per trace | "
             "Actual server dummies per trace | Inserted ÷ configured max | Cell overhead % | kFP accuracy % |")
    L.append("|---|---|---|---|---|---|---|")
    for n in EXP3:
        c, s = cond[n][2], cond[n][3]
        L.append(f"| {tag[n]} | {100 * c / (c + s):.0f}% | {dummies(n, 'client_dummies')} | {dummies(n, 'server_dummies')} | "
                 f"{kept(n)} | {fmt_range(rows, n, 'overhead_pct')} | {fmt_range(rows, n, 'accuracy', 100)} |")
    L.append("\n![Experiment 3](exp3_split.png)\n")
    L.append("The configured values are maximums: for each trace FRONT draws its dummy count uniformly between 1 and "
             "the limit (about half on average), then drops any dummy cell scheduled after the original trace ends. "
             "So the same configured total can produce different inserted counts, which is why actual counts are reported.\n")

    L.append("## Experiment 4: shifting the padding window later\n")
    L.append("Limits held at 1,100 / 1,100 cells; the window range stays 13 s wide while both bounds move later; "
             "`start_padding_time = 0`.\n")
    L.append("| Window (min_wnd–max_wnd) | Median dummy-cell time | Inserted ÷ configured max | "
             "Cell overhead % | kFP accuracy % |")
    L.append("|---|---|---|---|---|")
    for n in EXP4:
        L.append(f"| {cond[n][4]}–{cond[n][5]} s | {median_time(n)} | {kept(n)} | {fmt_range(rows, n, 'overhead_pct')} | "
                 f"{fmt_range(rows, n, 'accuracy', 100)} |")
    L.append("\n![Experiment 4](exp4_shift.png)\n")
    L.append("Compare with Eden's #2, where only the upper bound moves.\n")

    L.append("## Sanity checks\n")
    L.append(f"- All {n_traces} input traces were found, and FRONT wrote {n_traces} output traces in every run.")
    L.append("- Original (non-dummy) cell count was identical before and after FRONT in every run.")
    L.append(f"- Every condition was scored on the same held-out trace IDs as the undefended reference: "
             f"**{'yes' if allmatch else 'NO, check runs.csv'}** (`splits_match` column).\n")

    L.append("## Method\n")
    L.append("1. Each condition is its own section in `defenses/front/config.ini` (`reah_*`), copied from "
             "`[checkpoint]` with only the listed values changed, and run with the team's unchanged FRONT simulator.")
    L.append("2. Overhead is measured on the raw FRONT output before normalisation: "
             "100 × (+888 and −888 cells) ÷ original cells.")
    L.append("3. Directions are normalised to ±1 (as `utils/norm.py` does) so kFP can't see the dummy marker.")
    L.append("4. kFP is the team's feature code (`extract.py`) with `new_main.py`'s closed-world evaluation: "
             "random forest (1,000 trees) + 3 nearest neighbours, all 3 must agree; 10 stratified 90/10 splits, "
             "`random_state=0`. The forest is refit for every condition and run.\n")

    L.append("## Files\n")
    L.append("| File | What it is |")
    L.append("|---|---|")
    L.append("| `runs.csv` | One row per condition and padding run: parameters, counts, accuracy, overhead |")
    L.append("| `predictions/*.csv` | Held-out predictions with trace IDs for every run |")
    L.append("| `splits.csv` | Held-out trace IDs for each of the 10 folds |")
    L.append("| `timing/*.csv` | Dummy cells per second of trace time (run 1), used for the timing plot |")
    L.append("| `samples/` | Trace `0-0` before FRONT and after each condition (run 1); dummy cells are ±888 |")
    L.append("| `exp3_split.png`, `exp4_shift.png` | Plots for the slides |")
    L.append("| `../../reah_experiments.py` | The script that produced all of this |")
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
