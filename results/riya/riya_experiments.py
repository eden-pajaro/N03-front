"""
Riya's FRONT experiments for Team N03 (project plan experiments #5 and #6).

  #5 Budget x timing interaction: limit 550 or 1,650 cells per direction x
     window 1-8 s or 1-24 s (min_wnd = 1, start_padding_time = 0), 3 seeds each.
  #6 Timing sensitivity on short vs long visits: reuses #2's 1,100/1,100 runs at
     1-8 s and 1-24 s (seeds 301-303), regenerated exactly with Eden's procedure.

This is Eden's experiments/front-sweeps/run_front_sweeps.py with the same seeding,
same FRONT code (defenses/front/main.py, RP), same kFP features and evaluation
(attacks/kfingerprinting, 10 stratified 90/10 splits, random_state=0, np seed 1123,
n_jobs=1). The only addition is logging, per trace, how many dummy cells FRONT
scheduled and how many it kept (it drops any scheduled after the trace ends).
Wrapping getTimestamps does not consume random numbers, so results match Eden's.

Usage: python3 riya_experiments.py <repo root> <out dir> [N:W ...]
"""
import csv, logging, runpy, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
SEEDS = [301, 302, 303]
DEFAULT = [(550, 8), (550, 24), (1650, 8), (1650, 24), (1100, 8), (1100, 24)]


def run(case):
    n, w, seed = case
    front_dir = ROOT / 'defenses/front'
    kfp_dir = ROOT / 'attacks/kfingerprinting'
    sys.path[:0] = [str(front_dir), str(kfp_dir)]
    front = runpy.run_path(str(front_dir / 'main.py'))['RP']
    front.__globals__.update(client_dummy_pkt_num=n, server_dummy_pkt_num=n,
        client_min_dummy_pkt_num=1, server_min_dummy_pkt_num=1, min_wnd=1,
        max_wnd=w, start_padding_time=0, logger=logging.getLogger('front'))
    scheduled = []
    orig_ts = front.__globals__['getTimestamps']
    def logged_ts(wnd, num):
        out = orig_ts(wnd, num)
        scheduled.append(len(out))
        return out
    front.__globals__['getTimestamps'] = logged_ts

    features = runpy.run_path(str(kfp_dir / 'extract.py'))['TOTAL_FEATURES']
    kfp = runpy.run_path(str(kfp_dir / 'new_main.py'))
    kfp['closed_world_acc'].__globals__['MON_SITE_NUM'] = 10
    source = ROOT / 'data/checkpoint/cells'
    trace_ids = [f'{s}-{v}' for s in range(10) for v in range(100)]
    file_ids = trace_ids + [str(i) for i in range(10)]
    stats, vectors = [], []
    np.random.seed(seed)
    for i, name in enumerate(file_ids):
        trace = np.loadtxt(source / f'{name}.cell')
        scheduled.clear()
        padded = front(trace)
        flags = np.abs(padded[:, 1]) == 888
        assert np.array_equal(padded[~flags], trace), name
        assert len(scheduled) == 2, name
        stats.append(dict(trace_id=name, duration_s=trace[-1, 0] - trace[0, 0],
                          real_out=int((trace[:, 1] > 0).sum()), real_in=int((trace[:, 1] < 0).sum()),
                          client_scheduled=scheduled[0], client_sent=int((padded[:, 1] == 888).sum()),
                          server_scheduled=scheduled[1], server_sent=int((padded[:, 1] == -888).sum())))
        if i < len(trace_ids):
            assert len(trace) >= 50, name
            vectors.append(features([f'{t:.4f}\t{int(np.sign(d))}' for t, d in padded]))
    X = np.asarray(vectors)
    y = np.repeat(np.arange(10), 100)
    assert np.isfinite(X).all()
    splits = StratifiedShuffleSplit(n_splits=10, test_size=.1, random_state=0)
    accs, preds = [], []
    np.random.seed(1123)
    for fold, (train, test) in enumerate(splits.split(X, y), 1):
        run_forest = kfp['kfingerprinting']
        classifier = run_forest.__globals__['RandomForestClassifier']
        run_forest.__globals__['RandomForestClassifier'] = lambda **kw: classifier(**{**kw, 'n_jobs': 1})
        try:
            a, b = run_forest(X[train], X[test], y[train], y[test])
        finally:
            run_forest.__globals__['RandomForestClassifier'] = classifier
        neighbors = [kfp['get_neighbor']((a, row, y[train], 3)) for row in b]
        accs.append(kfp['closed_world_acc'](neighbors, y[test])[0])
        for idx, near in zip(test, neighbors):
            preds.append([fold, trace_ids[idx], y[idx], *near, int(all(x == y[idx] for x in near))])

    tag = f'N{n}-W{w}-seed{seed}'
    with (OUT / f'predictions-{tag}.csv').open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['fold', 'trace_id', 'true_site', 'neighbor_1', 'neighbor_2', 'neighbor_3', 'correct'])
        wr.writerows(preds)
    with (OUT / f'trace-stats-{tag}.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(stats[0]))
        wr.writeheader()
        wr.writerows(stats)
    orig = sum(s['real_out'] + s['real_in'] for s in stats)
    cs = sum(s['client_sent'] for s in stats); ss = sum(s['server_sent'] for s in stats)
    sched = sum(s['client_scheduled'] + s['server_scheduled'] for s in stats)
    return dict(limit_per_direction=n, max_wnd_s=w, seed=seed, processed=len(file_ids), scored=len(preds),
                accuracy_pct=100 * np.mean(accs), original_cells=orig, client_dummies=cs, server_dummies=ss,
                overhead_pct=100 * (cs + ss) / orig, client_overhead_pct=100 * cs / sum(s['real_out'] for s in stats),
                server_overhead_pct=100 * ss / sum(s['real_in'] for s in stats),
                scheduled_dummies=sched, kept_pct=100 * (cs + ss) / sched)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    conds = [tuple(map(int, a.split(':'))) for a in sys.argv[3:]] or DEFAULT
    cases = [(n, w, s) for n, w in conds for s in SEEDS
             if not (OUT / f'predictions-N{n}-W{w}-seed{s}.csv').exists()]
    runs_path = OUT / 'runs.csv'
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(run, cases):
            new = not runs_path.exists()
            with runs_path.open('a', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    wr.writeheader()
                wr.writerow(row)
            print(row, flush=True)
