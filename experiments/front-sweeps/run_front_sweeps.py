import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
CONDITIONS = [(0, 0)] + [(n, w) for n, w in [(550,14),(1100,14),(1650,14),(1100,8),(1100,24)]]
CASES = [(n,w,seed) for n,w in CONDITIONS for seed in ([0] if not n else [301,302,303])]

def run(case):
    n,w,seed = case
    front_dir = ROOT / 'defenses/front'
    kfp_dir = ROOT / 'attacks/kfingerprinting'
    sys.path[:0] = [str(front_dir), str(kfp_dir)]
    front = runpy.run_path(str(front_dir / 'main.py'))['RP']
    front.__globals__.update(client_dummy_pkt_num=n, server_dummy_pkt_num=n,
        client_min_dummy_pkt_num=1, server_min_dummy_pkt_num=1, min_wnd=1,
        max_wnd=w, start_padding_time=0, logger=logging.getLogger('front'))
    features = runpy.run_path(str(kfp_dir / 'extract.py'))['TOTAL_FEATURES']
    kfp = runpy.run_path(str(kfp_dir / 'new_main.py'))
    kfp['closed_world_acc'].__globals__['MON_SITE_NUM'] = 10
    source = ROOT / 'data/checkpoint/cells'
    trace_ids = [f'{site}-{visit}' for site in range(10) for visit in range(100)]
    file_ids = trace_ids + [str(i) for i in range(10)]
    original = added = early = 0
    vectors = []
    np.random.seed(seed)
    for i, name in enumerate(file_ids):
        trace = np.loadtxt(source / f'{name}.cell')
        padded = front(trace) if n else trace
        flags = np.abs(padded[:,1]) == 888
        assert np.array_equal(padded[~flags], trace), name
        original += len(trace)
        added += int(flags.sum())
        early += int((flags & (padded[:,0] <= trace[0,0] + (trace[-1,0]-trace[0,0])/4)).sum())
        if name == '0-0' and seed in (0,301):
            first_trace = OUT / f'trace-0-0-N{n}-W{w}.cell'
            np.savetxt(first_trace,padded,fmt=['%.10g','%d'],delimiter='\t')
        if i < len(trace_ids):
            assert len(trace) >= 50, name
            rows = [f'{t:.4f}\t{int(np.sign(d))}' for t,d in padded] if n else (source / f'{name}.cell').read_text().splitlines()
            vectors.append(features(rows))
    X = np.asarray(vectors)
    y = np.repeat(np.arange(10),100)
    assert np.isfinite(X).all()
    splits = StratifiedShuffleSplit(n_splits=10,test_size=.1,random_state=0)
    accuracies = []
    detail = OUT / f'predictions-N{n}-W{w}-seed{seed}.csv'
    np.random.seed(1123)
    with detail.open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold','trace_id','true_site','neighbor_1','neighbor_2','neighbor_3'])
        for fold,(train,test) in enumerate(splits.split(X,y),1):
            run_forest = kfp['kfingerprinting']
            classifier = run_forest.__globals__['RandomForestClassifier']
            run_forest.__globals__['RandomForestClassifier'] = lambda **kw: classifier(**{**kw,'n_jobs':1})
            try:
                a,b = run_forest(X[train],X[test],y[train],y[test])
            finally:
                run_forest.__globals__['RandomForestClassifier'] = classifier
            neighbors = [kfp['get_neighbor']((a, row, y[train], 3)) for row in b]
            accuracies.append(kfp['closed_world_acc'](neighbors,y[test])[0])
            writer.writerows((fold,trace_ids[idx],y[idx],*near) for idx,near in zip(test,neighbors))
    return (n,w,seed,len(file_ids),len(trace_ids),original,added,early,100*np.mean(accuracies),100*added/original,100*early/added if added else '')

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool, (OUT/'runs.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['limit_per_direction','max_wnd_s','seed','processed','eligible_monitored','original_cells','dummy_cells','early_dummy_cells','accuracy_pct','overhead_pct','early_dummy_pct'])
        for row in pool.map(run, CASES):
            writer.writerow(row)
            f.flush()
            print(row,flush=True)
