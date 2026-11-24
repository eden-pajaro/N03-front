import csv
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
runs = pd.read_csv(out/'runs.csv')
assert len(runs) == 16
assert (runs.processed == 1010).all()
assert (runs.eligible_monitored == 1000).all()
assert runs.original_cells.nunique() == 1
assert np.allclose(runs.overhead_pct, 100*runs.dummy_cells/runs.original_cells)
expected_splits = None
for r in runs.itertuples():
    name = f'predictions-N{r.limit_per_direction}-W{r.max_wnd_s}-seed{r.seed}.csv'
    p = pd.read_csv(out/name)
    assert len(p) == 1000 and not p.duplicated(['fold','trace_id']).any()
    splits = p[['fold','trace_id']]
    if expected_splits is None:
        expected_splits = splits
    assert splits.equals(expected_splits)
    correct = (p[['neighbor_1','neighbor_2','neighbor_3']].values == p.true_site.values[:,None]).all(axis=1)
    assert np.isclose(100*correct.mean(),r.accuracy_pct)
metrics = ['accuracy_pct','overhead_pct','early_dummy_pct']
summary = runs.groupby(['limit_per_direction','max_wnd_s'],sort=True)[metrics].agg(['mean','min','max'])
summary.columns = ['_'.join(c) for c in summary.columns]
summary.to_csv(out/'summary.csv',float_format='%.4f')
print(summary.round(3).to_string())

def draw(ax, frame, x, y, label):
    m = frame[y+'_mean']
    ax.errorbar(x,m,yerr=[m-frame[y+'_min'],frame[y+'_max']-m],marker='o',capsize=4)
    ax.set_ylabel(label)
    ax.grid(alpha=.25)

budget=summary.reset_index().query('max_wnd_s == 14 or limit_per_direction == 0')
fig,ax=plt.subplots(figsize=(7,4))
draw(ax,budget,budget.overhead_pct_mean,'accuracy_pct','Closed-world kFP accuracy (%)')
for r in budget.itertuples():
    ax.annotate(f'{r.limit_per_direction:,}' if r.limit_per_direction else 'Undefended',
        (r.overhead_pct_mean,r.accuracy_pct_mean),xytext=(4,7),textcoords='offset points')
ax.set(xlabel='Measured cell overhead (%)',title='Experiment 1 | Symmetric padding limits')
ax.margins(x=.10,y=.18)
fig.tight_layout()
fig.savefig(out/'experiment-1-budget.png',dpi=180)
plt.close(fig)
timing=summary.reset_index().query('limit_per_direction == 1100').sort_values('max_wnd_s')
fig,axes=plt.subplots(3,1,figsize=(7,7),sharex=True)
for ax,y,label in zip(axes,metrics,['kFP accuracy (%)','Cell overhead (%)','Early dummy cells (%)']):
    draw(ax,timing,timing.max_wnd_s,y,label)
axes[0].set_title('Experiment 2 | Upper padding-window bound')
axes[-1].set(xlabel='Maximum window scale (s)',xticks=[8,14,24])
fig.tight_layout()
fig.savefig(out/'experiment-2-timing.png',dpi=180)
