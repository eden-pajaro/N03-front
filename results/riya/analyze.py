"""Analysis for experiments #5 and #6. Usage: python3 analyze.py <out dir>"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])
SEEDS = [301, 302, 303]
C = {8: '#2a78d6', 24: '#eb6834'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888', 'axes.grid': True, 'grid.color': '#e5e5e5'})
runs = pd.read_csv(OUT / 'runs.csv').sort_values(['limit_per_direction', 'max_wnd_s', 'seed'])
assert len(runs) == 18 and (runs.scored == 1000).all() and runs.original_cells.nunique() == 1
res = {}

# ------------------------------------------------------------------ #5
g = runs[runs.limit_per_direction.isin([550, 1650])]
cols = ['accuracy_pct', 'overhead_pct', 'client_overhead_pct', 'server_overhead_pct', 'kept_pct']
s5 = g.groupby(['limit_per_direction', 'max_wnd_s'])[cols].agg(['mean', 'min', 'max'])
s5.columns = ['_'.join(c) for c in s5.columns]
s5.to_csv(OUT / 'exp5_summary.csv', float_format='%.3f')

def per_seed(n, w, key='accuracy_pct'):
    return g[(g.limit_per_direction == n) & (g.max_wnd_s == w)].sort_values('seed')[key].values

e5 = {}
for w in (8, 24):
    d_acc = per_seed(1650, w) - per_seed(550, w)
    d_ovh = per_seed(1650, w, 'overhead_pct') - per_seed(550, w, 'overhead_pct')
    e5[w] = dict(delta_acc_mean=d_acc.mean(), delta_acc_seeds=d_acc.tolist(),
                 delta_ovh_mean=d_ovh.mean(),
                 acc_550=per_seed(550, w).mean(), acc_1650=per_seed(1650, w).mean(),
                 ovh_550=per_seed(550, w, 'overhead_pct').mean(), ovh_1650=per_seed(1650, w, 'overhead_pct').mean(),
                 pp_per_10pp_ovh=10 * d_acc.mean() / d_ovh.mean())
inter = np.array(e5[24]['delta_acc_seeds']) - np.array(e5[8]['delta_acc_seeds'])
# all 3x3x3x3 seed combinations, since seeds are not truly paired across conditions
combos = [(a2 - a1) - (b2 - b1) for a1 in per_seed(550, 24) for a2 in per_seed(1650, 24)
          for b1 in per_seed(550, 8) for b2 in per_seed(1650, 8)]
res['exp5'] = dict(by_window=e5, interaction_mean=float(np.mean(combos)),
                   interaction_range=[float(min(combos)), float(max(combos))],
                   interaction_share_negative=float(np.mean(np.array(combos) < 0)))

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for w in (8, 24):
    xs = [550, 1650]
    m = [e5[w]['acc_550'], e5[w]['acc_1650']]
    for x in xs:
        v = per_seed(x, w)
        ax.scatter([x] * 3, v, s=28, color=C[w], alpha=.4, edgecolor='white', zorder=3)
    ax.plot(xs, m, color=C[w], lw=2, marker='o', ms=8, mec='white', mew=1.5, zorder=4)
    for x, mm, o in zip(xs, m, [e5[w]['ovh_550'], e5[w]['ovh_1650']]):
        ax.annotate(f'{mm:.1f}%\n{o:.0f}% overhead', (x, mm), textcoords='offset points',
                    xytext=(-12 if x == 550 else 12, 0), ha='right' if x == 550 else 'left', va='center', fontsize=8.5, color='#333')
ax.set_xticks([550, 1650], ['550', '1,650'])
ax.set_xlim(150, 2050)
ax.set_xlabel('Dummy-cell limit per direction')
ax.set_ylabel('Closed-world kFP accuracy (%)')
ax.legend(handles=[plt.Line2D([], [], color=C[w], lw=2, marker='o', label=f'Window 1–{w} s  (raising budget: {e5[w]["delta_acc_mean"]:+.1f} pp)') for w in (8, 24)],
          frameon=False, loc='lower left')
ax.set_title('Exp. 5 · Budget × timing (dots = 3 seeds, line = mean)', loc='left', fontsize=10)
ax.margins(y=.2)
fig.tight_layout(); fig.savefig(OUT / 'exp5_budget_timing.png', dpi=170); plt.close(fig)

# ------------------------------------------------------------------ #6
# Groups are defined from original durations only, before any predictions are read.
ts = pd.read_csv(OUT / 'trace-stats-N1100-W8-seed301.csv')
mon = ts[ts.trace_id.str.contains('-')].copy()
mon['site'] = mon.trace_id.str.split('-').str[0].astype(int)
q1, q3 = mon.duration_s.quantile([.25, .75])
mon['global_group'] = np.where(mon.duration_s <= q1, 'short', np.where(mon.duration_s >= q3, 'long', 'middle'))
r = mon.groupby('site').duration_s.rank(method='first')
mon['site_group'] = np.where(r <= 25, 'short', np.where(r > 75, 'long', 'middle'))
mon[['trace_id', 'site', 'duration_s', 'global_group', 'site_group']].to_csv(OUT / 'exp6_groups.csv', index=False)
res['exp6_cutoffs_s'] = dict(q1=float(q1), q3=float(q3), median=float(mon.duration_s.median()))
comp = pd.crosstab(mon.site, mon.global_group)[['short', 'long']]
comp.to_csv(OUT / 'exp6_site_composition.csv')

rows, flips = [], []
for gcol in ('global_group', 'site_group'):
    for grp in ('short', 'long'):
        ids = set(mon.loc[mon[gcol] == grp, 'trace_id'])
        for seed in SEEDS:
            p = {}; st = {}
            for w in (8, 24):
                pr = pd.read_csv(OUT / f'predictions-N1100-W{w}-seed{seed}.csv')
                p[w] = pr[pr.trace_id.isin(ids)].set_index(['fold', 'trace_id']).correct.sort_index()
                t = pd.read_csv(OUT / f'trace-stats-N1100-W{w}-seed{seed}.csv').set_index('trace_id').loc[sorted(ids)]
                st[w] = dict(overhead_pct=100 * (t.client_sent + t.server_sent).sum() / (t.real_out + t.real_in).sum(),
                             kept_pct=100 * (t.client_sent + t.server_sent).sum() / (t.client_scheduled + t.server_scheduled).sum())
            assert p[8].index.equals(p[24].index)
            a, b = p[8].values, p[24].values
            lost = int(((a == 1) & (b == 0)).sum()); gained = int(((a == 0) & (b == 1)).sum())
            pval = binomtest(gained, lost + gained, .5).pvalue if lost + gained else 1.0
            rows.append(dict(grouping=gcol.replace('_group', ''), group=grp, seed=seed, n_visits=len(ids),
                             n_predictions=len(a), n_visits_tested=p[8].index.get_level_values(1).nunique(),
                             acc_w8=100 * a.mean(), acc_w24=100 * b.mean(), delta_acc=100 * (b.mean() - a.mean()),
                             ovh_w8=st[8]['overhead_pct'], ovh_w24=st[24]['overhead_pct'],
                             kept_w8=st[8]['kept_pct'], kept_w24=st[24]['kept_pct'],
                             correct_to_wrong=lost, wrong_to_correct=gained, mcnemar_p=pval))
e6 = pd.DataFrame(rows)
e6.to_csv(OUT / 'exp6_by_seed.csv', index=False, float_format='%.4f')
agg = e6.groupby(['grouping', 'group'], sort=False).agg(
    n_visits=('n_visits', 'first'), n_predictions=('n_predictions', 'first'),
    n_visits_tested=('n_visits_tested', 'first'), acc_w8=('acc_w8', 'mean'), acc_w24=('acc_w24', 'mean'), delta_acc=('delta_acc', 'mean'),
    delta_min=('delta_acc', 'min'), delta_max=('delta_acc', 'max'),
    ovh_w8=('ovh_w8', 'mean'), ovh_w24=('ovh_w24', 'mean'), kept_w8=('kept_w8', 'mean'), kept_w24=('kept_w24', 'mean'),
    correct_to_wrong=('correct_to_wrong', 'sum'), wrong_to_correct=('wrong_to_correct', 'sum'))
agg.to_csv(OUT / 'exp6_summary.csv', float_format='%.3f')
did = {}
for gr in ('global', 'site'):
    x = e6[e6.grouping == gr].pivot(index='seed', columns='group', values='delta_acc')
    did[gr] = dict(mean=float((x.short - x.long).mean()), by_seed=(x.short - x.long).round(2).tolist())
res['exp6_short_minus_long_delta'] = did
res['exp6_mcnemar_by_seed'] = e6[['grouping', 'group', 'seed', 'correct_to_wrong', 'wrong_to_correct', 'mcnemar_p']].round(4).to_dict('records')
res['exp6_sites_in_groups'] = dict(short_sites_with_10plus=int((comp.short >= 10).sum()),
                                   short_max_from_one_site=int(comp.short.max()), long_max_from_one_site=int(comp.long.max()))
# full-set reference for #6 (the #2 numbers)
full = runs[runs.limit_per_direction == 1100].groupby('max_wnd_s')[['accuracy_pct', 'overhead_pct', 'kept_pct']].mean()
res['exp6_full_set'] = full.round(3).to_dict('index')

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
axs[1].sharey(axs[0])
for ax, gr, title in [(axs[0], 'global', 'Accuracy · global quartiles'), (axs[1], 'site', 'Accuracy · within-site quartiles')]:
    sub = e6[e6.grouping == gr]
    for i, grp in enumerate(['short', 'long']):
        for w, off in ((8, -.12), (24, .12)):
            v = sub[sub.group == grp][f'acc_w{w}'].values
            ax.scatter([i + off] * 3, v, s=26, color=C[w], alpha=.4, edgecolor='white', zorder=3)
            ax.scatter([i + off], [v.mean()], s=70, color=C[w], edgecolor='white', lw=1.5, zorder=4)
        d = sub[sub.group == grp].delta_acc.mean()
        ax.annotate(f'Δ {d:+.1f} pp', (i, sub[sub.group == grp][['acc_w8', 'acc_w24']].values.min()),
                    textcoords='offset points', xytext=(0, -16), ha='center', fontsize=9)
    ax.set_xticks([0, 1], ['Short visits\n(bottom 25%)', 'Long visits\n(top 25%)'])
    ax.set_xlim(-.6, 1.6); ax.set_ylim(78, 97)
    ax.set_title(title, loc='left', fontsize=10)
    ax.set_ylabel('kFP accuracy (%)')
ax = axs[2]
sub = agg.loc['global']
for i, grp in enumerate(['short', 'long']):
    for w, off in ((8, -.12), (24, .12)):
        v = sub.loc[grp, f'kept_w{w}']
        ax.bar(i + off, v, width=.22, color=C[w], zorder=3)
        ax.annotate(f'{v:.0f}%', (i + off, v), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8.5)
ax.set_xticks([0, 1], ['Short visits', 'Long visits'])
ax.set_ylim(0, 110); ax.set_ylabel('Scheduled dummies kept (%)')
ax.set_title('Dummies kept · global quartiles', loc='left', fontsize=10)
fig.legend(handles=[plt.Line2D([], [], color=C[w], marker='o', lw=0, ms=8, label=f'Window 1–{w} s') for w in (8, 24)],
           frameon=False, loc='upper right', ncol=2)
fig.suptitle('Exp. 6 · 1–8 s vs 1–24 s on short and long visits (1,100 / 1,100 cells; dots = seeds)', x=.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(OUT / 'exp6_short_long.png', dpi=170); plt.close(fig)

(OUT / 'results.json').write_text(json.dumps(res, indent=2, default=float))
print(s5.round(2).to_string()); print(json.dumps(res['exp5'], indent=1, default=float))
print(agg.round(2).to_string()); print(json.dumps({k: res[k] for k in res if k != 'exp5' and k != 'exp6_mcnemar_by_seed'}, indent=1, default=float))
print(e6[['grouping', 'group', 'seed', 'delta_acc', 'correct_to_wrong', 'wrong_to_correct', 'mcnemar_p']].round(3).to_string())
