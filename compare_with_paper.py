# Step 2: compare our results with the paper (run after run_benchmark.py).
#
# Usage:  python compare_with_paper.py --results results
#
# Prints the tables and numbers used in the report and writes
# results/comparison_variant_rows.csv: in which row of the paper's Tables 2 and 3 our ECOD-L/R/B values appear.

import argparse
import json
import os

import numpy as np
import pandas as pd

from run_benchmark import DATASETS

BASELINES = ['ABOD', 'CBLOF', 'HBOS', 'IForest', 'KNN', 'LODA', 'LOF', 'OCSVM', 'PCA']
NO_RANDOMNESS = ['ABOD', 'HBOS', 'KNN', 'LOF', 'OCSVM']
# the columns of Table C: column name -> method name in our results
ECOD_VERSIONS = {'PyOD 3.6.7': 'ECOD (PyOD 3.6.7)', 'PyOD 2022': 'Own: PyOD 2022',
                 'Algorithm 1': 'Own: Algorithm 1', 'Automatic only': 'Own: automatic only'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    out = lambda name: os.path.join(args.results, name)

    # paper[metric][method][dataset] holds the paper's numbers (Tables 2-5)
    paper = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_values.json')))
    paper_series = lambda metric, method: pd.Series(paper[metric][method]).loc[DATASETS]
    # ours[method] is a table with one row per dataset (mean of the 10 runs)
    mean = pd.read_csv(out('mean_results.csv'))
    ours = {m: g.set_index('dataset').loc[DATASETS] for m, g in mean.groupby('method')}
    f = lambda x: f'{x:.3f}'

    # --- Table B: every method averaged over the 14 datasets, with its rank among the 10 ---
    rows = []
    for m in ['ECOD'] + BASELINES:
        o = ours['ECOD (PyOD 3.6.7)' if m == 'ECOD' else m]
        rows.append(dict(method='ECOD (PyOD 3.6.7)' if m == 'ECOD' else m,
                         roc_paper=paper_series('roc', m).mean(), roc_ours=o.roc.mean(),
                         ap_paper=paper_series('ap', m).mean(), ap_ours=o.ap.mean(),
                         within=int(((o.roc - paper_series('roc', m)).abs() <= 0.01).sum())))
    b = pd.DataFrame(rows)
    ranks = b[['roc_paper', 'roc_ours', 'ap_paper', 'ap_ours']].rank(ascending=False, method='min').astype(int)
    md = ['| Method | ROC paper | ROC ours | AP paper | AP ours | Within 0.01 |', '| --- | --- | --- | --- | --- | --- |']
    for i in b.sort_values('roc_paper', ascending=False).index:
        r, k = b.loc[i], ranks.loc[i]
        md.append(f'| {r.method} | {f(r.roc_paper)} ({k.roc_paper}) | {f(r.roc_ours)} ({k.roc_ours}) | '
                  f'{f(r.ap_paper)} ({k.ap_paper}) | {f(r.ap_ours)} ({k.ap_ours}) | {r.within} of 14 |')

    # --- Table C: ECOD per dataset, the paper and our four versions ---
    c = pd.DataFrame({'Paper': paper_series('roc', 'ECOD')})
    for col, method in ECOD_VERSIONS.items():
        c[col] = ours[method].roc
    md += ['', '| Dataset | ' + ' | '.join(c.columns) + ' |', '| --- |' + ' --- |' * len(c.columns)]
    for d, r in c.iterrows():
        md.append(f'| {"WBC" if d == "wbc" else d.capitalize()} | ' + ' | '.join(f(v) for v in r) + ' |')
    ap = [paper_series('ap', 'ECOD').mean()] + [ours[m].ap.mean() for m in ECOD_VERSIONS.values()]
    md.append('| **Average ROC** | ' + ' | '.join(f'**{f(v)}**' for v in c.mean()) + ' |')
    md.append('| **Average AP** | ' + ' | '.join(f'**{f(v)}**' for v in ap) + ' |')
    print('\n'.join(md) + '\n')

    # --- the methods without randomness should give the paper's numbers (3 decimals) ---
    exact = sum(abs(ours[m].loc[d, 'roc'] - paper['roc'][m][d]) <= 0.0015 for m in NO_RANDOMNESS for d in DATASETS)
    print(f'Methods without randomness: {exact} of {14 * len(NO_RANDOMNESS)} results equal the paper')

    # --- does the paper's ECOD value lie outside the range of our 10 runs? ---
    e, p = ours['ECOD (PyOD 3.6.7)'], paper_series('roc', 'ECOD')
    outside = ((p < e.roc_min) | (p > e.roc_max)).sum()
    print(f"the paper's ECOD value lies outside the range of our 10 runs on {outside} of 14 datasets")

    # --- rank of each ECOD version among the 10 methods (ROC-AUC) ---
    for col, method in ECOD_VERSIONS.items():
        rank = 1 + sum(ours[x].roc.mean() > ours[method].roc.mean() for x in BASELINES)
        print(f'ECOD {col:15s} average ROC {ours[method].roc.mean():.3f}, rank {rank} of 10')

    # --- the paper's Tables 2 and 3: in which printed row do our ECOD-L/R/B values appear? ---
    rows = []
    for metric, table in [('roc', 'table2_roc_all30'), ('ap', 'table3_ap_all30')]:
        printed = pd.DataFrame(paper[table])                  # all 30 rows as printed in the paper
        for d in DATASETS:
            mine = np.array([ours['Own: ' + v].loc[d, metric] for v in ['ECOD-L', 'ECOD-R', 'ECOD-B']])
            dist = np.abs(printed[['ECOD-L', 'ECOD-R', 'ECOD-B']].values - mine).max(axis=1)
            best = int(dist.argmin())                         # the printed row closest to our values
            rows.append(dict(metric=metric, dataset=d, ours=np.round(mine, 3).tolist(),
                             closest_printed_row=printed.loc[best, 'row'], max_abs_diff=round(dist[best], 3),
                             own_row=printed.loc[best, 'row'].lower() == f'{d} (mat)'))
    v = pd.DataFrame(rows)
    v.to_csv(out('comparison_variant_rows.csv'), index=False)
    for metric in ['roc', 'ap']:
        m = v[(v.metric == metric) & (v.max_abs_diff <= 0.001)]
        print(f'ECOD-L/R/B {metric}: {len(m)} of 14 found in the paper (within 0.001), {int(m.own_row.sum())} in their own row')


if __name__ == '__main__':
    main()
