# Step 1: reproduce the main experiment of ECOD (the paper's Tables 4 and 5) on 14 ODDS datasets.
#
# Protocol (Section 4.1 of the paper and the authors' benchmark script in PyOD):
#   * split each dataset 10 times (seeds 0-9) into 60% fitting data and 40% test data
#   * standardise the columns on the 60% part
#   * fit every detector on the 60% part and score the 40% part
#   * measure ROC-AUC and average precision (AP) and take the mean of the 10 runs
#
# Usage:  python run_benchmark.py --data data/ODDS --out results

import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.utils.utility import standardizer

from ecod_numpy import ECODNumpy

DATASETS = ['arrhythmia', 'breastw', 'cardio', 'ionosphere', 'lympho', 'mammography', 'optdigits',
            'pima', 'satellite', 'satimage-2', 'shuttle', 'speech', 'wbc', 'wine']
N_RUNS = 10
TEST_SIZE = 0.4


# Read one ODDS .mat file: X = data, y = 1 for outliers and 0 for normal rows.
# Like the paper we use 10,000 random rows of Shuttle (the file has 49,097).
def load(folder, name):
    m = sio.loadmat(os.path.join(folder, name + '.mat'))
    X = np.asarray(m['X'], dtype=float)
    y = np.asarray(m['y']).ravel().astype(int)
    if name == 'shuttle':
        rows = np.random.RandomState(0).choice(len(X), 10_000, replace=False)
        X, y = X[rows], y[rows]
    return X, y


# scikit-learn refuses scores that are NaN or infinite, so we replace them.
# ABOD gives NaN to points whose neighbours are all duplicates. We rank these as
# most outlying: only that gives the paper's ABOD value on Breastw (0.396, not 0.815).
def clean(scores):
    finite = np.isfinite(scores)
    if finite.all():
        return scores
    low, high = scores[finite].min(), scores[finite].max()
    return np.nan_to_num(scores, nan=high, posinf=high, neginf=low)


# All methods we run. The baselines use PyOD's defaults, as in the authors' script.
# Exceptions: CBLOF uses n_clusters=10 like the authors' script, and two scikit-learn
# defaults that changed after 2022 are set back to their 2022 values:
# KMeans n_init=10 (CBLOF) and svd_solver='full' (PCA).
# contamination only sets the 0/1 labels, so it does not change ROC-AUC or AP.
def detectors(contamination, seed):
    c = contamination
    return {
        'ECOD (PyOD 3.6.7)': lambda: ECOD(contamination=c),
        # our own versions of ECOD (ecod_numpy.py)
        'Own: PyOD 3.6.7': lambda: ECODNumpy('pyod'),
        'Own: PyOD 2022': lambda: ECODNumpy('pyod2022'),
        'Own: Algorithm 1': lambda: ECODNumpy('paper'),
        'Own: automatic only': lambda: ECODNumpy('auto'),
        'Own: ECOD-L': lambda: ECODNumpy('left'),
        'Own: ECOD-R': lambda: ECODNumpy('right'),
        'Own: ECOD-B': lambda: ECODNumpy('both'),
        # the nine baselines
        'ABOD': lambda: ABOD(contamination=c),
        'CBLOF': lambda: CBLOF(contamination=c, n_clusters=10, check_estimator=False, random_state=seed,
                               clustering_estimator=KMeans(n_clusters=10, n_init=10, random_state=seed)),
        'HBOS': lambda: HBOS(contamination=c),
        'IForest': lambda: IForest(contamination=c, random_state=seed),
        'KNN': lambda: KNN(contamination=c),
        'LODA': lambda: LODA(contamination=c, random_state=seed),
        'LOF': lambda: LOF(contamination=c),
        'OCSVM': lambda: OCSVM(contamination=c),
        'PCA': lambda: PCA(contamination=c, svd_solver='full', random_state=seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/ODDS')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, largest_diff = [], 0.0
    for name in DATASETS:
        X, y = load(args.data, name)
        # rows, columns and % outliers, to compare with Table 1 of the paper
        print(f'{name:12s} rows={len(X):6d} columns={X.shape[1]:4d} outliers={100 * y.mean():.2f}%', flush=True)
        for seed in range(N_RUNS):
            # the same split and scaling for every method
            X_tr, X_te, _, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
            X_tr, X_te = standardizer(X_tr, X_te)
            scores = {}
            for method, make in detectors(y.mean(), seed).items():
                s = clean(make().fit(X_tr).decision_function(X_te))
                scores[method] = s
                rows.append(dict(dataset=name, seed=seed, method=method,
                                 roc=roc_auc_score(y_te, s), ap=average_precision_score(y_te, s)))
            # check that our own ECOD gives the same scores as PyOD's ECOD
            largest_diff = max(largest_diff, np.abs(scores['ECOD (PyOD 3.6.7)'] - scores['Own: PyOD 3.6.7']).max())

    # mean, standard deviation and range over the 10 runs
    g = pd.DataFrame(rows).groupby(['dataset', 'method'])
    mean = g[['roc', 'ap']].mean()
    mean['roc_std'], mean['ap_std'] = g['roc'].std(), g['ap'].std()
    mean['roc_min'], mean['roc_max'] = g['roc'].min(), g['roc'].max()
    mean.reset_index().to_csv(os.path.join(args.out, 'mean_results.csv'), index=False)
    print('largest score difference between our ECOD and PyOD:', largest_diff)


if __name__ == '__main__':
    main()
