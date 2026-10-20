# Reproducing ECOD (Li et al., 2022)

Code for Task 2 of the INFOMDW Data Wrangling assignment. It repeats the main experiment of
[ECOD](https://arxiv.org/abs/2201.00382v1) on the paper's 14 ODDS datasets with 9 of its 11 other detectors.

## Files

| File | What it does |
| --- | --- |
| `ecod_numpy.py` | Our own ECOD in NumPy, in every version we compare |
| `run_benchmark.py` | Step 1: runs all methods on all datasets and writes `results/mean_results.csv` |
| `compare_with_paper.py` | Step 2: compares our results with the paper and prints the tables below |
| `paper_values.json` | The paper's numbers from its Tables 2 to 5 |

## How to run

Put the 14 ODDS `.mat` files from [ODDS](https://odds.cs.stonybrook.edu/) in `data/ODDS/`. Then run:

```bash
pip install -r requirements.txt
python run_benchmark.py --data data/ODDS --out results      # about 6 minutes
python compare_with_paper.py --results results
```

## Settings

Each dataset is split 10 times into 60% training and 40% test data with seeds 0 to 9. This follows
the authors' benchmark script in PyOD. All detectors use the PyOD 3.6.7 defaults. CBLOF uses
`n_clusters=10` like that script. CBLOF and PCA use the scikit-learn defaults of 2022:
`KMeans(n_init=10)` and `svd_solver='full'`.

## Results

All 10 methods averaged over the 14 datasets, with the rank in brackets. The last column counts
the datasets where our ROC-AUC is within 0.01 of the paper.

| Method | ROC paper | ROC ours | AP paper | AP ours | Within 0.01 |
| --- | --- | --- | --- | --- | --- |
| ECOD (PyOD 3.6.7) | 0.847 (1) | 0.800 (4) | 0.608 (1) | 0.525 (5) | 3 of 14 |
| IForest | 0.836 (2) | 0.837 (1) | 0.574 (2) | 0.568 (1) | 12 of 14 |
| HBOS | 0.832 (3) | 0.831 (2) | 0.543 (3) | 0.541 (2) | 14 of 14 |
| PCA | 0.804 (4) | 0.803 (3) | 0.534 (4) | 0.533 (3) | 12 of 14 |
| OCSVM | 0.797 (5) | 0.797 (5) | 0.528 (5) | 0.529 (4) | 14 of 14 |
| CBLOF | 0.763 (6) | 0.764 (6) | 0.466 (6) | 0.472 (6) | 13 of 14 |
| KNN | 0.757 (7) | 0.753 (7) | 0.424 (8) | 0.423 (8) | 13 of 14 |
| LODA | 0.737 (8) | 0.737 (8) | 0.455 (7) | 0.443 (7) | 4 of 14 |
| LOF | 0.666 (9) | 0.669 (9) | 0.327 (9) | 0.327 (9) | 13 of 14 |
| ABOD | 0.663 (10) | 0.659 (10) | 0.291 (10) | 0.292 (10) | 12 of 14 |

ECOD per dataset (ROC-AUC). *PyOD 3.6.7* is PyOD's ECOD today. *PyOD 2022* is the code before a
bug fix in April 2022. *Algorithm 1* follows Eq. 6 of the paper. *Automatic only* uses just the
tail chosen by the skewness.

| Dataset | Paper | PyOD 3.6.7 | PyOD 2022 | Algorithm 1 | Automatic only |
| --- | --- | --- | --- | --- | --- |
| Arrhythmia | 0.802 | 0.817 | 0.834 | 0.794 | 0.792 |
| Breastw | 0.994 | 0.991 | 0.991 | 0.989 | 0.994 |
| Cardio | 0.897 | 0.936 | 0.951 | 0.880 | 0.825 |
| Ionosphere | 0.831 | 0.736 | 0.770 | 0.717 | 0.767 |
| Lympho | 0.994 | 0.996 | 0.998 | 0.986 | 0.993 |
| Mammography | 0.894 | 0.909 | 0.909 | 0.899 | 0.905 |
| Optdigits | 0.733 | 0.600 | 0.614 | 0.707 | 0.733 |
| Pima | 0.664 | 0.591 | 0.516 | 0.585 | 0.678 |
| Satellite | 0.661 | 0.581 | 0.745 | 0.495 | 0.734 |
| Satimage-2 | 0.985 | 0.970 | 0.976 | 0.886 | 0.987 |
| Shuttle | 0.998 | 0.993 | 0.994 | 0.995 | 0.997 |
| Speech | 0.485 | 0.450 | 0.472 | 0.450 | 0.519 |
| WBC | 0.975 | 0.886 | 0.886 | 0.933 | 0.975 |
| Wine | 0.949 | 0.750 | 0.739 | 0.917 | 0.958 |
| **Average ROC** | **0.847** | **0.800** | **0.814** | **0.802** | **0.847** |
| **Average AP** | **0.608** | **0.525** | **0.539** | **0.480** | **0.605** |
