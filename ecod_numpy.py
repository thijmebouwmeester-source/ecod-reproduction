# Our own implementation of ECOD (Li et al., IEEE TKDE 2022) in NumPy.
#
# ECOD gives every row an outlier score in three steps:
#   1. Per column: how rare is each value on the left (share of rows <= value)
#      and on the right (share of rows >= value)?
#   2. Turn each share into points with -log. Rare values get many points:
#      a share of 0.01 gives 4.6 points and a share of 0.5 gives 0.7.
#   3. Add up the points of all columns into one score per row.
#
# Step 3 can be done in several ways. These are the versions we compare:
#   "paper"     Algorithm 1 / Eq. 6 of the paper: the highest of three totals
#   "pyod"      PyOD 3.6.7 (since 3 April 2022): per column the highest tail, then the sum
#   "pyod2022"  PyOD 0.9.7 and 0.9.8 (January to April 2022), see below
#   "auto"      only the tail that the skewness points to
#   "left", "right", "both"   ECOD-L, ECOD-R and ECOD-B from the paper's Table 2

import numpy as np
from scipy.stats import skew


# Step 1 and 2: left-tail points, right-tail points and the skewness sign per column
def tail_points(X):
    n = len(X)
    left_share = np.empty(X.shape)
    right_share = np.empty(X.shape)
    for j in range(X.shape[1]):
        sorted_col = np.sort(X[:, j])
        # share of rows with a value <= x (left tail) and >= x (right tail)
        left_share[:, j] = np.searchsorted(sorted_col, X[:, j], side="right") / n
        right_share[:, j] = (n - np.searchsorted(sorted_col, X[:, j], side="left")) / n
    U_l = -np.log(left_share)    # points for being rare on the left
    U_r = -np.log(right_share)   # points for being rare on the right
    # skewness sign: -1 = long left tail, +1 = long right tail, 0 = no skew (e.g. a constant column)
    sign = np.sign(np.nan_to_num(skew(X, axis=0)))
    return U_l, U_r, sign


# Step 3: add up the points of all columns into one score per row
def combine(U_l, U_r, sign, variant):
    if variant == "left":    # ECOD-L: only left tails
        return U_l.sum(axis=1)
    if variant == "right":   # ECOD-R: only right tails
        return U_r.sum(axis=1)
    if variant == "both":    # ECOD-B: the average of the two
        return (U_l.sum(axis=1) + U_r.sum(axis=1)) / 2
    if variant == "paper":   # Algorithm 1: skew < 0 -> left tail, skew >= 0 -> right tail
        U_auto = np.where(sign < 0, U_l, U_r)
        return np.maximum.reduce([U_l.sum(axis=1), U_r.sum(axis=1), U_auto.sum(axis=1)])

    # PyOD picks the tail per column like this: left if skew < 0, right if skew > 0
    # and both added up if skew == 0
    U_skew = np.where(sign < 0, U_l, 0) + np.where(sign > 0, U_r, 0) + np.where(sign == 0, U_l + U_r, 0)
    if variant == "auto":
        return U_skew.sum(axis=1)
    if variant == "pyod":
        return np.maximum(np.maximum(U_l, U_r), U_skew).sum(axis=1)
    if variant == "pyod2022":
        # The 2022 code says np.maximum(U_skew, U_l, U_r). The third argument of
        # np.maximum is where NumPy stores the result, so U_r is never compared.
        return np.maximum(U_skew, U_l).sum(axis=1)
    raise ValueError(f"unknown variant {variant!r}")


# Same fit / decision_function interface as PyOD's detectors
class ECODNumpy:
    def __init__(self, variant="paper"):
        self.variant = variant

    def fit(self, X):
        self.X_train = np.asarray(X, dtype=float)
        return self

    def decision_function(self, X):
        # Like PyOD: the shares are counted on the training rows and the new rows together.
        # Only the scores of the new rows are returned.
        X = np.asarray(X, dtype=float)
        U_l, U_r, sign = tail_points(np.vstack([self.X_train, X]))
        return combine(U_l, U_r, sign, self.variant)[-len(X):]
