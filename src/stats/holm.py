"""Holm adjustment and the explicitly chosen bootstrap tail-area statistic."""
import numpy as np


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a finite vector in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def bootstrap_pvalue(draws):
    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap p-value requires finite scalar replicates")
    return min(1.0, 2 * min((np.count_nonzero(values <= 0) + 1) / (len(values) + 1),
                          (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)))
