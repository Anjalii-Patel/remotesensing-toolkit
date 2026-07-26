import numpy as np

def minmax(a):
    lo = np.nanmin(a, (1, 2), keepdims=True); hi = np.nanmax(a, (1, 2), keepdims=True)
    return (a - lo) / np.maximum(hi - lo, 1e-8)

def zscore(a):
    return (a - np.nanmean(a, (1, 2), keepdims=True)) / np.maximum(np.nanstd(a, (1, 2), keepdims=True), 1e-8)

def percentile_clip(a, lo=2, hi=98):
    l = np.nanpercentile(a, lo, (1, 2), keepdims=True); h = np.nanpercentile(a, hi, (1, 2), keepdims=True)
    return np.clip(a, l, h)
