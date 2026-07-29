import numpy as np

def to_gray(rgb, w=(0.299, 0.587, 0.114)):
    """[3,H,W] -> [1,H,W] luminance."""
    return np.tensordot(np.array(w, "float32"), rgb[:3], 1)[None]

def robust_stretch(a, lo=1, hi=99):
    l, h = np.nanpercentile(a, (lo, hi))
    return np.clip((a - l) / max(h - l, 1e-8), 0, 1)

def hist_equalize(a, bins=256):
    h, e = np.histogram(a[~np.isnan(a)], bins)
    return np.where(np.isnan(a), np.nan, np.interp(a, e[1:], np.cumsum(h) / h.sum()))

def gradient_magnitude(gray):
    """Edge map (2D); shared structure with DEM slope/hillshade."""
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)
