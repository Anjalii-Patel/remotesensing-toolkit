import numpy as np

def band_stats(arr):
    """Per-band statistics for [C,H,W] array. Returns list of dicts."""
    stats = []
    for c in range(arr.shape[0]):
        band = arr[c]
        valid = band[np.isfinite(band)]
        stats.append({
            "band": c, "min": float(np.min(valid)), "max": float(np.max(valid)),
            "mean": float(np.mean(valid)), "std": float(np.std(valid)),
            "median": float(np.median(valid)),
            "valid_pct": round(100 * len(valid) / band.size, 2),
        })
    return stats

def coverage_report(arr):
    """Fraction of valid (finite) pixels per band and overall."""
    per_band = [float(np.isfinite(arr[c]).mean()) for c in range(arr.shape[0])]
    return {"per_band": per_band, "overall": float(np.isfinite(arr).all(axis=0).mean())}

def correlation_matrix(arr):
    """Inter-band Pearson correlation [C,C] from [C,H,W]."""
    C = arr.shape[0]
    flat = arr.reshape(C, -1)
    mask = np.isfinite(flat).all(axis=0)
    return np.corrcoef(flat[:, mask])

def print_stats(arr, band_names=None):
    """Pretty-print band statistics."""
    names = band_names or [f"Band {i}" for i in range(arr.shape[0])]
    stats = band_stats(arr)
    print(f"{'Band':<12} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Valid%':>7}")
    print("-" * 55)
    for s, n in zip(stats, names):
        print(f"{n:<12} {s['min']:8.3f} {s['max']:8.3f} {s['mean']:8.3f} {s['std']:8.3f} {s['valid_pct']:6.1f}%")
