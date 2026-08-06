import numpy as np

def _starts(n, size, stride):
    s = list(range(0, max(n - size, 0) + 1, stride))
    if s[-1] + size < n: s.append(n - size)
    return s

def tile_grid(h, w, size, overlap=0):
    stride = size - overlap
    return [(r, c) for r in _starts(h, size, stride) for c in _starts(w, size, stride)]

def extract_patches(arr, size, overlap=0, max_nan_frac=0.2):
    """Return patches[N,C,size,size] and their (row,col) origins."""
    _, h, w = arr.shape
    P, X = [], []
    for r, c in tile_grid(h, w, size, overlap):
        p = arr[:, r:r + size, c:c + size]
        if p.shape[1:] == (size, size) and np.isnan(p).mean() <= max_nan_frac:
            P.append(p); X.append((r, c))
    if not P:
        return np.empty((0, arr.shape[0], size, size), dtype=arr.dtype), np.empty((0, 2), dtype=int)
    return np.stack(P), np.array(X)

def spatial_split(coords, size, ratios=(0.7, 0.15, 0.15), block=1024, seed=0):
    """Split patches by spatial blocks to avoid train/test leakage.

    Why spatial splitting?
        Randomly splitting adjacent satellite patches produces overly
        optimistic evaluation because spatially correlated samples appear
        in both training and test sets (Tobler's First Law of Geography).
        This function assigns entire spatial *blocks* to a single split,
        so nearby patches never leak across train / val / test.
        (Note: patches that straddle block borders may still have minor overlap 
        with adjacent blocks if extraction uses an overlap parameter).

    Algorithm:
        1. Map each patch to a coarse grid cell of size *block* pixels.
        2. Randomly assign each grid cell to train (0), val (1), or test (2)
           according to *ratios*.
        3. Return per-patch index arrays for each split.

    Returns list of three index arrays: [train_idx, val_idx, test_idx].
    """
    ids = np.array([(r // block, c // block) for r, c in coords])
    _, inv = np.unique(ids, axis=0, return_inverse=True)
    inv = inv.ravel(); nb = inv.max() + 1
    order = np.random.default_rng(seed).permutation(nb)
    n1 = int(round(ratios[0] * nb)); n2 = n1 + int(round(ratios[1] * nb))
    grp = np.empty(nb, int); grp[order[:n1]] = 0; grp[order[n1:n2]] = 1; grp[order[n2:]] = 2
    g = grp[inv]
    return [np.where(g == k)[0] for k in range(3)]
