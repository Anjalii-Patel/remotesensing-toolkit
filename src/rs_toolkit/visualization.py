import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def show(arr, bands=(0, 1, 2), title=None, ax=None):
    """Percentile-stretched display of [C,H,W]; 1 band -> gray."""
    a = arr[list(bands)] if arr.shape[0] > 1 else arr[:1]
    lo, hi = np.nanpercentile(a, (2, 98))
    a = np.clip((a - lo) / max(hi - lo, 1e-8), 0, 1)
    ax = ax or plt.gca()
    ax.imshow(a[0] if a.shape[0] == 1 else np.moveaxis(a, 0, -1), cmap="gray")
    ax.set_title(title or ""); ax.axis("off")
    return ax

def show_index(idx, title="Index", cmap="RdYlGn", vmin=-1, vmax=1, ax=None):
    """Display a 2-D spectral index with a diverging colormap centered at 0."""
    ax = ax or plt.gca()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(idx, cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.axis("off")
    return ax

def show_histogram(arr, band_labels=None, bins=128, ax=None):
    """Per-band histogram of a [C,H,W] array (NaN-aware)."""
    ax = ax or plt.gca()
    C = arr.shape[0]
    labels = band_labels or [f"Band {i}" for i in range(C)]
    for c in range(C):
        vals = arr[c][np.isfinite(arr[c])]
        ax.hist(vals.ravel(), bins=bins, alpha=0.55, label=labels[c])
    ax.set_xlabel("Value"); ax.set_ylabel("Count"); ax.legend(fontsize=7)
    ax.set_title("Band histograms")
    return ax

def show_grid(patches, n=16, bands=(0, 1, 2)):
    """Quick grid display of the first *n* patches [N,C,H,W]."""
    n = min(n, len(patches)); cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i, ax in enumerate(np.asarray(axes).ravel()):
        if i < n: show(patches[i], bands=bands, ax=ax)
        else: ax.axis("off")
    fig.tight_layout()
    return fig

def show_split_map(coords, splits, size, title="Spatial split", ax=None):
    """Colour patches by train/val/test membership on a 2-D scatter."""
    ax = ax or plt.gca(); colors = ["#2196F3", "#FF9800", "#F44336"]; names = ["Train", "Val", "Test"]
    for idx_arr, c, n in zip(splits, colors, names):
        pts = coords[idx_arr]
        ax.scatter(pts[:, 1], pts[:, 0], s=4, c=c, label=n, alpha=0.7)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_title(title)
    return ax
