"""Synthetic DEM + optical demo: derive, normalize, tile, split, plot."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rs_toolkit import terrain, optical, normalization, tiling
from rs_toolkit.visualization import show

rng = np.random.default_rng(0)
y, x = np.mgrid[0:512, 0:512]
dem = 30 * np.sin(x / 40) + 25 * np.cos(y / 55) + rng.normal(0, .3, x.shape)  # metres
res = 1.0
hs = terrain.hillshade(dem, res)
opt = np.clip(hs * (0.8 + 0.2 * rng.random(hs.shape)) + rng.normal(0, .05, hs.shape), 0, 1)  # fake optical
gray = optical.robust_stretch(optical.to_gray(np.stack([opt] * 3)))

stack = np.stack([terrain.slope(dem, res), hs, gray[0], optical.gradient_magnitude(gray[0])]).astype("float32")
stack = normalization.zscore(stack)
patches, xy = tiling.extract_patches(stack, 128, overlap=32)
tr, va, te = tiling.spatial_split(xy, 128, block=256)
print("patches", patches.shape, "split", len(tr), len(va), len(te))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
show(dem[None], title="DEM", ax=ax[0]); show(hs[None], title="Hillshade", ax=ax[1]); show(gray, title="Optical gray", ax=ax[2])
fig.savefig("demo.png", dpi=100, bbox_inches="tight")
