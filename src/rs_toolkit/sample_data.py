"""Generate synthetic rasters for testing and demos (no real data needed)."""
import numpy as np

def make_dem(height=512, width=512, seed=42):
    """Synthetic DEM with ridges, valleys, and optional NaN voids."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    dem = 30 * np.sin(x / 40) + 25 * np.cos(y / 55) + rng.normal(0, 0.3, (height, width))
    return dem.astype("float32")

def make_optical(height=512, width=512, bands=4, seed=42):
    """Synthetic multispectral image [C,H,W] float32 in [0,1]."""
    rng = np.random.default_rng(seed)
    arr = rng.random((bands, height, width), dtype=np.float32)
    # band 3 (NIR) brighter than band 2 (Red) to simulate vegetation
    arr[3] = np.clip(arr[3] * 1.5 + 0.1, 0, 1)
    return arr

def make_qa(height=512, width=512, cloud_frac=0.1, seed=42):
    """Synthetic QA band with cloud bits (bit 4) set randomly."""
    rng = np.random.default_rng(seed)
    qa = np.zeros((height, width), dtype=np.uint16)
    cloud = rng.random((height, width)) < cloud_frac
    qa[cloud] |= (1 << 4)   # set cloud bit
    return qa

def make_sample_dataset(out_dir, height=256, width=256, seed=42):
    """Write a complete sample dataset (DEM + optical + QA) to out_dir as .npy files."""
    import os; os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "dem.npy"), make_dem(height, width, seed))
    np.save(os.path.join(out_dir, "optical.npy"), make_optical(height, width, 4, seed))
    np.save(os.path.join(out_dir, "qa.npy"), make_qa(height, width, seed=seed))
    print(f"Sample dataset written to {out_dir}/")
