"""Shared fixtures for rasterio-based tests (no real GeoTIFF needed)."""
import numpy as np, pytest, tempfile, rasterio
from rasterio.transform import from_bounds

@pytest.fixture
def tmp_tif(tmp_path):
    """Create a 4-band 64x64 synthetic GeoTIFF and return its path."""
    path = tmp_path / "synthetic.tif"
    rng = np.random.default_rng(42)
    arr = rng.random((4, 64, 64), dtype=np.float32)
    arr[0, 0, 0] = -9999  # a nodata pixel
    transform = from_bounds(0, 0, 64, 64, 64, 64)
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64,
                       count=4, dtype="float32", crs="EPSG:32633",
                       transform=transform, nodata=-9999) as ds:
        ds.write(arr)
    return path

@pytest.fixture
def sample_arr():
    """4-band (C,H,W) float32 array with a few NaN pixels."""
    rng = np.random.default_rng(0)
    a = rng.random((4, 128, 128), dtype=np.float32)
    a[:, 0, 0] = np.nan
    return a
