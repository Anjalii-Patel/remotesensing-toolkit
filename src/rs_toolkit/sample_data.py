"""Generate synthetic rasters for testing and demos (no real data needed).

Includes a Landsat-8-like scene generator that writes a proper GeoTIFF with
real CRS, pixel size, and spectrally realistic band values so notebooks
can demonstrate the full preprocessing pipeline without downloading data.
"""
import numpy as np, os

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
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "dem.npy"), make_dem(height, width, seed))
    np.save(os.path.join(out_dir, "optical.npy"), make_optical(height, width, 4, seed))
    np.save(os.path.join(out_dir, "qa.npy"), make_qa(height, width, seed=seed))
    print(f"Sample dataset written to {out_dir}/")

# ── Landsat-8 style GeoTIFF generator ──────────────────────────────────

# Realistic surface-reflectance ranges per Landsat-8 OLI band (0-1 scale)
_LANDSAT8_BANDS = {
    "B2_Blue":  {"mean": 0.09, "std": 0.03},
    "B3_Green": {"mean": 0.11, "std": 0.04},
    "B4_Red":   {"mean": 0.10, "std": 0.05},
    "B5_NIR":   {"mean": 0.30, "std": 0.10},
    "B6_SWIR1": {"mean": 0.22, "std": 0.08},
    "B7_SWIR2": {"mean": 0.14, "std": 0.06},
}

def make_landsat_scene(
    path,
    height=1024,
    width=1024,
    pixel_size=30.0,
    epsg=32643,
    origin_x=500_000.0,
    origin_y=2_500_000.0,
    cloud_frac=0.08,
    nodata_frac=0.02,
    seed=42,
):
    """Write a 6-band Landsat-8-like GeoTIFF with realistic metadata.

    Band order: Blue, Green, Red, NIR, SWIR1, SWIR2
    (matches Landsat-8 OLI bands 2-7)

    The scene includes:
    - Spectrally realistic reflectance values per band
    - Spatial correlation (vegetation patches via Perlin-like noise)
    - Scattered nodata pixels (-9999)
    - A separate QA band written as <path>_QA.tif

    Parameters
    ----------
    path : str
        Output GeoTIFF path.
    height, width : int
        Scene dimensions in pixels.
    pixel_size : float
        Ground sampling distance in metres (default 30 m, Landsat).
    epsg : int
        EPSG code for UTM zone (default 32643 = UTM 43N, central India).
    origin_x, origin_y : float
        Upper-left corner easting/northing in CRS units.
    cloud_frac : float
        Approximate fraction of cloud pixels.
    nodata_frac : float
        Approximate fraction of nodata pixels.
    seed : int
        Random seed.

    Returns
    -------
    path : str   (the optical scene)
    qa_path : str (the QA band)
    """
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    # ── spatial structure: vegetation-like patches ──────────────────────
    y, x = np.mgrid[0:height, 0:width]
    veg = 0.5 + 0.5 * np.sin(x / 60.0) * np.cos(y / 45.0)  # 0-1, vegetation density
    veg += rng.normal(0, 0.15, veg.shape)
    veg = np.clip(veg, 0, 1)

    # ── generate bands with spectral correlation ───────────────────────
    bands = []
    for info in _LANDSAT8_BANDS.values():
        base = info["mean"] + info["std"] * rng.standard_normal((height, width))
        bands.append(base.astype(np.float32))

    arr = np.stack(bands)  # (6, H, W)

    # NIR (band 3) is high where vegetation is dense
    arr[3] = np.clip(arr[3] + 0.25 * veg, 0, 1)
    # Red (band 2) is suppressed where vegetation is dense
    arr[2] = np.clip(arr[2] - 0.08 * veg, 0, 1)
    # SWIR bands partially anti-correlate with vegetation
    arr[4] = np.clip(arr[4] - 0.05 * veg, 0, 1)
    arr[5] = np.clip(arr[5] - 0.04 * veg, 0, 1)

    arr = np.clip(arr, 0, 1)

    # ── inject nodata ──────────────────────────────────────────────────
    nodata_mask = rng.random((height, width)) < nodata_frac
    arr[:, nodata_mask] = -9999.0

    # ── write scene ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    band_names = list(_LANDSAT8_BANDS.keys())

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=6, dtype="float32", crs=f"EPSG:{epsg}",
        transform=transform, nodata=-9999, compress="deflate",
    ) as ds:
        ds.write(arr)
        for i, name in enumerate(band_names, 1):
            ds.set_band_description(i, name)

    # ── write QA band ──────────────────────────────────────────────────
    qa_path = path.replace(".tif", "_QA.tif")
    qa = make_qa(height, width, cloud_frac, seed)
    qa[nodata_mask] = 1  # bit 0 = fill

    with rasterio.open(
        qa_path, "w", driver="GTiff", height=height, width=width,
        count=1, dtype="uint16", crs=f"EPSG:{epsg}",
        transform=transform, nodata=0,
    ) as ds:
        ds.write(qa[np.newaxis])
        ds.set_band_description(1, "QA_PIXEL")

    print(f"Landsat-8 scene:  {path}  ({height}×{width}, {pixel_size}m, EPSG:{epsg})")
    print(f"QA band:          {qa_path}")
    print(f"Bands:            {', '.join(band_names)}")
    return path, qa_path
