import numpy as np, rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from .io import read_raster

def nodata_mask(arr):
    """True where any band is NaN."""
    return np.isnan(arr).any(axis=0)

def bitmask(qa, bits):
    """Cloud/quality mask: True where any listed QA bit is set."""
    m = np.zeros(qa.shape, bool)
    for b in bits:
        m |= ((qa.astype(np.uint32) >> b) & 1).astype(bool)
    return m

def apply_mask(arr, mask):
    out = arr.copy(); out[:, mask] = np.nan; return out

def stack_bands(paths):
    """Stack single-band rasters (same grid) into [C,H,W]."""
    arrs, prof = zip(*(read_raster(p, [1]) for p in paths))
    return np.concatenate(arrs, 0), prof[0]

def reproject_raster(src, dst, crs, res=None, resampling=Resampling.bilinear):
    with rasterio.open(src) as s:
        t, w, h = calculate_default_transform(s.crs, crs, s.width, s.height, *s.bounds, resolution=res)
        p = s.profile.copy(); p.update(crs=crs, transform=t, width=w, height=h)
        with rasterio.open(dst, "w", **p) as d:
            for i in range(1, s.count + 1):
                reproject(rasterio.band(s, i), rasterio.band(d, i), src_transform=s.transform,
                          src_crs=s.crs, dst_transform=t, dst_crs=crs, resampling=resampling)
