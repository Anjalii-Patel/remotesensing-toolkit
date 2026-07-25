import numpy as np, rasterio

def read_raster(path, bands=None):
    """Return (array[C,H,W] float32 with nodata->NaN, profile)."""
    with rasterio.open(path) as ds:
        a = (ds.read(bands) if bands else ds.read()).astype("float32")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        return a, ds.profile

def write_raster(path, arr, profile):
    p = profile.copy()
    p.update(count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
             dtype="float32", nodata=np.nan)
    with rasterio.open(path, "w", **p) as ds:
        ds.write(arr.astype("float32"))
