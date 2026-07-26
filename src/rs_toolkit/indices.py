import numpy as np

def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((a + b) == 0, np.nan, (a - b) / (a + b))

def ndvi(nir, red): return _nd(nir, red)
def ndwi(green, nir): return _nd(green, nir)
def ndbi(swir, nir): return _nd(swir, nir)

def savi(nir, red, L=0.5):
    """Soil Adjusted Vegetation Index.  L=0.5 is the standard value."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((nir + red + L) == 0, np.nan,
                        ((nir - red) / (nir + red + L)) * (1 + L))

def evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """Enhanced Vegetation Index (Landsat/MODIS formula)."""
    denom = nir + C1 * red - C2 * blue + L
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, np.nan, G * (nir - red) / denom)

def nbr(nir, swir):
    """Normalized Burn Ratio — fire severity mapping."""
    return _nd(nir, swir)

def ndsi(green, swir):
    """Normalized Difference Snow Index."""
    return _nd(green, swir)

def bsi(blue, red, nir, swir):
    """Bare Soil Index."""
    num = (swir + red) - (nir + blue)
    den = (swir + red) + (nir + blue)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, np.nan, num / den)

INDEX_REGISTRY = {
    "ndvi": {"fn": ndvi,  "bands": ("nir", "red"),              "range": (-1, 1)},
    "ndwi": {"fn": ndwi,  "bands": ("green", "nir"),            "range": (-1, 1)},
    "ndbi": {"fn": ndbi,  "bands": ("swir", "nir"),             "range": (-1, 1)},
    "savi": {"fn": savi,  "bands": ("nir", "red"),              "range": (-1.5, 1.5)},
    "evi":  {"fn": evi,   "bands": ("nir", "red", "blue"),      "range": (-1, 1)},
    "nbr":  {"fn": nbr,   "bands": ("nir", "swir"),             "range": (-1, 1)},
    "ndsi": {"fn": ndsi,  "bands": ("green", "swir"),           "range": (-1, 1)},
    "bsi":  {"fn": bsi,   "bands": ("blue", "red", "nir", "swir"), "range": (-1, 1)},
}
