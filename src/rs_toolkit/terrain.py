import numpy as np

def _grad(dem, res):
    ry, rx = (res, res) if np.isscalar(res) else res
    dy, dx = np.gradient(dem, ry, rx)
    return dx, -dy  # east, north (rows increase southward)

def slope(dem, res=1.0):
    dx, dy = _grad(dem, res)
    return np.degrees(np.arctan(np.hypot(dx, dy)))

def aspect(dem, res=1.0):
    dx, dy = _grad(dem, res)
    return np.arctan2(dy, -dx)  # radians

def hillshade(dem, res=1.0, azimuth=315, altitude=45):
    dx, dy = _grad(dem, res)
    s = np.arctan(np.hypot(dx, dy)); a = np.arctan2(dy, -dx)
    zen = np.radians(90 - altitude); az = np.radians(360 - azimuth + 90)
    return np.clip(np.cos(zen) * np.cos(s) + np.sin(zen) * np.sin(s) * np.cos(az - a), 0, 1)

def fill_voids(dem, iters=50):
    """Fill NaN voids by iterative 3x3 neighbour averaging."""
    d = dem.copy()
    for _ in range(iters):
        m = np.isnan(d)
        if not m.any(): break
        p = np.pad(d, 1, constant_values=np.nan)
        st = np.stack([p[i:i + d.shape[0], j:j + d.shape[1]] for i in range(3) for j in range(3)])
        with np.errstate(all="ignore"):
            d[m] = np.nanmean(st, 0)[m]
    return d
