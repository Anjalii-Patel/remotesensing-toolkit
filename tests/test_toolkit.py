import numpy as np, pytest
from rs_toolkit.tiling import extract_patches, spatial_split
from rs_toolkit.indices import ndvi, ndwi, ndbi, savi, evi, nbr, ndsi, bsi
from rs_toolkit.normalization import minmax, zscore, percentile_clip

# ── IO ──────────────────────────────────────────────────────────────────
def test_read_write_roundtrip(tmp_tif, tmp_path):
    from rs_toolkit.io import read_raster, write_raster
    arr, prof = read_raster(str(tmp_tif))
    assert arr.shape == (4, 64, 64)
    assert np.isnan(arr[0, 0, 0]), "nodata should be NaN"
    out = tmp_path / "out.tif"
    write_raster(str(out), arr, prof)
    arr2, _ = read_raster(str(out))
    valid = np.isfinite(arr) & np.isfinite(arr2)
    assert np.allclose(arr[valid], arr2[valid])

def test_read_single_band(tmp_tif):
    from rs_toolkit.io import read_raster
    arr, _ = read_raster(str(tmp_tif), bands=[1])
    assert arr.shape[0] == 1

# ── Preprocessing ───────────────────────────────────────────────────────
def test_nodata_mask(sample_arr):
    from rs_toolkit.preprocessing import nodata_mask
    m = nodata_mask(sample_arr)
    assert m[0, 0] is np.bool_(True)  # NaN pixel flagged

def test_bitmask():
    from rs_toolkit.preprocessing import bitmask
    qa = np.array([[0b11010]], dtype=np.uint16)
    assert bitmask(qa, [1]).item()      # bit 1 is set
    assert not bitmask(qa, [0]).item()  # bit 0 is not

def test_apply_mask(sample_arr):
    from rs_toolkit.preprocessing import apply_mask, nodata_mask
    m = nodata_mask(sample_arr)
    out = apply_mask(sample_arr, m)
    assert np.isnan(out[:, 0, 0]).all()

def test_stack_bands(tmp_tif, tmp_path):
    from rs_toolkit.io import read_raster, write_raster
    from rs_toolkit.preprocessing import stack_bands
    arr, prof = read_raster(str(tmp_tif), [1])
    p1 = tmp_path / "b1.tif"; write_raster(str(p1), arr, prof)
    p2 = tmp_path / "b2.tif"; write_raster(str(p2), arr, prof)
    stacked, _ = stack_bands([str(p1), str(p2)])
    assert stacked.shape[0] == 2

# ── Normalization ───────────────────────────────────────────────────────
def test_ndvi():
    assert np.isclose(ndvi(np.array([.6]), np.array([.2]))[0], .5)

def test_ndvi_zero_denom():
    assert np.isnan(ndvi(np.array([0.0]), np.array([0.0]))[0])

def test_minmax():
    m = minmax(np.random.rand(2, 8, 8).astype("f"))
    assert m.min() == 0 and np.isclose(m.max(), 1)

def test_zscore():
    a = np.random.rand(2, 32, 32).astype("f")
    z = zscore(a)
    assert np.allclose(np.nanmean(z, (1, 2)), 0, atol=1e-5)

def test_percentile_clip():
    a = np.arange(100, dtype="f").reshape(1, 10, 10)
    c = percentile_clip(a, lo=10, hi=90)
    assert c.min() >= np.percentile(a, 10) - 1

# ── Indices ─────────────────────────────────────────────────────────────
def test_all_indices():
    nir = np.random.rand(10, 10).astype("f") * 0.5 + 0.3
    red = np.random.rand(10, 10).astype("f") * 0.3
    grn = np.random.rand(10, 10).astype("f") * 0.3
    blu = np.random.rand(10, 10).astype("f") * 0.1
    swi = np.random.rand(10, 10).astype("f") * 0.3
    for fn, args in [(ndvi, (nir, red)), (ndwi, (grn, nir)), (ndbi, (swi, nir)),
                     (savi, (nir, red)), (evi, (nir, red, blu)),
                     (nbr, (nir, swi)), (ndsi, (grn, swi)), (bsi, (blu, red, nir, swi))]:
        r = fn(*args)
        assert r.shape == (10, 10)
        assert np.isfinite(r).any()

# ── Tiling ──────────────────────────────────────────────────────────────
def test_tiles_and_split():
    P, X = extract_patches(np.random.rand(3, 300, 300).astype("f"), 64, 16)
    assert P.shape[1:] == (3, 64, 64)
    tr, va, te = spatial_split(X, 64, block=128)
    assert len(set(tr) | set(va) | set(te)) == len(X)

def test_nan_rejection():
    """Patches that are mostly NaN should be rejected."""
    a = np.full((1, 64, 64), np.nan, dtype="f")
    P, X = extract_patches(a, 32, max_nan_frac=0.2)
    assert len(P) == 0

# ── Terrain ─────────────────────────────────────────────────────────────
def test_terrain_lidar():
    from rs_toolkit.terrain import slope, hillshade, fill_voids, aspect
    from rs_toolkit.optical import hist_equalize, to_gray
    y, x = np.mgrid[0:20, 0:20]; dem = (x * 1.0)
    assert np.allclose(slope(dem, 1.0), 45)
    a = aspect(dem, 1.0); assert a.shape == dem.shape
    dem[5, 5] = np.nan; assert not np.isnan(fill_voids(dem)).any()
    dem_f = fill_voids(dem)
    assert 0 <= hillshade(dem_f).min() and hillshade(dem_f).max() <= 1
    assert np.nanmax(hist_equalize(np.random.rand(10, 10))) <= 1
    assert to_gray(np.random.rand(3, 8, 8)).shape == (1, 8, 8)

# ── Visualization (smoke tests) ────────────────────────────────────────
def test_show_smoke(sample_arr):
    import matplotlib; matplotlib.use("Agg")
    from rs_toolkit.visualization import show, show_index, show_histogram
    show(sample_arr[:3])
    show(sample_arr[:1])  # single-band gray
    show_index(np.random.rand(32, 32) * 2 - 1, title="test")
    show_histogram(sample_arr[:3])

def test_show_split_map():
    import matplotlib; matplotlib.use("Agg")
    from rs_toolkit.visualization import show_split_map
    xy = np.random.randint(0, 512, (50, 2))
    splits = [np.arange(30), np.arange(30, 40), np.arange(40, 50)]
    show_split_map(xy, splits, 64)
