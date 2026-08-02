# rs-toolkit

Preprocessing utilities for multimodal remote sensing (optical, SAR, LiDAR intensity, DEM).

## Quick start

```bash
pip install -e .[dev]
```

## CLI

```bash
# Tile a GeoTIFF into 256×256 patches with 32 px overlap
rs-tool tile image.tif --size 256 --overlap 32 --out tiles/

# Compute a spectral index and write to GeoTIFF
rs-tool index ndvi image.tif --red 3 --nir 4 --out ndvi.tif

# Reproject to a different CRS
rs-tool reproject image.tif EPSG:4326 --out reproj.tif

# Derive slope + hillshade from a DEM
rs-tool derive dem.tif --out dem_derived.tif
```

## Modules

| Module | Purpose |
|---|---|
| `io` | GeoTIFF read / write (nodata → NaN, profile round-trip) |
| `preprocessing` | Cloud / nodata masking, QA bitmask, band stacking, reprojection |
| `normalization` | Min-max, z-score, percentile clip per-band |
| `tiling` | Grid tiling, patch extraction, **spatial train/val/test split** |
| `indices` | NDVI, NDWI, NDBI, SAVI, EVI, NBR, NDSI, BSI |
| `terrain` | Slope, aspect, hillshade, void filling |
| `optical` | Grayscale conversion, robust stretch, histogram equalize, gradient magnitude |
| `visualization` | Raster display, index colormap, band histograms, patch grid, split map |
| `cli` | `rs-tool` command-line entry point |

## Spectral indices

```
NDVI = (NIR − Red) / (NIR + Red)
NDWI = (Green − NIR) / (Green + NIR)
NDBI = (SWIR − NIR) / (SWIR + NIR)
SAVI = ((NIR − Red) / (NIR + Red + L)) × (1 + L)
EVI  = G × (NIR − Red) / (NIR + C₁·Red − C₂·Blue + L)
NBR  = (NIR − SWIR) / (NIR + SWIR)
NDSI = (Green − SWIR) / (Green + SWIR)
BSI  = ((SWIR + Red) − (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
```

## Python usage

```python
from rs_toolkit.io import read_raster, write_raster
from rs_toolkit.preprocessing import nodata_mask, apply_mask, reproject_raster
from rs_toolkit.normalization import minmax
from rs_toolkit.tiling import extract_patches, spatial_split
from rs_toolkit.indices import ndvi
from rs_toolkit.visualization import show, show_index

# Load a Landsat scene
arr, prof = read_raster("LC08_B432.tif")  # (C, H, W) float32

# Mask nodata
mask = nodata_mask(arr)
arr = apply_mask(arr, mask)

# Compute NDVI  (band 4 = NIR, band 3 = Red for Landsat-8)
vi = ndvi(arr[3], arr[2])

# Normalize and tile
arr_n = minmax(arr)
patches, coords = extract_patches(arr_n, size=256, overlap=32)
train_idx, val_idx, test_idx = spatial_split(coords, 256, block=1024)

# Visualise
show(arr[:3], title="RGB")
show_index(vi, title="NDVI")
```

## Terrain from DEM / LiDAR

```python
from rs_toolkit.terrain import slope, hillshade, fill_voids

dem, prof = read_raster("dtm.tif", [1])
dem = fill_voids(dem[0])
s = slope(dem, res=prof["transform"].a)
hs = hillshade(dem, res=prof["transform"].a)
```

## Tests

```bash
pytest -v
```

## Project layout

```
remote-sensing-toolkit/
├── src/rs_toolkit/
│   ├── io.py               # raster I/O
│   ├── preprocessing.py     # masks, stacking, reprojection
│   ├── normalization.py     # per-band normalizers
│   ├── tiling.py            # grid tiles, patches, spatial split
│   ├── indices.py           # NDVI, NDWI, NDBI, SAVI, EVI, …
│   ├── terrain.py           # slope, aspect, hillshade, void fill
│   ├── optical.py           # grayscale, stretch, edges
│   ├── visualization.py     # display helpers
│   └── cli.py               # rs-tool CLI
├── tests/
│   ├── conftest.py          # synthetic fixtures
│   └── test_toolkit.py      # unit tests
├── examples/
│   └── demo.py              # synthetic end-to-end demo
├── notebooks/
├── README.md
└── pyproject.toml
```
