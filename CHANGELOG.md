# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2026-08-05

### Added
- **io**: GeoTIFF read/write with nodata→NaN conversion
- **preprocessing**: nodata mask, QA bitmask, apply_mask, band stacking, CRS reprojection
- **normalization**: min-max, z-score, percentile clip (all per-band, NaN-aware)
- **tiling**: grid tiling, overlapping patch extraction, spatial train/val/test split
- **indices**: NDVI, NDWI, NDBI, SAVI, EVI, NBR, NDSI, BSI with INDEX_REGISTRY
- **terrain**: slope, aspect, hillshade, iterative NaN void filling
- **optical**: grayscale conversion, robust stretch, histogram equalize, gradient magnitude
- **visualization**: raster display, index colormap, band histograms, patch grid, split scatter map
- **stats**: per-band statistics, coverage report, inter-band correlation matrix
- **sample_data**: synthetic DEM, multispectral, and QA generators for testing
- **cli**: `rs-tool` with `tile`, `index`, `reproject`, `derive` subcommands
- **ci**: GitHub Actions test matrix (Python 3.9–3.12)
- Test suite with 17+ tests covering all modules
- Example demo script and Jupyter notebook
