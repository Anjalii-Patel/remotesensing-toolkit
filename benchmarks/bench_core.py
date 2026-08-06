"""
Simple benchmark for core rs-toolkit operations.

Run:   python benchmarks/bench_core.py
"""
import time, numpy as np, os, sys

# make sure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rs_toolkit.sample_data import make_landsat_scene
from rs_toolkit.io import read_raster
from rs_toolkit.preprocessing import nodata_mask, apply_mask, bitmask
from rs_toolkit.normalization import minmax, zscore
from rs_toolkit.indices import ndvi, ndwi, savi, evi
from rs_toolkit.tiling import extract_patches, spatial_split

# ── config ──────────────────────────────────────────────────────────────
SCENE = "_bench_scene.tif"
HEIGHT, WIDTH = 2048, 2048
PATCH_SIZE = 256
OVERLAP = 32
N_RUNS = 5

def bench(name, fn, n=N_RUNS):
    """Time *fn* over *n* runs and print results."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"  {name:<30s}  {avg:8.1f} ms  ± {std:5.1f} ms")
    return avg

def main():
    print(f"Benchmark: {HEIGHT}×{WIDTH} scene, {PATCH_SIZE}px patches, {N_RUNS} runs each\n")

    # ── generate test scene ────────────────────────────────────────────
    scene_path, qa_path = make_landsat_scene(SCENE, HEIGHT, WIDTH, seed=0)
    print()

    # ── load once for array-based benchmarks ───────────────────────────
    arr, prof = read_raster(scene_path)

    # number of expected patches
    patches, coords = extract_patches(arr, PATCH_SIZE, OVERLAP)
    n_patches = len(patches)

    print(f"Scene shape: {arr.shape}")
    print(f"Patches:     {n_patches} × ({arr.shape[0]}, {PATCH_SIZE}, {PATCH_SIZE})")
    print(f"{'='*60}")

    # ── benchmarks ─────────────────────────────────────────────────────
    print("\nI/O")
    bench("GeoTIFF loading",        lambda: read_raster(scene_path))

    print("\nMasking")
    bench("nodata_mask",            lambda: nodata_mask(arr))
    qa, _ = read_raster(qa_path)
    qa_band = np.nan_to_num(qa[0], nan=0).astype(np.uint16)
    bench("bitmask (cloud QA)",     lambda: bitmask(qa_band, [4]))
    nd = nodata_mask(arr)
    bench("apply_mask",             lambda: apply_mask(arr, nd))

    print("\nNormalization")
    bench("minmax",                 lambda: minmax(arr))
    bench("zscore",                 lambda: zscore(arr))

    print("\nSpectral indices")
    nir, red, grn, blu = arr[3], arr[2], arr[1], arr[0]
    bench("NDVI",                   lambda: ndvi(nir, red))
    bench("NDWI",                   lambda: ndwi(grn, nir))
    bench("SAVI",                   lambda: savi(nir, red))
    bench("EVI",                    lambda: evi(nir, red, blu))

    print("\nTiling")
    bench("extract_patches (256px)", lambda: extract_patches(arr, PATCH_SIZE, OVERLAP))
    bench("spatial_split",           lambda: spatial_split(coords, PATCH_SIZE, block=1024))

    print(f"\n{'='*60}")
    print("Done.\n")

    # cleanup
    for f in [SCENE, SCENE.replace(".tif", "_QA.tif")]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
