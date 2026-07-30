import argparse, os, rasterio
from rasterio.windows import Window
from .io import read_raster, write_raster
from .tiling import tile_grid
from .preprocessing import reproject_raster
from . import indices, terrain

def tile(a):
    os.makedirs(a.out, exist_ok=True)
    with rasterio.open(a.image) as ds:
        for r, c in tile_grid(ds.height, ds.width, a.size, a.overlap):
            w = Window(c, r, a.size, a.size)
            p = ds.profile.copy()
            p.update(height=a.size, width=a.size, transform=ds.window_transform(w))
            with rasterio.open(f"{a.out}/tile_{r}_{c}.tif", "w", **p) as d:
                d.write(ds.read(window=w))

def index(a):
    arr, prof = read_raster(a.image)
    b = lambda i: arr[i - 1]
    x = {"ndvi": lambda: indices.ndvi(b(a.nir), b(a.red)),
         "ndwi": lambda: indices.ndwi(b(a.green), b(a.nir)),
         "ndbi": lambda: indices.ndbi(b(a.swir), b(a.nir))}[a.name]()
    write_raster(a.out, x[None], prof)

def derive(a):
    arr, prof = read_raster(a.dem, [1])
    res = (abs(prof["transform"].e), abs(prof["transform"].a))
    d = terrain.fill_voids(arr[0])
    write_raster(a.out, __import__("numpy").stack([terrain.slope(d, res), terrain.hillshade(d, res)]), prof)

def main():
    p = argparse.ArgumentParser(prog="rs-tool"); s = p.add_subparsers(required=True)
    t = s.add_parser("tile"); t.add_argument("image"); t.add_argument("--size", type=int, default=256)
    t.add_argument("--overlap", type=int, default=0); t.add_argument("--out", default="tiles"); t.set_defaults(f=tile)
    i = s.add_parser("index"); i.add_argument("name", choices=["ndvi", "ndwi", "ndbi"]); i.add_argument("image")
    i.add_argument("--out", default="index.tif")
    for k, d in dict(red=3, green=2, nir=4, swir=5).items(): i.add_argument(f"--{k}", type=int, default=d)
    i.set_defaults(f=index)
    r = s.add_parser("reproject"); r.add_argument("image"); r.add_argument("crs"); r.add_argument("--out", default="reproj.tif")
    r.set_defaults(f=lambda a: reproject_raster(a.image, a.out, a.crs))
    d = s.add_parser("derive"); d.add_argument("dem"); d.add_argument("--out", default="dem_derived.tif")
    d.set_defaults(f=derive)
    a = p.parse_args(); a.f(a)
