"""More real environments: USGS 3DEP aerial LiDAR from many US cities.

    python -m ig3d.cities search                 # find tiles per city, write a manifest
    python -m ig3d.cities download               # fetch them (resumable)
    python -m ig3d.cities build                  # voxelise into 48^3 blocks
    python -m ig3d.cities summary                # blocks per city and split

Split. Whole cities are held out, never tiles of a training city, so every number on a
held-out city is a cross-city result. Austin (the paper's tile) stays held out. The
held-out cities are what the Gazebo/PX4 flights (`ig3d.sim`) fly in.

Same 2.5D voxelisation as the paper (`ig3d.lidar`): each column is solid up to its highest
return. Training cities use half-overlapping blocks (stride 24) for more data; held-out
cities use the paper's non-overlapping blocks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .lidar import SIZE, VOXEL_M, blocks_from_surface, save_blocks, surface
from .paths import CACHE
from .volumes import Volume

ROOT = CACHE / "lidar" / "cities"
TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"

# name: (min_lon, min_lat, max_lon, max_lat), split, kind. Boxes are small (~1 km) and sit
# on each downtown, plus one forest and one suburb so the data is not all high-rises.
CITIES = {
    "nyc_midtown":       ((-73.990, 40.750, -73.975, 40.762), "train", "urban"),
    "chicago_loop":      ((-87.636, 41.876, -87.622, 41.886), "train", "urban"),
    "sf_downtown":       ((-122.405, 37.787, -122.392, 37.797), "train", "urban"),
    "boston_downtown":   ((-71.062, 42.352, -71.050, 42.360), "train", "urban"),
    "denver_downtown":   ((-104.998, 39.742, -104.985, 39.752), "train", "urban"),
    "atlanta_midtown":   ((-84.392, 33.770, -84.380, 33.782), "train", "urban"),
    "dc_downtown":       ((-77.040, 38.895, -77.025, 38.905), "train", "urban"),
    "houston_downtown":  ((-95.372, 29.752, -95.358, 29.764), "train", "urban"),
    "la_downtown":       ((-118.260, 34.045, -118.246, 34.056), "train", "urban"),
    "portland_forest":   ((-122.770, 45.545, -122.755, 45.555), "train", "forest"),
    "phoenix_suburb":    ((-111.940, 33.500, -111.925, 33.510), "train", "suburban"),
    # held out: cross-city test + Gazebo/PX4 flights
    "seattle_downtown":  ((-122.342, 47.603, -122.328, 47.613), "heldout", "urban"),
    "miami_brickell":    ((-80.198, 25.758, -80.188, 25.768), "heldout", "urban"),
}
# Austin is the paper's tile, already in datasets/cache/lidar, and is also held out.
HELDOUT = tuple(["austin"] + [c for c, v in CITIES.items() if v[1] == "heldout"])
TRAIN = tuple(c for c, v in CITIES.items() if v[1] == "train")
TRAIN_STRIDE, HELDOUT_STRIDE = SIZE // 2, SIZE


def city_dir(city: str) -> Path:
    return ROOT / city


def blocks_file(city: str, voxel_m: float = VOXEL_M) -> Path:
    return city_dir(city) / "blocks_s{:g}.npz".format(voxel_m)


def resolve(names) -> list[str]:
    """Expand the aliases 'train', 'heldout' and 'all' into city names."""
    out = []
    for n in names:
        if n == "train":
            out += list(TRAIN)
        elif n == "heldout":
            out += [c for c in HELDOUT if c != "austin"]
        elif n == "all":
            out += list(CITIES)
        elif n in CITIES or n == "austin":
            out.append(n)
        else:
            raise SystemExit("unknown city {!r}; known: {}".format(n, ", ".join(CITIES)))
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------------------ search

def _query(bbox, max_items=100):
    q = urllib.parse.urlencode({
        "bbox": ",".join(str(v) for v in bbox),
        "datasets": "Lidar Point Cloud (LPC)",
        "prodFormats": "LAZ",
        "max": str(max_items),
        "outputFormat": "JSON",
    })
    req = urllib.request.Request(TNM_API + "?" + q, headers={"User-Agent": "ig3d-lidar/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["items"]


def _project(url: str) -> str:
    parts = url.split("/Projects/")
    return parts[1].split("/")[0] if len(parts) > 1 else "unknown"


def pick_tiles(items, bbox, max_tiles):
    """Tiles of the newest project covering the box, nearest the box centre first."""
    items = [it for it in items if it.get("downloadURL", "").lower().endswith(".laz")]
    if not items:
        return []
    newest = max(items, key=lambda it: it.get("publicationDate") or "")
    proj = _project(newest["downloadURL"])
    same = [it for it in items if _project(it["downloadURL"]) == proj]
    cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])

    def dist(it):
        b = it["boundingBox"]
        return math.hypot(0.5 * (b["minX"] + b["maxX"]) - cx, 0.5 * (b["minY"] + b["maxY"]) - cy)

    return sorted(same, key=dist)[:max_tiles]


def search(cities, max_tiles=3):
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for c in cities:
        if c == "austin":
            continue
        bbox, split, kind = CITIES[c]
        try:
            tiles = pick_tiles(_query(bbox), bbox, max_tiles)
        except Exception as e:  # noqa: BLE001
            print("{:18s} search failed: {}".format(c, e))
            continue
        manifest[c] = dict(bbox=bbox, split=split, kind=kind, tiles=[
            dict(title=t["title"], url=t["downloadURL"], bytes=t.get("sizeInBytes"),
                 published=t.get("publicationDate"), project=_project(t["downloadURL"]))
            for t in tiles])
        gb = sum(t.get("sizeInBytes") or 0 for t in tiles) / 1e9
        proj = _project(tiles[0]["downloadURL"]) if tiles else "-"
        print("{:18s} {:8s} {} tile(s), {:.2f} GB, project {}".format(
            c, split, len(tiles), gb, proj))
        time.sleep(0.5)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print("wrote " + str(manifest_path))
    return manifest


# ---------------------------------------------------------------------------- download

def _wait_s(err, attempt):
    """Seconds to wait before retrying; rockyweb rate-limits (HTTP 429) aggressively."""
    import urllib.error
    if isinstance(err, urllib.error.HTTPError) and err.code in (429, 503):
        ra = err.headers.get("Retry-After") if err.headers else None
        if ra and ra.strip().isdigit():
            return min(900, int(ra) + 5)
        return min(600, 30 * 2 ** attempt)
    return 5 * (attempt + 1)


def _download(url: str, dest: Path, retries: int = 8):
    """Resumable download to dest (via dest.part)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "ig3d-lidar/1.0",
                                                   **({"Range": "bytes=%d-" % have} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(part, "ab" if have else "wb") as fh:
                if have and r.status != 206:
                    fh.truncate(0)
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
            part.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            w = _wait_s(e, attempt)
            print("  retry {} for {} in {}s: {}".format(attempt + 1, dest.name, w, e), flush=True)
            time.sleep(w)
    raise RuntimeError("download failed (rerun `download` later; it resumes): " + url)


def download(cities):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for c in cities:
        if c == "austin":
            continue
        if c not in manifest:
            print("{}: not in manifest; run `search` first".format(c))
            continue
        for t in manifest[c]["tiles"]:
            dest = city_dir(c) / "laz" / t["url"].rsplit("/", 1)[1]
            if dest.exists():
                continue
            print("{:18s} {} ({:.0f} MB)".format(c, dest.name, (t["bytes"] or 0) / 1e6), flush=True)
            try:
                _download(t["url"], dest)
            except RuntimeError as e:
                print("  SKIPPED: {}".format(e), flush=True)
            time.sleep(3)


# ------------------------------------------------------------------------------- build

def build(cities, voxel_m=VOXEL_M, seed=0):
    for c in cities:
        if c == "austin":
            continue
        tiles = sorted((city_dir(c) / "laz").glob("*.laz"))
        if not tiles:
            print("{}: no tiles downloaded".format(c))
            continue
        stride = HELDOUT_STRIDE if CITIES[c][1] == "heldout" else TRAIN_STRIDE
        vols, info, metas = [], [], []
        for ti, path in enumerate(tiles):
            top, covered, meta = surface(path, voxel_m)
            v, inf = blocks_from_surface(top, covered, voxel_m, seed, stride, tag=ti + 1)
            for r in inf:
                r["tile"] = path.name
                if r.get("used"):
                    r["index"] += len(vols)
            vols += v
            info += inf
            metas.append(meta)
            print("{:18s} {}: grid {}, unit scale {}, {} blocks".format(
                c, path.name, meta["grid"], meta["unit_scale"], len(v)), flush=True)
        if not vols:
            print("{}: no usable blocks".format(c))
            continue
        save_blocks(blocks_file(c, voxel_m), vols,
                    dict(city=c, split=CITIES[c][1], kind=CITIES[c][2], stride=stride, tiles=metas),
                    info)
        print("{:18s} -> {} blocks in {}".format(c, len(vols), blocks_file(c, voxel_m)), flush=True)


def load_city_blocks(city: str, voxel_m: float = VOXEL_M) -> list[Volume]:
    path = blocks_file(city, voxel_m)
    if not path.exists():
        raise SystemExit("{} not built; run `python -m ig3d.cities build --cities {}`".format(
            path, city))
    z = np.load(path)
    return [Volume(occ=o, start=tuple(int(a) for a in s), goal=tuple(int(a) for a in g))
            for o, s, g in zip(z["occ"], z["start"], z["goal"])]


def available(names) -> list[str]:
    """The cities among `names` whose blocks are built (Austin always is)."""
    out = []
    for c in resolve(names):
        if c == "austin" or blocks_file(c).exists():
            out.append(c)
    return out


def summary():
    print("{:18s} {:8s} {:9s} {:>7s}".format("city", "split", "kind", "blocks"))
    tot = {"train": 0, "heldout": 0}
    for c in ["austin"] + list(CITIES):
        split, kind = ("heldout", "urban") if c == "austin" else CITIES[c][1:]
        try:
            from .lidar import load_blocks
            n = len(load_blocks(city=c))
        except (SystemExit, FileNotFoundError):
            n = 0
        tot[split] += n
        print("{:18s} {:8s} {:9s} {:7d}".format(c, split, kind, n))
    print("total: {} training blocks, {} held-out blocks".format(tot["train"], tot["heldout"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["search", "download", "build", "summary"])
    ap.add_argument("--cities", nargs="*", default=["all"],
                    help="names, or the aliases train / heldout / all")
    ap.add_argument("--max-tiles", type=int, default=3, help="tiles per city (search)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cities = resolve(a.cities)
    if a.what == "search":
        search(cities, a.max_tiles)
    elif a.what == "download":
        download(cities)
    elif a.what == "build":
        build(cities)
    else:
        summary()


if __name__ == "__main__":
    main()
