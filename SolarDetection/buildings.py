import os

import geopandas as gpd
import pandas as pd
import numpy as np

import fiona
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from shapely.geometry import box
from shapely.geometry import shape

from shapely.errors import ShapelyError
from TreeDetection.config import Config


class ShapeCache:
    _instance = None

    @classmethod
    def initialize(cls, index_path, max_size=3):
        if cls._instance is None:
            cls._instance = cls(index_path, max_size)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("ShapeCache not initialized. Call ShapeCache.initialize() first.")
        return cls._instance

    def __init__(self, index_path, max_size=3):
        self.index = gpd.read_file(index_path)
        self.cache = OrderedDict()
        self.max_size = max_size
        self.crs = self.index.crs

    def _load_tile(self, filename):
        try:
            return gpd.read_file(filename)
        except Exception as e:
            print(f"[WARNING] Failed to read tile {filename}: {e}")
            return None

    def get_tiles(self, prediction_geom):
        if prediction_geom is None:
            return gpd.GeoDataFrame(columns=["geometry"], crs=self.crs)

        overlaps = self.index[self.index.intersects(prediction_geom)]
        loaded_tiles = []

        for _, row in overlaps.iterrows():
            tile_id = row["tile_id"]
            if tile_id in self.cache:
                
                print(f"[INFO] Avoided loading tile {tile_id} from disk, using cached version.")
                self.cache.move_to_end(tile_id)
                tile = self.cache[tile_id]
            else:
                tile = self._load_tile(row["filename"])
                if tile is not None:
                    self.cache[tile_id] = tile
                    if len(self.cache) > self.max_size:
                        self.cache.popitem(last=False)
            if tile is not None:
                loaded_tiles.append(tile)

        if loaded_tiles:
            return gpd.GeoDataFrame(pd.concat(loaded_tiles, ignore_index=True), crs=self.crs)
        else:
            return gpd.GeoDataFrame(columns=["geometry"], crs=self.crs)

    def clear(self):
        self.cache.clear()

    def clear(self):
        self.cache.clear()

def partition_geometries(file, output_dir, max_per_tile=10000, logger=None):
    if not logger:
        logger = Config().logger
    os.makedirs(output_dir, exist_ok=True)

    # First pass: scan bounding box and geometry count
    total_bounds = None
    count = 0
    with fiona.open(file, layer=0) as src:
        total = len(src)
        for i, feat in enumerate(src):
            if logger:
                current_percent = int(100 * (i + 1) / total)
                previous_percent = int(100 * i / total)
                if (current_percent // 5) != (previous_percent // 5) or current_percent == 100 or i == 0:
                    logger.info(f"Scanning Geometry {current_percent}% ({i+1}/{total})")

            geom = shape(feat["geometry"])            
            # Filter invalid geometries
            if not geom.is_valid:
                geom = geom.buffer(0)
                if not geom.is_valid:
                    if logger:
                        logger.warning(f"Skipping invalid geometry at index {i}")
                    continue

            # Optional: skip GeometryCollection if you don't want them
            if geom.geom_type == 'GeometryCollection':
                if logger:
                    logger.warning(f"Skipping GeometryCollection at index {i}")
                continue


            if total_bounds is None:
                total_bounds = list(geom.bounds)
            else:
                gminx, gminy, gmaxx, gmaxy = geom.bounds
                total_bounds[0] = min(total_bounds[0], gminx)
                total_bounds[1] = min(total_bounds[1], gminy)
                total_bounds[2] = max(total_bounds[2], gmaxx)
                total_bounds[3] = max(total_bounds[3], gmaxy)
            count += 1
        crs = src.crs

    minx, miny, maxx, maxy = total_bounds
    num_tiles_est = count // max_per_tile + 1
    num_div = int(np.sqrt(num_tiles_est)) + 1

    x_bins = np.linspace(minx, maxx, num_div + 1)
    y_bins = np.linspace(miny, maxy, num_div + 1)

    tile_buffers = defaultdict(list)
    tile_bounds = {}

    # Second pass: process geometries
    with fiona.open(file, layer=0) as src:
        total = len(src)
        for i, feat in enumerate(src):
            if logger:
                current_percent = int(100 * (i + 1) / total)
                previous_percent = int(100 * i / total)
                if (current_percent // 5) != (previous_percent // 5) or current_percent == 100 or i == 0:
                    logger.info(f"Processing Geometry {current_percent}% ({i+1}/{total})")

            geom = shape(feat["geometry"])   
            # Filter invalid geometries
            if not geom.is_valid:
                geom = geom.buffer(0)
                if not geom.is_valid:
                    if logger:
                        logger.warning(f"Skipping invalid geometry at index {i}")
                    continue

            # Optional: skip GeometryCollection if you don't want them
            if geom.geom_type == 'GeometryCollection':
                if logger:
                    logger.warning(f"Skipping GeometryCollection at index {i}")
                continue
            centroid = geom.centroid
            props = feat["properties"]

            x_idx = np.digitize([centroid.x], x_bins)[0] - 1
            y_idx = np.digitize([centroid.y], y_bins)[0] - 1
            tile_id = f"{x_idx}_{y_idx}"
            tile_buffers[tile_id].append({**props, "geometry": geom})

            if len(tile_buffers[tile_id]) >= max_per_tile:
                _flush_tile(tile_buffers, tile_id, output_dir, crs, tile_bounds)

    # Final flush
    keys = list(tile_buffers.keys())
    total_flush = len(keys)
    for i, tile_id in enumerate(keys):
        if logger:
            current_percent = int(100 * (i + 1) / total_flush)
            previous_percent = int(100 * i / total_flush)
            if (current_percent // 5) != (previous_percent // 5) or current_percent == 100 or i == 0:
                logger.info(f"Flushing Tile {current_percent}% ({i+1}/{total_flush})")
        _flush_tile(tile_buffers, tile_id, output_dir, crs, tile_bounds)

    # Build tile index
    index_records = []
    for tile_id, bounds in tile_bounds.items():
        index_records.append({
            "tile_id": tile_id,
            "filename": os.path.join(output_dir, f"tile_{tile_id}.gpkg"),
            "geometry": box(*bounds)
        })

    index_gdf = gpd.GeoDataFrame(index_records, crs=crs)
    index_gdf.to_file(os.path.join(output_dir, "tile_index.gpkg"), driver="GPKG")
    return index_gdf

def _flush_tile(tile_buffers, tile_id, output_dir, crs, tile_bounds):
    from shapely.ops import unary_union

    records = tile_buffers[tile_id]
    gdf = gpd.GeoDataFrame(records, crs=crs)
    path = os.path.join(output_dir, f"tile_{tile_id}.gpkg")

    if os.path.exists(path):
        # Append mode not supported by GPKG → concatenate in memory
        existing = gpd.read_file(path)
        gdf = pd.concat([existing, gdf], ignore_index=True)
    
    gdf.to_file(path, driver="GPKG")
    
    bounds = gdf.total_bounds
    if tile_id not in tile_bounds:
        tile_bounds[tile_id] = bounds
    else:
        # Expand bounding box
        old = tile_bounds[tile_id]
        tile_bounds[tile_id] = [
            min(old[0], bounds[0]), min(old[1], bounds[1]),
            max(old[2], bounds[2]), max(old[3], bounds[3])
        ]
    
    tile_buffers[tile_id].clear()

def find_roof_solar(features, file, bounds):
    """
    Check if feature geometries intersect preprocessed building tiles
    using bounding-box acceleration followed by precise geometry checks.
    """
    config = Config()
    index_dir = os.path.join(config.output_directory, 'building_index')
    os.makedirs(index_dir, exist_ok=True)

    index_path = os.path.join(index_dir, "tile_index.gpkg")

    # Ensure preprocessing is done
    try:
        if not os.path.exists(index_path):
            #print("Preprocessed building chunks not found. Running preprocessing, this may take a while ...")
            partition_geometries(file, index_dir, max_per_tile=10000)
    except Exception as e:
        print(f"[ERROR] Failed to partition geometries: {e}")
        return

    # Load tile cache
    try:
        cache = ShapeCache.get_instance()
        building_shapes = cache.get_tiles(box(*bounds))
        building_shapes = building_shapes[building_shapes.is_valid & building_shapes.geometry.notnull()]
    except Exception as e:
        print(f"[ERROR] Failed to load tiles for prediction: {e}")
        return

    #total = len(features)
    for i, feature in enumerate(features):
        #current_percent = int(100 * (i + 1) / total)
        #previous_percent = int(100 * i / total)
        #if i == 0 or (current_percent // 5) != (previous_percent // 5) or current_percent == 100:
        #    print(f"Checking feature {i + 1}/{total} ({current_percent}%)")

        try:
            feature_geom = shape(feature['geometry'])

            # Fast check: bounding box
            feature_bounds = feature_geom.bounds
            bbox_candidates = building_shapes[building_shapes.intersects(box(*feature_bounds))]

            # Precise check: actual geometry
            matched = any(b.geometry.intersects(feature_geom) for _, b in bbox_candidates.iterrows())


            feature['properties']['In_building'] = int(matched)

        except (ShapelyError, ValueError, TypeError) as e:
            feature['properties']['In_building'] = -1  # Mark as error

    return features
     
def load_tiles_for_prediction(index_path, prediction_geom):
    try:
        index_gdf = gpd.read_file(index_path)
    except Exception as e:
        raise RuntimeError(f"Could not read index file '{index_path}': {e}")
    if prediction_geom is None:
        return gpd.GeoDataFrame(columns=["geometry"], crs=index_gdf.crs)
    overlapping = index_gdf[index_gdf.intersects(prediction_geom)]
    dfs = []

    for _, row in overlapping.iterrows():
        try:
            tile = gpd.read_file(row["filename"])
            dfs.append(tile)
        except Exception as e:
            print(f"[WARNING] Failed to read tile {row['filename']}: {e}")

    if dfs:
        return gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=index_gdf.crs)
    else:
        #print("[INFO] No valid tiles loaded.")
        return gpd.GeoDataFrame(columns=["geometry"], crs=index_gdf.crs)