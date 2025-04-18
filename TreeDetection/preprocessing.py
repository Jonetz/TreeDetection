import asyncio
import json
import os
import aiofiles
import geopandas as gpd
import numpy as np
import rasterio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import box
from rasterio.mask import geometry_window
import cupy as cp
import yaml

"""
Tiling orthomosaic data.

Adapted from detectree2.preprocessing.tiling.tile_data function. (Released under MIT License, Version 1.2.x)
"""

def get_features(gdf: gpd.GeoDataFrame):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them."""
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]

def write_metadata(meta_name, metadata):
    """write metadata to a JSON file."""
    with open(meta_name, "w") as meta_file:
        meta_file.write(json.dumps(metadata))   
        
def tile_single_file(
    data_path: str,
    out_dir: str,
    buffer: int = 0,
    tile_width: int = 50,
    tile_height: int = 50,
    forest_bounds_gpu: cp.ndarray = None,  # Pass preloaded GPU data
    forest_regions: gpd.GeoDataFrame = None,  # Keep CPU data for detailed checks
    logger=None,
):
    """Tiling a single raster file with GPU-accelerated bounding box checks."""
    if not os.path.exists(data_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(data_path) as data:
        out_path = Path(out_dir)
        crs = data.crs.to_epsg()

        tilename = Path(data.name).stem
        transform = data.transform

        tile_metadata_dict = {}

        for minx in np.arange(data.bounds[0], data.bounds[2], tile_width):
            for miny in np.arange(data.bounds[1], data.bounds[3], tile_height):
                tile_id = f"{tilename}_{int(minx)}_{int(miny)}_{int(tile_width)}_{int(buffer)}_{crs}"
                bbox = box(
                    minx - buffer,
                    miny - buffer,
                    minx + tile_width + buffer,
                    miny + tile_height + buffer,
                )

                only_forest, only_urban = False, False

                if forest_bounds_gpu is not None:
                    # GPU-based bounding box intersection checks
                    tile_bounds_gpu = cp.array([minx, miny, minx + tile_width, miny + tile_height])
                    overlaps = (
                        (forest_bounds_gpu[:, 2] > tile_bounds_gpu[0]) &  # forest max_x > tile min_x
                        (forest_bounds_gpu[:, 0] < tile_bounds_gpu[2]) &  # forest min_x < tile max_x
                        (forest_bounds_gpu[:, 3] > tile_bounds_gpu[1]) &  # forest max_y > tile min_y
                        (forest_bounds_gpu[:, 1] < tile_bounds_gpu[3])    # forest min_y < tile max_y
                    )

                    overlap_indices = cp.where(overlaps)[0]
                    if overlap_indices.size > 0:
                        
                        # Transfer candidate bounding boxes back to CPU for precise checks
                        candidate_indices = cp.asnumpy(overlap_indices)
                        candidates = forest_regions.iloc[candidate_indices]

                        # Create a union of all intersecting bounding boxes
                        intersecting = candidates[candidates.intersects(bbox)]
                        if not intersecting.empty:
                            # Check if the union of the intersecting geometries covers the bbox
                            union_bbox = intersecting.unary_union
                            if union_bbox.contains(bbox):  # Replace area check with spatial containment
                                only_forest = True  # Complete coverage
                        else:
                            only_urban = True
                    else:
                        only_urban = True

                geo = gpd.GeoDataFrame({"geometry": [bbox]}, index=[0], crs=data.crs)
                coords = get_features(geo)

                try:
                    window = geometry_window(data, coords, pad_x=0, pad_y=0)
                    out_transform = data.window_transform(window)
                except Exception:
                    raise ValueError('Input shapes do not overlap raster, check geometry of incoming Tifs.')

                # Collect metadata for this tile
                metadata = {
                    "crs": crs,
                    "transform": out_transform,  # serializable
                    "bounds": [
                        minx - buffer,
                        miny - buffer,
                        minx + tile_width + buffer,
                        miny + tile_height + buffer,
                    ],
                    "only_forest": only_forest,
                    "only_urban": only_urban,
                }
                tile_metadata_dict[tile_id] = metadata
                
        out_file = Path(out_dir) / f"{tilename}.json"
        write_metadata(out_file, tile_metadata_dict)

def tile_data(
    file_list: list,
    out_dir: str,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    parallel: bool = False,
    max_workers: int = 4,
    forest_shapefile: str = None,
    logger=None,
):
    """Tiling multiple raster files with GPU-accelerated bounding box checks."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # --- Load recovery info if available ---
    file_list, recovered_processed = load_recovery_data(file_list, buffer, tile_width, tile_height, logger, out_dir, os.path.join(out_dir, "recovery.yaml"))
    if not file_list:
        if logger:
            logger.info("All files have already been processed. Exiting Tiling.")
        else:
            print("All files have already been processed. Exiting Tiling.")
        return
    
    forest_regions = None
    forest_bounds_gpu = None

    # Preprocess forest shapefile if provided
    if forest_shapefile:
        # Load the forest shapefile into a GeoDataFrame, ensuring only valid geometries are included
        forest_regions = gpd.read_file(forest_shapefile, columns='geometry')
        forest_regions = forest_regions[forest_regions.geometry.notnull()]
        with rasterio.open(file_list[0]) as src:
            forest_regions = forest_regions.to_crs(crs=src.crs)
        if forest_regions.empty:
            raise ValueError(f"No valid geometries found in the forest shapefile {forest_shapefile}.")
        forest_bounds = np.array([list(geom.bounds) for geom in forest_regions.geometry])
        if forest_bounds.size == 0:
            raise ValueError(f"No valid bounding boxes found in the forest shapefile {forest_shapefile}.")
        forest_bounds_gpu = cp.array(forest_bounds)

    def process_file(data_path):
        try:
            tile_single_file(
                data_path=data_path,
                out_dir=out_dir,
                buffer=buffer,
                tile_width=tile_width,
                tile_height=tile_height,
                forest_bounds_gpu=forest_bounds_gpu,
                forest_regions=forest_regions,
                logger=logger,
            )
        except Exception as e:
            if logger:
                logger.error(f"Error processing file: {e}")
            else:
                print(f"Error processing file: {e}")

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_file, data_path)
                for data_path in file_list
            ]

            for i, future in enumerate(as_completed(futures)):
                try:
                    if logger and i % 100 == 0:
                        logger.info(f"Tiling file {i + 1}/{len(file_list)} ({int((i + 1)/len(file_list))} % )")
                    future.result()  # Raises exception if one occurred
                except Exception as e:
                    if logger:
                        logger.error(f"Error in thread execution: {e}")
                    else:
                        print(f"Error in thread execution: {e}")
    else:
        for i, data_path in enumerate(file_list):
            if logger and i % 100 == 0:
                logger.info(f"Tiling file {i + 1}/{len(file_list)} ({int((i + 1)/len(file_list))} % )")
            process_file(data_path)
            
    save_recovery_data(file_list, buffer, tile_width, tile_height, logger, recovered_processed, file_list, os.path.join(out_dir, "recovery.yaml"))

def load_recovery_data(file_list, buffer, tile_width, tile_height, logger, out_dir, recovery_file):    
    recovered_processed = set()
    skip_count = 0
    if os.path.exists(recovery_file):
        try:
            with open(recovery_file, "r") as f:
                recovery_data = yaml.safe_load(f)
            same_params = (
                recovery_data.get("buffer") == buffer and
                recovery_data.get("tile_width") == tile_width and
                recovery_data.get("tile_height") == tile_height
            )
            if same_params:
                recovered_processed = set(recovery_data.get("processed_files", []))
                original_len = len(file_list)
                
                for f in file_list:
                    if f in recovered_processed:
                        Path(out_dir).mkdir(parents=True, exist_ok=True)
                        tile_file = Path(out_dir) / f"{Path(f).stem}.json"
                        if not os.path.exists(tile_file):
                            recovered_processed.remove(f)                
                file_list = [f for f in file_list if f not in recovered_processed]
                skip_count = original_len - len(file_list)
                if logger and skip_count > 0:
                    logger.debug(f"Recovered {skip_count} already-processed files. Skipping them.")
                else:
                    print(f"Recovered {skip_count} already-processed files. Skipping them.")
        except Exception as e:
            if logger:
                logger.warning(f"Could not load recovery file: {e}")
            else:
                print(f"Could not load recovery file: {e}")
    return file_list,recovered_processed
            
def save_recovery_data(file_list, buffer, tile_width, tile_height, logger, recovered_processed, processed_files, recovery_file):
    try:
        with open(recovery_file, "w") as f:
            with open(recovery_file, "w") as f:
                yaml.safe_dump({
                    "buffer": buffer,
                    "tile_width": tile_width,
                    "tile_height": tile_height,
                    "file_list": file_list + list(recovered_processed),
                    "processed_files": processed_files + list(recovered_processed)
                }, f, sort_keys=False)
        if logger:
            logger.debug(f"Saved recovery file with {len(processed_files)} processed files.")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save recovery file: {e}")
        else:
            print(f"Failed to save recovery file: {e}")
            
if __name__ == "__main__":
    # Example usage
    file_list = ["file1.tif", "file2.tif", "file3.tif"]
    output_directory = "./output_tiles"
    tile_data(file_list, output_directory, buffer=30, tile_width=200, tile_height=200, max_workers=4)
