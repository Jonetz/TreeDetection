import asyncio
import json
import os
import aiofiles
import geopandas as gpd
import numpy as np
import rasterio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import box
from rasterio.mask import geometry_window
import cupy as cp

"""
Tiling orthomosaic data.

Adapted from detectree2.preprocessing.tiling.tile_data function. (Released under MIT License, Version 1.2.x)
"""

def get_features(gdf: gpd.GeoDataFrame):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them."""
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


async def write_metadata_async(meta_name, metadata):
    """Asynchronously write metadata to a JSON file."""
    async with aiofiles.open(meta_name, "w") as meta_file:
        await meta_file.write(json.dumps(metadata))

async def tile_single_file(
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

        for minx in np.arange(data.bounds[0], data.bounds[2], tile_width):
            for miny in np.arange(data.bounds[1], data.bounds[3], tile_height):
                out_path_root = out_path / f"{tilename}_{int(minx)}_{int(miny)}_{int(tile_width)}_{int(buffer)}_{crs}"

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

                # Adjust bbox to align with the pixel grid
                try:
                    window = geometry_window(data, coords, pad_x=0, pad_y=0)
                    out_transform = data.window_transform(window)
                except Exception:
                    ValueError('Input shapes do not overlap raster, check geometry of incoming Tifs.')

                # Metadata is written asynchronously
                meta_name = out_path_root.with_suffix(".json")
                metadata = {
                    "crs": crs,
                    "transform": out_transform,
                    "bounds": [
                        minx - buffer,
                        miny - buffer,
                        minx + tile_width + buffer,
                        miny + tile_height + buffer,
                    ],
                    "only_forest": only_forest,
                    "only_urban": only_urban,
                }
                await write_metadata_async(meta_name, metadata)

async def tile_data(
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

    async def process_file(data_path):
        img_out_dir = os.path.join(out_dir, Path(data_path).stem)
        try:
            await tile_single_file(
                data_path=data_path,
                out_dir=img_out_dir,
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
        # Choose executor type based on use_threadpool
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, process_file, data_path)
                for data_path in file_list
            ]
            await asyncio.gather(*tasks)
    else:
        for data_path in file_list:
            await process_file(data_path)

if __name__ == "__main__":
    # Example usage
    file_list = ["file1.tif", "file2.tif", "file3.tif"]
    output_directory = "./output_tiles"
    tile_data(file_list, output_directory, buffer=30, tile_width=200, tile_height=200, max_workers=4)
