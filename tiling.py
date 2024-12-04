import asyncio
import json
import os
import aiofiles
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import box
from rasterio.mask import mask
import shutil

from helpers import delete_contents
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
    dtype_bool: bool = False,
    forest_shapefile: str = None,
    logger=None,
):
    """Tiling a single raster file and saving output tiles, with forest/urban flagging."""
    if not os.path.exists(data_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load forest regions if provided
    forest_regions = None
    if forest_shapefile:
        forest_regions = gpd.read_file(forest_shapefile).to_crs(crs=rasterio.open(data_path).crs)
        forest_sindex = forest_regions.sindex

    with rasterio.open(data_path) as data:
        out_path = Path(out_dir)
        crs = data.crs.to_epsg()
        tilename = Path(data.name).stem

        for minx in np.arange(data.bounds[0], data.bounds[2], tile_width):
            for miny in np.arange(data.bounds[1], data.bounds[3], tile_height):
                out_path_root = out_path / f"{tilename}_{int(minx)}_{int(miny)}_{int(tile_width)}_{int(buffer)}_{crs}"

                bbox = box(
                    minx - buffer,
                    miny - buffer,
                    minx + tile_width + buffer,
                    miny + tile_height + buffer,
                )

                # Initialize flags
                only_forest, only_urban = False, False

                if forest_regions is not None:
                    # Efficiently check bounding box overlap using spatial index
                    possible_matches_index = list(forest_sindex.intersection(bbox.bounds))
                    possible_matches = forest_regions.iloc[possible_matches_index]
                    intersecting = possible_matches[possible_matches.intersects(bbox)]

                    if not intersecting.empty:
                        # If the intersection is not empty, it's at least partially in the forest
                        only_urban = False  # It's not urban if it intersects with the forest

                        # Check if the intersection is exactly the same as the bounding box
                        only_forest = intersecting.unary_union.equals(bbox)  # Full containment check
                    else:
                        # If there's no intersection, it's entirely outside the forest
                        only_urban = True
                geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)
                coords = get_features(geo)

                # Synchronously perform masking and transformations
                out_img, out_transform = mask(data, shapes=coords, crop=True)

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
    """Tiling multiple raster files and saving output tiles."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    async def process_file(data_path):
        img_out_dir = os.path.join(out_dir, Path(data_path).stem)
        try:
            await tile_single_file(
                data_path=data_path,
                out_dir=img_out_dir,
                buffer=buffer,
                tile_width=tile_width,
                tile_height=tile_height,
                forest_shapefile=forest_shapefile,
                logger=logger,
            )
        except Exception as e:
            if logger:
                logger.error(f"Error processing file: {e}")
            else:
                print(f"Error processing file: {e}")

    if parallel:
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
