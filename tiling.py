import json
import os
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import box
from rasterio.mask import mask
import shutil

from TreeDetection.helpers import delete_contents
"""
Tiling orthomosaic data.

Adapted from detectree2.preprocessing.tiling.tile_data function. (Released under MIT License, Version 1.2.x)
"""
def get_features(gdf: gpd.GeoDataFrame):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them."""
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]

def tile_single_file(
    data_path: str,
    out_dir: str,
    buffer: int = 0,
    tile_width: int = 50,
    tile_height: int = 50,
    dtype_bool: bool = False,
    logger=None
) -> None:
    if not os.path.exists(data_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    """Tiling a single raster file and saving output tiles."""
    with rasterio.open(data_path) as data:
        out_path = Path(out_dir)
        os.makedirs(out_path, exist_ok=True)
        crs = rasterio.CRS.from_string(data.crs.wkt)
        crs = data.crs.to_epsg()
        
        tilename = Path(data.name).stem
        # Calculate total tiles accounting for the entire bounds
        total_tiles = int(
            np.ceil((data.bounds[2] - data.bounds[0]) / tile_width) *
            np.ceil((data.bounds[3] - data.bounds[1]) / tile_height)
        )
        
        tile_count = 0
        if logger:
            logger.debug(f"Tiling {data_path} into {total_tiles} tiles")

        # Create tiles for all rows and columns, including the last row/column
        for minx in np.arange(data.bounds[0], data.bounds[2], tile_width):
            for miny in np.arange(data.bounds[1], data.bounds[3], tile_height):
                tile_count += 1
                out_path_root = out_path / f"{tilename}_{int(minx)}_{int(miny)}_{int(tile_width)}_{int(buffer)}_{crs}"
                
                # Create bounding box with buffer
                bbox = box(minx - buffer, miny - buffer, minx + tile_width + buffer, miny + tile_height + buffer)
                geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)
                coords = get_features(geo)  # Assuming this function retrieves geometrical features
                
                # Mask the data using the coordinates and crop
                out_img, out_transform = mask(data, shapes=coords, crop=True)
                out_sumbands = np.sum(out_img, 0)
                
                # Mask checks for zero and NaN values
                zero_mask = np.where(out_sumbands == 0, 1, 0)
                nan_mask = np.where(out_sumbands == 765, 1, 0)
                totalpix = out_img.shape[1] * out_img.shape[2]

                # Skip tiles with excessive zero or NaN values
                if zero_mask.sum() > 0.25 * totalpix or nan_mask.sum() > 0.25 * totalpix:
                    continue

                # Update metadata for the output tile
                out_meta = data.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "nodata": None,
                })

                if dtype_bool:
                    out_meta.update({"dtype": "uint8"})

                # Write the output TIFF file
                out_tif = out_path_root.with_suffix(".tif")
                if not os.path.exists(out_path_root.parent):
                    os.makedirs(out_path_root.parent)

                try:
                    with rasterio.open(out_tif, "w", **out_meta) as dest:
                        dest.write(out_img)
                except Exception as e:
                    logger.error(f"Failed to write {out_tif}: {e}")
                if not os.path.exists(out_tif):
                    logger.error(f'Out tif does not exist {out_tif}')

                # Read the saved TIFF file and convert to RGB for PNG output
                arr = rasterio.open(out_tif).read()
                rgb = np.dstack((arr[2], arr[1], arr[0]))  # BGR for cv2
                rgb_rescaled = 255 * rgb / 65535 if np.max(arr[1]) > 255 else rgb
                cv2.imwrite(str(out_path_root.with_suffix(".png").resolve()), rgb_rescaled)

def tile_data(
    file_list: list,
    out_dir: str,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    dtype_bool: bool = False,
    parallel: bool = False,
    max_workers: int = 4, 
    logger=None
) -> None:
    """
    Tiling multiple raster files and saving output tiles. 
    
    TODO (Currently single-threaded, due to file issues with gdal)
    
    Args:
        file_list (list): List of raster files to tile.
        out_dir (str): Output directory to save the tiles.
        buffer (int): Buffer to add around the tiles.
        tile_width (int): Width of each tile.
        tile_height (int): Height of each tile.
        dtype_bool (bool): Convert the output to boolean.
        parallel (bool): whether to use several processes or just one
        max_workers (int): Number of parallel workers to use.
        logger (Logger): Logger object for logging messages

    Returns:
        None
    """
    logger.info(f"Tiling {len(file_list)} files into {out_dir}")

    # Remove any existing files and ensure existence in the directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(out_dir):
        raise NotADirectoryError(f"Output directory is not a directory: {out_dir}")
    
    if not parallel:
        for data_path in file_list:   
            img_out_dir = os.path.join(out_dir, Path(data_path).stem)            
            try:
                tile_single_file(
                            data_path,
                            img_out_dir,
                            buffer,
                            tile_width,
                            tile_height,
                            dtype_bool,
                            logger
                        )      
            except Exception as e:
                if logger:
                    logger.error(f"Error processing file: {e}")
                else:
                    print(f"Error processing file: {e}")   
    else:
        # Tiles multiple raster files in parallel and saves the output tiles.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for data_path in file_list:            
                img_out_dir = os.path.join(out_dir, Path(data_path).stem)
                futures = [
                    executor.submit(
                        tile_single_file,
                        data_path,
                        img_out_dir,
                        buffer,
                        tile_width,
                        tile_height,
                        dtype_bool,
                        logger
                    )
                    
                ]
            for future in futures:
                try:
                    future.result()  # Ensure any exceptions are raised
                except Exception as e:
                    if logger:
                        logger.error(f"Error processing file: {e}")
                    else:
                        print(f"Error processing file: {e}")
if __name__ == "__main__":
    # Example usage
    file_list = ["file1.tif", "file2.tif", "file3.tif"]
    output_directory = "./output_tiles"
    tile_data(file_list, output_directory, buffer=30, tile_width=200, tile_height=200, max_workers=4)
