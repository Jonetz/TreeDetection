import asyncio
import os
import sys

import aiofiles
import cv2
import json

import fiona
import numpy as np
import warnings
import traceback
import shutil
import time
import numba as nb

import geopandas as gpd
import pandas as pd

from pathlib import Path

import rasterio
from fiona.model import to_dict
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from numpy.f2py.auxfuncs import throw_error
from rasterio.coords import BoundingBox
from rasterio.merge import merge
from rasterio.transform import xy
from rasterio.crs import CRS
from rasterio.windows import Window
from shapely.geometry import box, shape, Polygon
from shapely.errors import ShapelyError

from pycocotools import mask as mask_util

from concurrent.futures import ThreadPoolExecutor
from affine import Affine

from shapely.geometry import box, shape
import cupy as cp

import aiofiles

def exclude_outlines(config):
    for outline in config.get('exclude_files', []):
        exclude_outline = gpd.read_file(outline)

        for file in os.listdir(os.path.join(config["output_directory"], 'geojson_predictions')):
            if not (file.endswith('.geojson') or file.endswith('.gpkg')) or not file.startswith('processed_'):
                continue
            file_path = os.path.join(config["output_directory"], 'geojson_predictions', file)
            crowns = gpd.read_file(file_path)
            exclude_outline = exclude_outline.to_crs(crowns.crs)

            # Get the bounds of the current crowns GeoDataFrame
            file_bounds = crowns.total_bounds  # [minx, miny, maxx, maxy]

            # Clip the exclude outline to the bounds of the current file to save computing time
            exclude_outline_clipped = exclude_outline.clip(
                box(file_bounds[0], file_bounds[1], file_bounds[2], file_bounds[3])  # Using shapely's box
            )

            # Check which geometries in 'crowns' are completely within the exclude outline
            crowns_filtered = crowns[~crowns.geometry.within(exclude_outline_clipped.geometry.union_all())]

            # Write the filtered crowns back to the original path, overwriting the original file
            crowns_filtered.to_file(file_path, driver='GPKG')

class RoundedFloatEncoder(json.JSONEncoder):
    def __init__(self, *args, precision=2, **kwargs):
        self.precision = precision
        super().__init__(*args, **kwargs)

    def encode(self, obj):
        if isinstance(obj, float):
            return format(obj, f".{self.precision}f")
        elif isinstance(obj, dict):
            return "{" + ", ".join(f"{self.encode(k)}: {self.encode(v)}" for k, v in obj.items()) + "}"
        elif isinstance(obj, list):
            return "[" + ", ".join(self.encode(v) for v in obj) + "]"
        return super().encode(obj)


def polygon_from_mask(masked_arr):
    """Convert RLE data from the output instances into Polygons.

    Leads to a small about of data loss but does not affect performance?
    https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- adapted from here
    And we adapted it from detectree2
    """

    contours, _ = cv2.findContours(
        masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points) -  for security we use 10
        if contour.size >= 10:
            contour = contour.flatten().tolist()
            # Ensure the polygon is closed (get rid of fiona warning?)
            if contour[:2] != contour[-2:]:  # if not closed
                # continue # better to skip?
                contour.extend(contour[:2])  # small artifacts due to this?
            segmentation.append(contour)
    if len(segmentation) > 0:
        return segmentation[0]  # , [x, y, w, h], area
    else:
        return 0


def get_filenames(directory: str):
    """Get the file names if no geojson is present.

    Allows for predictions where no delineations have been manually produced.

    Args:
        directory (str): directory of images to be predicted on.
    """
    dataset_dicts = []
    # Traverse subdirectories inside the `tiles` directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):  # Assuming we're working with PNG images
                file_path = os.path.join(root, file)
                dataset_dicts.append({"file_name": file_path})

    return dataset_dicts


def project_to_geojson(tiles_path, pred_fold, output_fold, max_workers=4, logger=None, verbose=False):
    """
    Projects JSON predictions back to geographic space and saves them as GeoJSON.

    Args:
        tiles_path (str): Path to the tiles folder.
        pred_fold (str): Path to the predictions folder.
        output_fold (str): Path to the output folder.
        max_workers (int): Maximum number of threads for parallel processing.

    Returns:
        None
    """
    if not os.path.exists(tiles_path) or not os.path.isdir(tiles_path):
        raise FileNotFoundError(f"Tiles path not found: {tiles_path}")
    if not os.path.exists(pred_fold) or not os.path.isdir(pred_fold):
        raise FileNotFoundError(f"Predictions path not found: {pred_fold}")
    if os.path.exists(output_fold):
        if logger:
            logger.debug(f"Removing existing output folder: {output_fold}")
        # shutil.rmtree(output_fold)

    # Ensure output directory exists
    Path(output_fold).mkdir(parents=True, exist_ok=True)

    # Get list of JSON prediction files from pred_fold
    pred_files = list(Path(pred_fold).rglob("*.json"))  # Search recursively if needed
    total_files = len(pred_files)

    logger.info(f"Projecting {len(pred_files)} from {tiles_path} files to GeoJSON")

    if total_files == 0 and logger:
        logger.debug("No JSON files found in the predictions folder.")
        return

    if logger:
        logger.debug(f"Projecting files to GeoJSON")

    # List all TIFF files in the tiles directory and create a lookup dictionary for fast matching
    tiff_files = list(Path(tiles_path).rglob("*.json"))
    # Create a dictionary with base_name as the key for fast lookup
    tif_lookup = {}
    for tif_file in tiff_files:
        tif_file = Path(tif_file)
        base_name = tif_file.stem.replace("Prediction_", "").replace(".json", "")
        tif_lookup[base_name] = tif_file

    def get_matching_tif_path(tile_image_name):
        """
        Finds the corresponding TIFF path for the given tile image name based on the start of the filename.
        """
        base_name = tile_image_name.replace("Prediction_", "").replace(".json", "")
        # Lookup the TIFF file directly in the dictionary
        return tif_lookup.get(base_name)  # Return None if no match is found

    def extract_base_image_name(tif_path):
        """
        Extracts the base image name from the TIFF file path.
        """
        # Split the path to get the folder and file name
        folder_name = tif_path.parent.name
        return folder_name  # Return the folder name which is the base image name

    def process_file(filename):
        try:
            # Extract the tile image name from the prediction file
            tile_image_name = filename.name

            # Find the corresponding TIFF path
            tifpath = get_matching_tif_path(tile_image_name)

            if tifpath is None:
                raise FileNotFoundError(f"No matching TIFF file found for {tile_image_name}")

            # Extract the base image name from the TIFF path
            base_image_name = extract_base_image_name(tifpath)

            # Generate output GeoJSON file path using the base image name
            output_image_folder = Path(output_fold) / base_image_name
            output_image_folder.mkdir(parents=True, exist_ok=True)  # Ensure the image directory exists
            output_geo_file = output_image_folder / (tile_image_name.replace(".json", "").replace("Prediction_",
                                                                                                  "") + ".gpkg")  # Use the tile image name for the GeoJSON name
            if os.path.isfile(output_geo_file):
                logger.debug(f"file {tile_image_name} already processed for projecting to geojson/gpkg")
                return

            metadata_path = tifpath.with_name(f"{tifpath.stem}.json")
            with open(metadata_path, "r") as meta_file:
                metadata = json.load(meta_file)
                epsg = metadata["crs"]
                raster_transform = metadata["transform"]
            raster_transform = Affine(*metadata["transform"]) if not isinstance(metadata["transform"], Affine) else \
                metadata["transform"]

            # Load the prediction JSON
            with open(filename, "r") as prediction_file:
                datajson = json.load(prediction_file)

            # List to collect all features as geometry and properties
            features = []

            # Process each polygon in the prediction
            for crown_data in datajson:
                confidence_score = crown_data["score"]
                mask_of_coords = mask_util.decode(crown_data["segmentation"])
                crown_coords = polygon_from_mask(mask_of_coords)

                if not crown_coords:
                    continue

                crown_coords_array = np.array(crown_coords).reshape(-1, 2)
                x_coords, y_coords = xy(raster_transform, crown_coords_array[:, 1], crown_coords_array[:, 0])
                moved_coords = list(zip(x_coords, y_coords))

                # Create a Polygon geometry
                polygon = Polygon(moved_coords)
                features.append({"geometry": polygon, "Confidence_score": confidence_score})

            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(features, crs=f"EPSG:{epsg}")

            # Save to GPKG with specified precision
            gdf.to_file(output_geo_file, driver="GPKG")

            return f"Successfully processed: {filename.name}"
        except Exception as e:
            return f"Failed to process {filename.name}: {e}"

    # Parallelize the processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_file, pred_files))

    # Log the processing results
    for result in results:
        if logger and verbose:
            logger.debug(result)
    if logger and verbose:
        logger.debug("GeoJSON/GPKG projection complete.")


def filename_geoinfo(filename):
    """Return geographic info of a tile from its filename.

    Copied directly from detectree2 
    """
    if os.path.splitext(filename)[1] == ".tif":
        parts = os.path.basename(filename).replace(".tif", "").split("_")

        try:
            x_coord = int(parts[1])
            y_coord = int(parts[2])
            return x_coord, y_coord
        except Exception as e:
            raise Exception("Filename not compatible.", e)
    else:
        parts = os.path.basename(filename).replace(".geojson", "").replace(".json", "").replace(".gpkg", "").split("_")

        parts = [int(part) for part in parts[-5:]]  # type: ignore
        minx = parts[0]
        miny = parts[1]
        width = parts[2]
        buffer = parts[3]
        crs = parts[4]
        return (minx, miny, width, buffer, crs)


def tif_geoinfo(filename):
    with rasterio.open(filename, 'r') as source:
        return source.transform, source.crs, source.width, source.height


def box_make(minx: int, miny: int, width: int, buffer: int, crs, shift: int = 0):
    """Generate bounding box from geographic specifications.

    Copied directly from detectree2 

    Args:
        minx: Minimum x coordinate.
        miny: Minimum y coordinate.
        width: Width of the tile.
        buffer: Buffer around the tile.
        crs: Coordinate reference system.
        shift: Number of meters to shift the size of the bounding box in by. This is to avoid edge crowns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the bounding box.
    """
    bbox = box(
        minx - buffer + shift,
        miny - buffer + shift,
        minx + width + buffer - shift,
        miny + width + buffer - shift,
    )
    geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=CRS.from_epsg(crs))
    return geo


def box_filter(filename, shift: int = 0):
    """Create a bounding box from a file name to filter edge crowns.

    Copied directly from detectree2 

    Args:
        filename: Name of the file.
        shift: Number of meters to shift the size of the bounding box in by. This is to avoid edge crowns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the bounding box.
    """
    minx, miny, width, buffer, crs = filename_geoinfo(filename)
    bounding_box = box_make(minx, miny, width, buffer, crs, shift)
    return bounding_box

def xy_gpu(raster_transform, y_coords, x_coords):
    """
    Transforms coordinates from raster space to the spatial reference system using the affine transformation matrix.

    Args:
        raster_transform (Affine): The affine transformation matrix.
        y_coords (np.ndarray): Array of y-coordinates.
        x_coords (np.ndarray): Array of x-coordinates.

    Returns:
        tuple: Transformed x and y coordinates.
    """
    # Convert coordinates to cupy arrays for GPU processing
    y_coords_gpu = cp.asarray(y_coords)
    x_coords_gpu = cp.asarray(x_coords)

    # Extract affine transformation parameters
    a, b, c, d, e, f, g, h, i = raster_transform

    # Perform the affine transformation (x', y' = Ax + By + C, Dx + Ey + F)
    # Using GPU with cupy for parallel processing
    transformed_x = a * x_coords_gpu + b * y_coords_gpu + c
    transformed_y = d * x_coords_gpu + e * y_coords_gpu + f

    # Convert the results back to numpy arrays (if needed) or return as cupy arrays for further GPU processing
    return cp.asnumpy(transformed_x), cp.asnumpy(transformed_y)

def stitch_crowns(folder: str, shift: int = 1, max_workers=4, logger=None, simplify_tolerance=0.2):
    """
    Stitch together predicted crowns from multiple geojson files, applying a spatial filter.

    Args:
        folder: Path to folder containing geojson files.
        shift: Number of meters to shift the size of the bounding box by to avoid edge crowns.
        max_workers: Maximum number of threads for parallel processing.
        logger: Logger object for logging messages.
        simplify_tolerance: Tolerance level for geometry simplification in meters. 
                            Higher values result in more simplification.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing all the crowns.
    """
    crowns_path = Path(folder)
    files = list(crowns_path.glob("*gpkg"))

    if len(files) == 0:
        raise FileNotFoundError(f"No geojson files found in folder {folder}.")

    # Get CRS from the first file, with error handling
    try:
        _, _, _, _, crs = filename_geoinfo(files[0])
    except Exception as e:
        raise ValueError(f"Failed to retrieve CRS from the first file: {files[0]}. Error: {e}")

    total_files = len(files)
    if logger:
        logger.debug(f"Stitching crowns from {total_files} files")

    # Suppress Fiona 'closed ring' warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Fiona.*closed ring.*")

    def process_file(file):
        try:
            # Read the GeoJSON file, handle file I/O errors
            crowns_tile = gpd.read_file(file)

            # Apply box filter based on shift, ensure no issues arise from bounding box
            geo = box_filter(file, shift)

            # Perform spatial join to filter crowns within the box
            crowns_tile = gpd.sjoin(crowns_tile, geo, "inner", "within")

            # Simplify geometries if tolerance is set
            if simplify_tolerance > 0:
                crowns_tile['geometry'] = crowns_tile['geometry'].simplify(simplify_tolerance, preserve_topology=True)

            return crowns_tile
        except FileNotFoundError:
            logger.warn(f"File not found: {file}")
        except gpd.errors.EmptyOverlayError:
            logger.debug(f"Spatial join failed (empty result) for file: {file}")
        except Exception as e:
            logger.warn(f"An error occurred while processing {file}: {e}, see debug for more details.")
            logger.debug(traceback.format_exc())

        return None

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        crowns_list = list(executor.map(process_file, files))

    # Filter out None results (in case of errors)
    crowns_list = [crowns for crowns in crowns_list if crowns is not None]

    if len(crowns_list) == 0:
        raise RuntimeError("No valid crowns were processed.")

    try:
        # Concatenate all crowns into one GeoDataFrame
        crowns = pd.concat(crowns_list, ignore_index=True)
    except ValueError as e:
        raise RuntimeError(f"Error concatenating crowns: {e}")

    # Drop unnecessary index column from spatial join
    if "index_right" in crowns.columns:
        crowns = crowns.drop("index_right", axis=1)

    # Ensure the output is a GeoDataFrame with the correct CRS
    if not isinstance(crowns, gpd.GeoDataFrame):
        try:
            crowns = gpd.GeoDataFrame(crowns, crs=CRS.from_epsg(crs))
        except Exception as e:
            raise RuntimeError(f"Error converting to GeoDataFrame with CRS: {e}")

    return crowns

def validate_paths(tiles_path, pred_fold, output_path):
    if not os.path.exists(tiles_path) or not os.path.isdir(tiles_path):
        raise FileNotFoundError(f"Tiles path not found: {tiles_path}")
    if not os.path.exists(pred_fold) or not os.path.isdir(pred_fold):
        raise FileNotFoundError(f"Predictions path not found: {pred_fold}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
       
def process_prediction_file_sync(file, tif_lookup, shift, simplify_tolerance, logger=None):
    try:
        # Match JSON file to corresponding TIFF file
        tifpath = tif_lookup.get(Path(file).stem.replace("Prediction_", ""))
        if not tifpath:
            raise FileNotFoundError(f"No matching TIFF file for {file}")
        
        metadata_path = tifpath.with_name(f"{tifpath.stem}.json")
        with open(metadata_path, "r") as meta_file:
            metadata = json.load(meta_file)
        
        epsg = metadata["crs"]
        raster_transform = Affine(*metadata["transform"])
        
        # Load predictions
        with open(file, "r") as pred_file:
            data = json.load(pred_file)

        bounding_box = box_filter(tifpath, shift)

        # Process each prediction
        features = []
        for crown_data in data:
            unreshaped_coords = None
            if 'polygon_coords' in crown_data:
                coords = np.array(crown_data["polygon_coords"]).reshape(-1, 2)
                unreshaped_coords = np.array(crown_data["polygon_coords"])
            else:
                if "bbox" in crown_data:
                    bbox = np.array(crown_data["bbox"])
                    crown_data["bbox"] = bbox.tolist()

                mask = mask_util.decode(crown_data["segmentation"])
                polygon_coords = polygon_from_mask(mask)
                if not polygon_coords:
                    continue
                crown_data["polygon_coords"] = polygon_coords
                coords = np.array(polygon_coords).reshape(-1, 2)
                unreshaped_coords = np.array(polygon_coords)
            polygon = Polygon(coords)

            # Check if it's near the tile border
            # TODO: This is buggy
            # is_inside = exclude_elements_near_border(unreshaped_coords, bounding_box)
            # if is_inside:
            #     features.append({"geometry": polygon, "Confidence_score": crown_data["score"]})
            features.append({"geometry": polygon, "Confidence_score": crown_data["score"]})

        gdf = gpd.GeoDataFrame(features, geometry=[feature["geometry"] for feature in features], crs=f"EPSG:{epsg}")
        
        if simplify_tolerance > 0:
            gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance, preserve_topology=True)
        filtered_gdf = gpd.sjoin(gdf, bounding_box, "inner", "within")
        if 'index_right' in filtered_gdf.columns:
            filtered_gdf = filtered_gdf.rename(columns={'index_right': 'filter_index_right'})
        
        return filtered_gdf
    except Exception as e:
        if logger:
            logger.warn(f"Error processing file {file}: {e}")
        return None


def exclude_elements_near_border(filtered_gdf, bounding_box):
    eps = 20

    minx = bounding_box.geometry.bounds.minx.iloc[0] + eps
    miny = bounding_box.geometry.bounds.miny.iloc[0] + eps
    maxx = bounding_box.geometry.bounds.maxx.iloc[0] - eps
    maxy = bounding_box.geometry.bounds.maxy.iloc[0] - eps

    x_coords = filtered_gdf[:, :, 0]
    y_coords = filtered_gdf[:, :, 1]

    if (np.array(x_coords[0]) < minx).any() or (np.array(x_coords[0]) > maxx).any() or (np.array(y_coords[0]) < miny).any() or (np.array(y_coords[0]) > maxy).any():
        return False
    return True


def process_folder_sync(folder, tiles_path, pred_fold, output_path, shift, simplify_tolerance, logger=None):
    try:
        if logger:
            logger.info(f"Starting {folder}. ")
        image_folder_path = os.path.join(tiles_path, folder)
        prediction_folder_path = os.path.join(pred_fold, folder)
        tiff_files = list(Path(image_folder_path).rglob("*.json"))
        tif_lookup = {Path(tif).stem: Path(tif) for tif in tiff_files}

        pred_files = list(Path(prediction_folder_path).rglob("*.json"))
        
        # Process each prediction file
        results = [
            process_prediction_file_sync(file, tif_lookup, shift, simplify_tolerance, logger)
            for file in pred_files
        ]
        valid_results = [res for res in results if res is not None]

        if not valid_results:
            if logger:
                logger.debug(f"No valid results for folder {folder}. Creating empty output.")
            combined_gdf = gpd.GeoDataFrame(pd.DataFrame(), crs="EPSG:4326", geometry=gpd.GeoSeries([]))
        else:
            combined_gdf = gpd.GeoDataFrame(pd.concat(valid_results, ignore_index=True))
        
        output_file = os.path.join(output_path, f"{folder}.gpkg")
        combined_gdf.to_file(output_file, driver="GPKG")
        if logger:
            logger.info(f"Processed folder {folder} -> {output_file}")

        return output_file
    except Exception as e:
        if logger:
            logger.error(f"Error processing folder {folder}: {e}")
        return None


def process_and_stitch_predictions(tiles_path, pred_fold, output_path, max_workers=50, shift=1, simplify_tolerance=0.2, logger=None):
    validate_paths(tiles_path, pred_fold, output_path)
    folders = [f for f in os.listdir(tiles_path) if os.path.isdir(os.path.join(tiles_path, f))]

    async def process_all_folders():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    process_folder_sync,
                    folder,
                    tiles_path,
                    pred_fold,
                    output_path,
                    shift,
                    simplify_tolerance,
                    logger,
                )
                for folder in folders
            ]
            return await asyncio.gather(*tasks)
    
    results = asyncio.run(process_all_folders())
    return output_path

def calc_iou(shape1, shape2):
    """Calculate the IoU of two shapes."""
    iou = shape1.intersection(shape2).area / shape1.union(shape2).area
    return iou


def round_coordinates(crowns: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Round the coordinates of the geometries in the GeoDataFrame."""

    def round_geometry(geometry):
        # Check if geometry is valid
        if geometry.is_empty or not geometry.is_valid:
            return geometry

        # Round each coordinate
        coords = np.array(geometry.exterior.coords)
        rounded_coords = [(round(x, 1), round(y, 1)) for x, y in coords]

        # Create a new Polygon with rounded coordinates
        return gpd.GeoSeries([gpd.GeoPolygon(rounded_coords)])

    # Apply rounding to each geometry
    crowns['geometry'] = crowns['geometry'].apply(round_geometry)

    return crowns


def clean_crowns(crowns: gpd.GeoDataFrame,
                 iou_threshold: float = 0.7,
                 confidence: float = 0.2,
                 area_threshold: float = 1,
                 field: str = "Confidence_score",
                 logger=None) -> gpd.GeoDataFrame:
    """Clean overlapping crowns.

    copied from detectree2 

    Outputs can contain highly overlapping crowns including in the buffer region.
    This function removes crowns with a high degree of overlap with others but a
    lower Confidence Score.

    Args:
        crowns (gpd.GeoDataFrame): Crowns to be cleaned.
        iou_threshold (float, optional): IoU threshold that determines whether crowns are overlapping.
        confidence (float, optional): Minimum confidence score for crowns to be retained. Defaults to 0.2. Note that
            this should be adjusted to fit "field".
        area_threshold (float, optional): Minimum area of crowns to be retained. Defaults to 1m2 (assuming UTM).
        field (str): Field to used to prioritise selection of crowns. Defaults to "Confidence_score" but this should
            be changed to "Area" if using a model that outputs area.

    Returns:
        gpd.GeoDataFrame: Cleaned crowns.
    """
    # Ensure the input is a GeoDataFrame with valid geometries
    if not isinstance(crowns, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame")

    # Filter any rows with empty or invalid geometry
    crowns = crowns[crowns.is_valid & ~crowns.is_empty]

    # Filter any rows with polygon area less than the area threshold
    crowns = crowns[crowns.area > area_threshold].reset_index(drop=True)

    if crowns.empty:
        if logger:
            logger.debug("No valid crowns remaining after initial filtering.")
        return gpd.GeoDataFrame(columns=crowns.columns, crs=crowns.crs)

    if logger:
        logger.debug(f"Cleaning {len(crowns)} crowns")

    cleaned_crowns = []

    # Calculate IoU for all pairs once and store in a DataFrame
    for index, row in crowns.iterrows():
        if index % 1000 == 0 and logger:
            logger.debug(f"Processing {index} / {len(crowns)} crowns cleaned")

        try:
            # Find intersecting crowns
            intersecting_rows = crowns[crowns.intersects(shape(row.geometry))]

            if len(intersecting_rows) > 1:
                iou_values = intersecting_rows.geometry.apply(lambda x: calc_iou(row.geometry, x))
                intersecting_rows = intersecting_rows.assign(iou=iou_values)

                # Filter rows with IoU over threshold and get the one with the highest confidence score
                match = intersecting_rows[intersecting_rows["iou"] > iou_threshold].nlargest(1, field)

                if match.empty or match["iou"].iloc[0] < 1:
                    continue  # Skip if no match found or IoU is less than 1
            else:
                match = row.to_frame().T  # No overlaps, retain current crown

            cleaned_crowns.append(match)

        except ShapelyError as e:
            if logger:
                logger.warn(f"Shapely error while processing row {index}: {e}")
        except Exception as e:
            if logger:
                logger.warn(f"Error while processing row {index}: {e}")

    if cleaned_crowns:
        crowns_out = pd.concat(cleaned_crowns, ignore_index=True)
    elif logger:
        logger.debug("No crowns were cleaned. Returning an empty GeoDataFrame.")
    if not cleaned_crowns:
        return gpd.GeoDataFrame(columns=crowns.columns, crs=crowns.crs)

    # Drop 'iou' column and ensure crowns_out is a GeoDataFrame
    crowns_out = crowns_out.drop(columns=["iou"], errors='ignore')
    try:
        crowns_out = gpd.GeoDataFrame(crowns_out)
        crowns_out.set_crs(crowns.crs, allow_override=True)
    except Exception as e:
        if logger:
            logger.debug(f"Error converting to GeoDataFrame: {e} with CRS: {crowns.crs} trying to fix it.")
        crowns_out = gpd.GeoDataFrame(crowns_out)
        crowns_out.set_crs(crowns.crs, allow_override=True)
        return gpd.GeoDataFrame(columns=crowns.columns)

    # Filter remaining crowns based on confidence score
    if confidence > 0:
        crowns_out = crowns_out[crowns_out[field] > confidence]

    return crowns_out.reset_index(drop=True)


def fuse_predictions(urban_fold, forrest_fold, forrest_path, output_dir, logger=None):
    """
    Fuse the predictions according to the configuration.

    Args:
        urban_fold (str): Path to the urban predictions folder.
        forrest_fold (str): Path to the forest predictions folder.
        forrest_path (str): Path to the forest boundary.
        output_dir (str): Path to the output directory.
        logger: Logger object for logging messages.
    """
    if not os.path.exists(urban_fold) or not os.path.isdir(urban_fold):
        raise FileNotFoundError(f"Urban predictions path not found: {urban_fold}")
    if not os.path.exists(forrest_fold) or not os.path.isdir(forrest_fold):
        raise FileNotFoundError(f"Forest predictions path not found: {forrest_fold}")
    if not os.path.exists(forrest_path) or not os.path.isfile(forrest_path):
        raise FileNotFoundError(f"Forest boundary path not found: {forrest_path}")
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        if os.path.exists(output_dir) and not os.path.isdir(output_dir):
            os.unlink(output_dir)  # Synchronously remove directories
        os.makedirs(output_dir, exist_ok=True)

    forest_boundary = gpd.read_file(forrest_path)
    # Ensure that the geometries are not None and valid
    none_geometries_count = forest_boundary['geometry'].isnull().sum()
    if none_geometries_count > 0:
        logger.debug(f"Found {none_geometries_count} None geometries in forest shape. Removing them.")
        # Remove the rows with None geometries
        forest_boundary = forest_boundary[~forest_boundary['geometry'].isnull()]
    # Log invalid geometries
    if not forest_boundary.is_valid.all():
        logger.debug("Invalid geometries found. Attempting to fix...")
        # Fix invalid geometries by calling make_valid
        forest_boundary["geometry"] = forest_boundary["geometry"].apply(
            lambda geom: geom.make_valid() if not geom.is_valid else geom
        )
        # Check if all geometries are valid after make_valid
        if not forest_boundary.is_valid.all():
            logger.warn(f"Some geometries are still invalid after the fix.")

    # Optionally, remove empty geometries
    forest_boundary = forest_boundary[~forest_boundary.is_empty]

    # Apply buffer(0) to fix remaining issues (e.g., self-intersections)
    forest_boundary["geometry"] = forest_boundary["geometry"].apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom
    )

    for top_folder in [urban_fold]:
        # For each folder, process corresponding urban and forest GeoJSONs
        for name in os.listdir(top_folder):
            if not (name.endswith(".geojson") or name.endswith(".gpkg")):
                continue
            urban_geojson_path = os.path.join(urban_fold, name)
            forest_geojson_path = os.path.join(forrest_fold, name)
            if not os.path.exists(urban_geojson_path):
                if logger:
                    logger.error(
                        f"Urban GeoJSON for tile {name} at path {urban_geojson_path} not found. Skipping tile.")
                continue

            if not os.path.exists(forest_geojson_path):  #
                if logger:
                    logger.error(
                        f"Forest GeoJSON for tile {name} at path {forest_geojson_path} not found. Skipping tile.")
                continue

            if os.path.exists(urban_geojson_path) and os.path.exists(forest_geojson_path):
                if logger:
                    logger.debug(f"Fusing predictions for tile {name}...")

                output_path = os.path.join(output_dir, os.path.basename(name))
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                #if os.path.isfile(output_path):
                #    logger.debug(f"file {name} already processed")
                #    continue

                # Read the urban and forest predictions
                urban_shapes = gpd.read_file(urban_geojson_path)
                forest_shapes = gpd.read_file(forest_geojson_path)

                # If there are no urban shapes, skip the fusion and only process forest
                if urban_shapes is None or urban_shapes.empty:
                    output_path = os.path.join(output_dir, os.path.basename(name))
                    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                    forest_shapes.to_file(output_path, driver="GPKG")
                    logger.debug(f"Only forest file saved to {output_path}")
                    continue

                # If there are no forest shapes, skip the fusion and only process urban
                if forest_shapes is None or forest_shapes.empty:
                    output_path = os.path.join(output_dir, os.path.basename(name))
                    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                    urban_shapes.to_file(output_path, driver="GPKG")
                    logger.debug(f"Only urban file saved to {output_path}")
                    continue

                # Ensure CRS is the same for all geometries
                if not urban_shapes.crs == forest_shapes.crs == forest_boundary.crs:
                    if logger:
                        logger.warning("CRS mismatch detected. Aligning CRS to match the forest boundary.")
                    urban_shapes = urban_shapes.to_crs(forest_boundary.crs)
                    forest_shapes = forest_shapes.to_crs(forest_boundary.crs)
                
                # Step 1: Get combined bounds of urban_shapes and forest_shapes
                urban_bounds = urban_shapes.total_bounds  # [minx, miny, maxx, maxy]
                forest_bounds = forest_shapes.total_bounds  # [minx, miny, maxx, maxy]

                # Combine the two bounding boxes
                combined_bounds = [
                    min(urban_bounds[0], forest_bounds[0]),  # minx
                    min(urban_bounds[1], forest_bounds[1]),  # miny
                    max(urban_bounds[2], forest_bounds[2]),  # maxx
                    max(urban_bounds[3], forest_bounds[3]),  # maxy
                ]
                
                # Step 2: Clip forest_boundary to combined_bounds
                combined_bbox = box(*combined_bounds)
                forest_clipped = forest_boundary[forest_boundary.geometry.intersects(combined_bbox)]

                # Step 3: Perform intersection and exclusion operations
                # Find forest shapes that intersect with the clipped forest boundary
                forest_intersecting = forest_shapes[forest_shapes.geometry.intersects(forest_clipped.unary_union)]

                # Exclude urban areas that are fully within the forest boundary
                urban_outside_forest = urban_shapes[
                    ~urban_shapes.geometry.within(forest_clipped.unary_union)
                ]
                # Combine the results: clipped forest_shapes and urban_shapes outside the forest boundary
                fused_shapes = pd.concat([forest_intersecting, urban_outside_forest], ignore_index=True)

                # Ensure all geometries are valid
                fused_shapes["geometry"] = fused_shapes.geometry.apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

                # Ensure the fused geometries are valid
                if not all(fused_shapes.is_valid):
                    forest_boundary = fused_shapes.make_valid()
                    fused_shapes['geometry'] = fused_shapes.buffer(0)  # Fix invalid geometries
                    if not all(fused_shapes.is_valid) and logger:
                        logger.warning(
                            f"Invalid geometries detected in fused shapes for tile {name}. Attempting to fix.")

                        # Save the fused result as a new GeoJSON file
                fused_shapes.to_file(output_path, driver="GPKG")
                logger.debug(f"File saved to {output_path}")

def delete_contents(out_dir, logger=None):
    # Check for existing files and remove them synchronously
    if os.listdir(out_dir):
        if logger:
            logger.debug(f"Removing existing files in {out_dir}")
        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Synchronously remove individual files
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Synchronously remove directories
            except Exception as e:
                if logger:
                    logger.error(f"Failed to delete {file_path}: {e}")
                else:
                    print(f"Failed to delete {file_path}: {e}")

def check_similarity_bounds(bounds1: BoundingBox, bounds2: BoundingBox, tolerance=1e-3):
    left = abs(bounds1.left - bounds2.left) < tolerance
    right = abs(bounds1.right - bounds2.right) < tolerance
    top = abs(bounds1.top - bounds2.top) < tolerance
    bottom = abs(bounds1.bottom - bounds2.bottom) < tolerance
    return left and right and top and bottom

@nb.njit(fastmath=True)
def ndvi_index(red_value, nir_value):
    """
    Calculate the NDVI index from the red and NIR values.

    Args:
        red_value (float): Red channel value.
        nir_value (float): NIR channel value.

    Returns:
        float: NDVI index value.
    """
    epsilon = 1e-10
    ndvi_value = (nir_value - red_value) / (nir_value + red_value + epsilon)
    if ndvi_value < -1.0 or ndvi_value > 1.0:
        raise ValueError(f"NDVI value out of range: {ndvi_value}")
    return ndvi_value

@nb.njit(fastmath=True)
def ndvi_array_from_rgbi(rgbi_array: np.ndarray):
    """
    Calculate the NDVI index from the RGBI array.

    Args:
        rgbi_array (np.ndarray): Array containing the RGBI values.

    Returns:
        np.ndarray: Array containing the NDVI index values
    """

    ndvi_array = np.zeros(shape=(rgbi_array.shape[1], rgbi_array.shape[2]))
    for i in range(rgbi_array.shape[1]):
        for j in range(rgbi_array.shape[2]):
            ndvi_array[i, j] = ndvi_index(rgbi_array[0, i, j] / 255.0, rgbi_array[3, i, j] / 255.0)
    return ndvi_array


def create_ndvi_image_from_rgbi(rgbi_path: str, ndvi_path: str, export_tif: bool = True, export_png: bool = False):
    """
    Create an NDVI image from the RGBI image. Mainly for debugging purposes.

    Args:
        rgbi_path (str): Path to the RGBI image.
        ndvi_path (str): Path to save the NDVI image.
        export_tif (bool): Export the NDVI image as a TIFF file.
        export_png (bool): Export the NDVI image as a PNG file

    Raises:
        FileNotFoundError: If the RGBI file is not found.
    """
    if not os.path.exists(rgbi_path) or not os.path.isfile(rgbi_path):
        raise FileNotFoundError(f" RGB File not found: {rgbi_path}")

    with rasterio.open(rgbi_path) as rgb_src:
        print(type(rgb_src))
        rgbi_array = rgb_src.read()

    ndvi_array = np.zeros(shape=(rgbi_array.shape[1], rgbi_array.shape[2]))

    rgb_normalized = rgbi_array / 255.0

    for i in range(rgb_normalized.shape[1]):
        for j in range(rgb_normalized.shape[2]):
            ndvi_array[i, j] = ndvi_index(rgb_normalized[0, i, j], rgb_normalized[3, i, j])

    ndvi_flattened = np.squeeze(ndvi_array)

    image_min = np.min(ndvi_flattened)
    image_max = np.max(ndvi_flattened)

    # Normalize the values to 0–1
    normalized_image = (ndvi_flattened - image_min) / (image_max - image_min) * 255.0

    out_path_root_png = Path(ndvi_path)

    if export_png:
        cv2.imwrite(str(out_path_root_png), normalized_image)

    if export_tif:
        normalized_image = np.expand_dims(normalized_image, axis=0)
        out_meta = rgb_src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": rgb_src.shape[0],
            "width": rgb_src.shape[1],
            "transform": rgb_src.transform,
            "nodata": None,
            "count": 1,
        })

        # Write the output TIFF file
        out_tif = out_path_root_png.with_suffix(".tif")

        try:
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(normalized_image)
        except Exception as e:
            print(f"Failed to write {out_tif}: {e}")


def plot_ndvi_values(values_array: np.ndarray):
    # Normalize the data to range [0, 1] for the colormap
    norm = Normalize(vmin=-1, vmax=1)

    # Apply the viridis colormap
    colormap = plt.cm.viridis
    mapped_data = colormap(norm(values_array))  # Convert normalized data to RGBA

    # Specify resolution
    width, height = 5000, 5000  # Resolution in pixels
    dpi = 300  # Dots per inch
    figsize = (width / dpi, height / dpi)  # Size of the figure in inches

    # Create and save the image
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(mapped_data, origin='upper')
    plt.axis('off')  # Turn off axes for visualization
    plt.savefig("viridis_image_high_res.png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.show()


def retrieve_neighboring_image_filenames(filename, other_filenames):
    """
    Retrieve the filenames of the neighboring images.

    Args:
        filename (str): Filename of the image.
        other_filenames (list): List of other filenames.
    """
    transform, crs, width, height = tif_geoinfo(filename)
    x = transform.c
    y = transform.f

    left = None
    right = None
    up = None
    down = None

    eps = 1e-3

    for other in other_filenames:
        if other == filename:
            continue

        other_transform, other_crs, other_width, other_height = tif_geoinfo(other)

        if abs(other_transform.c - (x - (width*other_transform.a))) < eps and abs(other_transform.f - y) < eps:
            left = other

        if abs(other_transform.c - (x + (width*other_transform.a))) < eps and abs(other_transform.f - y) < eps:
            right = other

        if abs(other_transform.f - (y + (height*other_transform.a))) < eps and abs(other_transform.c - x) < eps:
            up = other

        if abs(other_transform.f - (y - (height*other_transform.a))) < eps and abs(other_transform.c - x) < eps:
            down = other

    return (left, right, up, down)


def merge_images(src1, src2):
    """
    Merge two images with the same CRS.

    Args:
        src1 (rasterio.DatasetReader): First image.
        src2 (rasterio.DatasetReader): Second image.
    """
    # Open the input images
    if src1.crs != src2.crs:
        raise ValueError("CRS of the two images do not match.")

    # Merge the images
    merged_data, merged_transform = merge([src1, src2])

    # Update metadata for the merged image
    merged_meta = src1.meta.copy()
    merged_meta.update({
        "driver": "GTiff",
        "height": merged_data.shape[1],
        "width": merged_data.shape[2],
        "transform": merged_transform,
    })
    return merged_data, merged_meta


def crop_image(src, width, height):
    """
    Crop an image to the specified dimensions.

    Args:
        src (rasterio.DatasetReader): Source image.
        width (int): Width of the cropped image.
        height (int): Height of the cropped image.
    """
    # Get the image dimensions
    img_width, img_height = src.width, src.height

    # Compute the center of the image
    center_x, center_y = img_width // 2, img_height // 2

    # Calculate the bounds of the cropping window
    window_left = max(center_x - width // 2, 0)
    window_top = max(center_y - height // 2, 0)
    window = Window(window_left, window_top, width, height)

    # Read the cropped window
    cropped_data = src.read(window=window)

    # Update metadata for the cropped image
    cropped_transform = src.window_transform(window)
    cropped_meta = src.meta.copy()
    cropped_meta.update({
        "width": width,
        "height": height,
        "transform": cropped_transform
    })

    # Save the cropped image
    return cropped_data, cropped_meta