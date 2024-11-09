import os
import glob
import cv2
import json
import numpy as np
import rasterio
import warnings
import traceback
import shutil
import time

import geopandas as gpd
import pandas as pd

from pathlib import Path
from rasterio.transform import xy
from rasterio.crs import CRS
from shapely.geometry import box, shape
from shapely.errors import ShapelyError

from pycocotools import mask as mask_util

from concurrent.futures import ThreadPoolExecutor


def polygon_from_mask(masked_arr):
    """Convert RLE data from the output instances into Polygons.

    Leads to a small about of data loss but does not affect performance?
    https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- adapted from here
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

    [x, y, w, h] = cv2.boundingRect(masked_arr)

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
        logger.debug(f"Projecting {total_files} files to GeoJSON")

    # List all TIFF files in the tiles directory
    tiff_files = list(Path(tiles_path).rglob("*.tif"))

    def get_matching_tif_path(tile_image_name):
        """
        Finds the corresponding TIFF path for the given tile image name based on the start of the filename.
        """
        base_name = tile_image_name.replace("Prediction_", "").replace(".json", "")
        for tif_file in tiff_files:
            if tif_file.name.startswith(base_name):  # Compare the start of the TIFF filename
                return tif_file  # Return the first matching TIFF file found
        return None  # No matching TIFF found

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
                                                                                                  "") + ".geojson")  # Use the tile image name for the GeoJSON name
            if os.path.isfile(output_geo_file):
                logger.debug(f"file {tile_image_name} already processed for projecting to geojson")
                return

            with rasterio.open(tifpath) as data:
                epsg = CRS.from_string(data.crs.wkt).to_epsg()
                raster_transform = data.transform

            # Create a GeoJSON structure
            geofile = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": f"urn:ogc:def:crs:EPSG::{epsg}"
                    },
                },
                "features": [],
            }

            # Load the prediction JSON
            with open(filename, "r") as prediction_file:
                datajson = json.load(prediction_file)

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

                feature = {
                    "type": "Feature",
                    "properties": {
                        "Confidence_score": confidence_score
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [moved_coords],
                    },
                }
                geofile["features"].append(feature)

            with open(output_geo_file, "w") as dest:
                json.dump(geofile, dest)
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
        logger.debug("GeoJSON projection complete.")


def filename_geoinfo(filename):
    """Return geographic info of a tile from its filename.

    Copied directly from detectree2 
    """
    parts = os.path.basename(filename).replace(".geojson", "").split("_")

    parts = [int(part) for part in parts[-5:]]  # type: ignore
    minx = parts[0]
    miny = parts[1]
    width = parts[2]
    buffer = parts[3]
    crs = parts[4]
    return (minx, miny, width, buffer, crs)


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


def stitch_crowns(folder: str, shift: int = 1, max_workers=4, logger=None):
    """
    Stitch together predicted crowns from multiple geojson files, applying a spatial filter.

    Args:
        folder: Path to folder containing geojson files.
        shift: Number of meters to shift the size of the bounding box by to avoid edge crowns.
        max_workers: Maximum number of threads for parallel processing.
        logger: Logger object for logging messages.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing all the crowns.
    """
    crowns_path = Path(folder)
    files = list(crowns_path.glob("*geojson"))

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
        print(confidence)
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
    # Ensure that the forest boundary geometries are valid
    if not all(forest_boundary.is_valid):
        if logger:
            logger.error("Invalid geometries detected in the forest boundary. Attempting to fix.")
        forest_boundary = forest_boundary.buffer(0)  # A common fix for invalid geometries

    for top_folder in [urban_fold]:
        # For each folder, process corresponding urban and forest GeoJSONs
        for name in os.listdir(top_folder):
            if not name.endswith(".geojson"):
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
                    logger.info(f"Fusing predictions for tile {name}...")

                output_path = os.path.join(output_dir, os.path.basename(name))
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                if os.path.isfile(output_path):
                    logger.debug(f"file {name} already processed")
                    continue

                # Read the urban and forest predictions
                urban_shapes = gpd.read_file(urban_geojson_path)
                forest_shapes = gpd.read_file(forest_geojson_path)
                # Ensure CRS is the same for all geometries
                if not urban_shapes.crs == forest_shapes.crs == forest_boundary.crs:
                    if logger:
                        logger.warning("CRS mismatch detected. Aligning CRS to match the forest boundary.")
                    urban_shapes = urban_shapes.to_crs(forest_boundary.crs)
                    forest_shapes = forest_shapes.to_crs(forest_boundary.crs)
                # Simplify geometries before clipping and overlaying
                urban_shapes['geometry'] = urban_shapes.simplify(tolerance=0.1, preserve_topology=True)
                forest_shapes['geometry'] = forest_shapes.simplify(tolerance=0.1, preserve_topology=True)

                # Clip the forest shapes only to the extent of the urban shapes
                urban_bounds = urban_shapes.total_bounds
                forest_clipped = forest_boundary.cx[urban_bounds[0]:urban_bounds[2], urban_bounds[1]:urban_bounds[3]]

                # Clip the forest shapes to be within the forest boundary
                forest_clipped = gpd.clip(forest_shapes, forest_clipped)
                # Exclude areas inside the forest boundary from the urban shapes
                urban_outside_forest = gpd.sjoin(urban_shapes, forest_clipped, how="left")

                # Filter out rows where 'index_right' is NaN (i.e., areas outside the forest boundary)
                urban_outside_forest = urban_outside_forest[urban_outside_forest['index_right'].isna()].drop(
                    columns='index_right')

                # Create a new 'Confidence_score' by prioritizing 'Confidence_score_left'
                urban_outside_forest['Confidence_score'] = urban_outside_forest['Confidence_score_left'].fillna(
                    urban_outside_forest['Confidence_score_right'])

                # Now that we have 'Confidence_score', we can drop 'Confidence_score_left' and 'Confidence_score_right'
                urban_outside_forest = urban_outside_forest.drop(
                    columns=['Confidence_score_left', 'Confidence_score_right'])

                # Handle any remaining NaN values in 'Confidence_score' by replacing them with a default (e.g., 0)
                urban_outside_forest['Confidence_score'] = urban_outside_forest['Confidence_score'].fillna(0)

                # Combine the clipped forest shapes and urban shapes outside the forest
                fused_shapes = pd.concat([forest_clipped, urban_outside_forest], ignore_index=True)
                # Ensure the fused geometries are valid
                if not all(fused_shapes.is_valid):
                    if logger:
                        logger.warning(
                            f"Invalid geometries detected in fused shapes for tile {name}. Attempting to fix.")
                    fused_shapes['geometry'] = fused_shapes.buffer(0)  # Fix invalid geometries

                # Save the fused result as a new GeoJSON file
                fused_shapes.to_file(output_path, driver="GeoJSON")


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
