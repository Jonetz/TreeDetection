import asyncio
import datetime
import os
import re
import sys
import time
import warnings
from pathlib import Path

import rasterio

from TreeDetection.prediction import Predictor

from TreeDetection.config import get_config, setup_model_cfg, Config
from TreeDetection.preprocessing import tile_data
from TreeDetection.helpers import retrieve_neighboring_image_filenames, merge_images, crop_image, tif_geoinfo
from TreeDetection.helpers import process_and_stitch_predictions, fuse_predictions, exclude_outlines
from TreeDetection.postprocessing import process_files_in_directory

import geopandas as gpd
import shutil

gpd.options.display_precision = 2

def postprocess_files(config):
    """
    postprocess the files according to the configuration.
    """
    config_obj = Config()
    config_obj._load_into_config(config)
    logger = config["logger"]
    logger.info("Postprocessing the predictions.")
    filename_pattern = (config.get('image_regex', "(\\d+)\\.tif"), config.get('height_data_regex', "(\\d+)\\.tif"))
    #1. Filter with exclude outlines
    logger.info("Excluding Outlines.")
    exclude_outlines(config)

    # 2. Filter with post-processing rules
    process_files_in_directory(os.path.join(config["output_directory"], 'geojson_predictions'),
                               config['height_data_path'],
                               config['image_directory'],
                               parallel=False,
                               filename_pattern=filename_pattern)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # 4. Save the final predictions as gpkg in another folder
    for file in os.listdir(os.path.join(config["output_directory"], 'geojson_predictions')):
        if not (file.endswith('.geojson') or file.endswith('.gpkg')) or not file.startswith('processed_'):
            continue
        crowns = gpd.read_file(os.path.join(config["output_directory"], 'geojson_predictions', file))
        logger.debug(f" File {file}, # crowns {len(crowns)} ")
        # If the option for timestamps is set in config.yml, we change the filename to include the timestamp
        filename_without_processed = file.replace('processed_', '')
        if config["timestamped_output_directory"]:
            timestamp_directory = f"{config['output_directory']}/{timestamp}"
            os.makedirs(timestamp_directory, exist_ok=True)
            crowns.to_file(os.path.join(timestamp_directory, filename_without_processed))
            crowns.to_file(os.path.join(config["output_directory"], filename_without_processed))
        else:
            crowns.to_file(os.path.join(config["output_directory"], filename_without_processed))


def predict_on_model(config, model_path, tiles_path, output_path, batch_size=10, exclude_vars=None):
    """
    Predict the tiles according to the configuration using mixed precision and parallel inference.
    
    Args:
        config (dict): Detectron Configuration dictionary for the model.
        model_path (str): Path to the model.
        tiles_path (str): Path to the directory containing the tiles.
        output_path (str): Path to the output directory.
        batch_size (int): Number of images processed simultaneously.
        exclude_vars (list): List of variables to exclude from tile metadata. Default is None.
    """
    logger = config.get("logger", None)

    # Check paths
    for path, name in [(model_path, "Model file"), (tiles_path, "Tiles directory")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
        if name == "Tiles directory" and not os.path.isdir(path):
            raise NotADirectoryError(f"{name} is not a directory: {path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Initialize model configuration and predictor with the exclude_vars flag
    cfg = setup_model_cfg(update_model=model_path, device=config["device"])
    predictor = Predictor(cfg, device_type=config["device"], max_batch_size=batch_size, output_dir=output_path,
                          exclude_vars=exclude_vars)

    # Collect all TIF files
    images_directory = Path(config["image_directory"])
    images_paths = [str(f) for f in images_directory.glob("*.tif")]

    merged_directory = Path(f"{images_directory}/{config['merged_path']}")
    merged_images = [str(f) for f in merged_directory.glob("*.tif")]
    images_paths.extend(merged_images)

    if not images_paths:
        logger.warning("No TIF files found for prediction.")
        return

    # Parallel processing with asyncio
    def process_image(file_path):
        tile_dir = os.path.join(tiles_path, os.path.basename(file_path).replace('.tif', ''))
        os.makedirs(tile_dir, exist_ok=True)

        try:
            predictor(file_path, tile_dir)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Launch tasks
    for fp in images_paths:
        process_image(fp)

    logger.info(f"Completed prediction for {len(images_paths)} images.")


def predict_tiles(config):
    """
    Predict the tiles according to the configuration.
    """
    config_obj = Config()
    config_obj._load_into_config(config)
    logger = config["logger"]

    # 1. If urban model is available, predict the tiles using the urban model
    if "urban_model" in config and "forrest_model" in config and "forrest_outline" in config and\
            config["urban_model"] and os.path.exists(config["urban_model"]) and \
            config["forrest_model"] and os.path.exists(config["forrest_model"]) and \
            config["forrest_outline"] and os.path.exists(config["forrest_outline"]):
        logger.info("Urban, forrest models and forrest outline are available. Starting prediction...")

        urban_fold = os.path.join(config["output_directory"], "urban_geojson")
        forrest_fold = os.path.join(config["output_directory"], "forrest_geojson")
        # Predict the tiles using the urban modelasyncio.run(predict_on_model(config, config["urban_model"], config["tiles_path"], config["output_path"]))
        logger.info(f'Starting prediction with model {config["urban_model"]}...')
        start = time.time()
        predict_on_model(config, config["urban_model"], config["tiles_path"],
                         os.path.join(config["output_directory"], "urban_predictions"), batch_size=config["batch_size"],
                         exclude_vars=["only_forest"])
        end = time.time()
        predict_on_model_urban_duration = end - start
        # Predict the tiles using the forrest model
        logger.info(f'Starting prediction with model {config["forrest_model"]}...')
        start = time.time()
        predict_on_model(config, config["forrest_model"], config["tiles_path"],
                         os.path.join(config["output_directory"], "forrest_predictions"),
                         batch_size=config["batch_size"], exclude_vars=["only_urban"])
        end = time.time()
        predict_on_model_forrest_duration = end - start
        # Process and stitch predictions for the urban model
        start = time.time()
        process_and_stitch_predictions(
            tiles_path=config["tiles_path"],
            pred_fold=os.path.join(config["output_directory"], "urban_predictions"),
            output_path=urban_fold,
            max_workers=config["num_workers"],
            shift=1,
            simplify_tolerance=config['simplify_tolerance'],
            logger=config["logger"],
            # verbose=config["verbose"]
        )
        end = time.time()
        process_and_stitch_predictions_urban_duration = end - start

        # Process and stitch predictions for the forest model
        start = time.time()
        process_and_stitch_predictions(
            tiles_path=config["tiles_path"],
            pred_fold=os.path.join(config["output_directory"], "forrest_predictions"),
            output_path=forrest_fold,
            max_workers=config["num_workers"],
            shift=1,
            simplify_tolerance=config['simplify_tolerance'],
            logger=config["logger"],
            # verbose=config["verbose"]
        )
        end = time.time()
        process_and_stitch_predictions_forrest_duration = end - start

        logger.info("Predictions have been processed and stitched. Begin fusing the predictions.")

        # Step 4: Fusion based on forest outline
        start = time.time()
        fuse_predictions(
            urban_fold,
            forrest_fold,
            config["forrest_outline"],
            os.path.join(config["output_directory"], 'geojson_predictions'),
            logger=config["logger"]
        )
        end = time.time()
        fuse_predictions_duration = end - start
        logger.info("Fusion based on forest outline has been completed.")

        logger.debug(f"predict on model for urban took {predict_on_model_urban_duration} seconds")
        logger.debug(f"predict on model for forrest took {predict_on_model_forrest_duration} seconds")
        logger.debug(f"process and stitch predictions for urban took {process_and_stitch_predictions_urban_duration} seconds")
        logger.debug(f"process and stitch predictions for forrest took {process_and_stitch_predictions_forrest_duration} seconds")
        logger.debug(f"fuse prediction took {fuse_predictions_duration} seconds")


    elif "combined_model" in config and config["combined_model"] and os.path.exists(config["combined_model"]):
        logger.info("Only Combined Model is given. Starting prediction...")

        folder = os.path.join(config["output_directory"], "geojson_predictions")
        # Predict the tiles using the urban modelasyncio.run(predict_on_model(config, config["urban_model"], config["tiles_path"], config["output_path"]))
        logger.info(f'Starting prediction with model {config["combined_model"]}...')
        start = time.time()
        predict_on_model(config, config["combined_model"], config["tiles_path"],
                         os.path.join(config["output_directory"], "predictions"), batch_size=config["batch_size"])
        end = time.time()
        predict_on_model_duration = end - start

        # Process and stitch predictions for the urban model
        start = time.time()
        process_and_stitch_predictions(
            tiles_path=config["tiles_path"],
            pred_fold=os.path.join(config["output_directory"], "predictions"),
            output_path=folder,
            max_workers=config["num_workers"],
            shift=1,
            simplify_tolerance=config['simplify_tolerance'],
            logger=config["logger"]
        )
        end = time.time()
        process_and_stitch_predictions_duration = end - start

        logger.info("Predictions have been processed and stitched. Begin fusing the predictions.")

        logger.debug(f"Prediction took {predict_on_model_duration} seconds")
        logger.debug(f"process and stitch predictions for urban took {process_and_stitch_predictions_duration} seconds")
    else:
        raise FileNotFoundError(
            "No model available for prediction. Either urban model or forrest model + outline or combined model must be available.")


def preprocess_files(config):
    """
    Preprocess the files according to the configuration.
    """
    config_obj = Config()
    config_obj._load_into_config(config)
    logger = config["logger"]

    # 1. Read from the dictionary
    images_directory = config["image_directory"]
    height_data_directory = config["height_data_path"]
    if not os.path.exists(images_directory):
        raise FileNotFoundError(f"Image directory not found: {images_directory}")
    if not os.path.isdir(images_directory):
        raise NotADirectoryError(f"Image directory is not a directory: {images_directory}")
    if not os.path.exists(height_data_directory):
        raise FileNotFoundError(f"Height directory not found: {height_data_directory}")
    if not os.path.isdir(height_data_directory):
        raise NotADirectoryError(f"Height directory is not a directory: {height_data_directory}")

    # Collect all TIF files in the directory
    images_paths = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if f.endswith('.tif')]
    height_paths = [os.path.join(height_data_directory, f) for f in os.listdir(height_data_directory) if
                    f.endswith('.tif')]

    # Remove files that have already been processed
    if os.path.exists(config["continue"]):
        with open(config["continue"], 'r') as f:
            continue_files = f.read().splitlines()
        images_paths = [f for f in images_paths if f not in continue_files]

    # Regex patterns
    image_regex_pattern = re.compile(config["image_regex"])
    height_data_regex_pattern = re.compile(config["height_data_regex"])

    # Filter image paths based on the `image_regex` pattern applied to the base name
    images_paths = [f for f in images_paths if image_regex_pattern.search(os.path.basename(f))]

    # Filter for height data files based on base names
    height_data_paths = [f for f in height_paths if height_data_regex_pattern.search(os.path.basename(f))]

    # Process image identifiers
    image_identifiers = {}
    for f in images_paths:
        match = image_regex_pattern.search(os.path.basename(f))
        if match:
            # Smash all groups together without any separator
            image_identifiers["".join(match.groups())] = f

    # Process height data identifiers
    height_data_identifiers = {}
    for f in height_data_paths:
        match = height_data_regex_pattern.search(os.path.basename(f))
        if match:
            # Smash all groups together without any separator
            height_data_identifiers["".join(match.groups())] = f

    if config["use_overlap"]:
        # Filter out images that have already been processed (can be identified by the __ in the filename)
        images_paths = [path for path in images_paths if "__" not in path]
        height_paths = [path for path in height_paths if "__" not in path]

        merged_directory = config["merged_path"]
        def save_cropped_images(images_path, rgbi=True):
            """
            Save the cropped images based on the neighboring images.

            Args:
                images_path (list): List of image paths.
            """
            cropped_image_names = []
            for f in images_path:
                left, right, up, down = retrieve_neighboring_image_filenames(f, images_path)

                directory = os.path.dirname(f)
                result_directory = f"{directory}/{merged_directory}"
                os.makedirs(result_directory, exist_ok=True)
                f_basename = os.path.basename(f).replace(".tif", "").split("_")[0]
                f_name_end = os.path.basename(f).replace(".tif", "").split("_")[-1]
                # f_x_coord, f_y_coord = tif_geoinfo(f)
                transform, crs, width, height = tif_geoinfo(f)
                f_x_coord, f_y_coord = transform.c, transform.f

                # We only look at the right and the bottom neighbors so that we don't process the same cropped image twice
                if right is not None:
                    transform, crs, width, height = tif_geoinfo(right)
                    right_x_coord, right_y_coord = transform.c, transform.f

                    with rasterio.open(f) as left_img, rasterio.open(f"{right}") as right_img:
                        merged_img, merged_img_meta = merge_images(left_img, right_img)

                        # Write the merged file only to memory for faster processing
                        with rasterio.MemoryFile() as memfile:
                            with memfile.open(**merged_img_meta) as merged_src:
                                merged_src.write(merged_img)
                                if rgbi:
                                    output_filename = f"{f_basename}_{round(f_x_coord)}_{round(f_y_coord)}_{round(right_x_coord)}_{round(right_y_coord)}_{f_name_end}.tif"
                                else:
                                    output_filename = f"{f_basename}_{round(f_x_coord)}{round(f_y_coord)}{round(right_x_coord)}{round(right_y_coord)}_{f_name_end}.tif"
                                # Perform cropping here
                                cropped_data, cropped_meta = crop_image(merged_src,
                                                                        (config["tile_width"] + 2 * config["buffer"]) *
                                                                        config["overlapping_tiles_width"],
                                                                        merged_src.height)
                                # Save the cropped image
                                with rasterio.open(f"{result_directory}/{output_filename}", "w", **cropped_meta) as dest:
                                    dest.write(cropped_data)

                                cropped_image_names.append(f"{result_directory}/{output_filename}")

                if down is not None:
                    transform, crs, width, height = tif_geoinfo(down)
                    down_x_coord, down_y_coord = transform.c, transform.f

                    with rasterio.open(f) as top_img, rasterio.open(f"{down}") as bottom_img:
                        merged_img, merged_img_meta = merge_images(top_img, bottom_img)

                        # Write the merged file only to memory for faster processing
                        with rasterio.MemoryFile() as memfile:
                            with memfile.open(**merged_img_meta) as merged_src:
                                merged_src.write(merged_img)
                                if rgbi:
                                    output_filename = f"{f_basename}_{round(f_x_coord)}_{round(f_y_coord)}_{round(down_x_coord)}_{round(down_y_coord)}_{f_name_end}.tif"
                                else:
                                    output_filename = f"{f_basename}_{round(f_x_coord)}{round(f_y_coord)}{round(down_x_coord)}{round(down_y_coord)}_{f_name_end}.tif"
                                # Perform cropping here
                                cropped_data, cropped_meta = crop_image(merged_src, merged_src.width,
                                                                        (config["tile_height"] + 2 * config["buffer"]) *
                                                                        config["overlapping_tiles_height"])
                                # Save the cropped image
                                with rasterio.open(f"{result_directory}/{output_filename}", "w", **cropped_meta) as dest:
                                    dest.write(cropped_data)

                                cropped_image_names.append(f"{result_directory}/{output_filename}")

            return cropped_image_names

        # Here we merge and crop neighboring images
        try:
            cropped_image_filenames = save_cropped_images(images_paths, rgbi=True)
            cropped_height_filenames = save_cropped_images(height_paths, rgbi=False)

            # Include the image paths of the cropped images to the list of images to be processed
            images_paths.extend(cropped_image_filenames)
            height_paths.extend(cropped_height_filenames)
        except Exception as e:
            logger.error(f"Error merging and cropping images: {e}")

    # Validate height data availability
    missing_height_data = []
    for identifier, image_path in image_identifiers.items():
        if identifier not in height_data_identifiers:
            warnings.warn(f"Warning: No corresponding height data found for image file {image_path}")
            missing_height_data.append(image_path)

    if not images_paths:
        raise FileNotFoundError(
            f"No image TIF-files matching the pattern found in the directory: {images_directory} or all files have already been processed.")

    # Continue with tiling if there are valid images
    logger.info(f"Found {len(images_paths)} images for processing. Starting tiling...")
    if images_paths:
        asyncio.run(
            tile_data(images_paths, config["tiles_path"], config["buffer"], config["tile_width"], config["tile_height"],
                      parallel=False,
                      max_workers=config["num_workers"], logger=config["logger"],
                      forest_shapefile=config.get("forrest_outline", None)))

    return images_paths


def process_files(config):
    """
    Process the files according to the configuration.
    """
    logger = config["logger"]
    config_obj = Config()
    config_obj._load_into_config(config)

    start = time.time()
    # Read the files and tile them
    preprocess_files(config)
    end = time.time()
    preprocess_files_duration = end - start

    start = time.time()
    # Predict the tiles
    predict_tiles(config)
    end = time.time()
    predict_tiles_duration = end - start

    start = time.time()
    # Post-process the predictions
    postprocess_files(config)
    end = time.time()
    postprocess_files_duration = end - start

    cleanup_files(config)

    # Print stats about the processing
    logger.debug(f"preprocess step took {preprocess_files_duration} seconds. ")
    logger.debug(f"predict step took {predict_tiles_duration} seconds. ")
    logger.debug(f"postprocess step took {postprocess_files_duration} seconds. ")

def cleanup_files(config):
    if not config.get('keep_intermediate', False):
        try:
            shutil.rmtree(config["tiles_path"])  # Remove the tiles directory
            shutil.rmtree(config["image_directory"] + "/" + config["merged_path"])  # Remove the merged image directory
            shutil.rmtree(config["height_data_path"] + "/" + config["merged_path"])  # Remove the merged tile directory
        except FileNotFoundError:
            pass

        # Remove merged/cropped files in height / image directory
        for file in os.listdir(config["image_directory"]):
            if "__" in file:
                os.remove(os.path.join(config["image_directory"], file))

        for file in os.listdir(config["height_data_path"]):
            if "__" in file:
                os.remove(os.path.join(config["height_data_path"], file))


    for folder in os.listdir(config["output_directory"]):
        folder = os.path.join(config["output_directory"], folder)
        keep_folders = ["logs"]
        if os.path.isdir(folder) and os.path.basename(folder) not in keep_folders and not config.get(
                'keep_intermediate', False):
            shutil.rmtree(folder)

if __name__ == "__main__":
    config, _ = get_config("config.yml")

    # Print Information about the configuration

    # Start reading the files and validate the configuration

    # Start the processing
    process_files(config)
