import sys
import os
import re
import warnings

import torch

# Add the root project directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config, setup_model_cfg
from tiling import tile_data
from helpers import get_filenames, project_to_geojson, stitch_crowns, clean_crowns, fuse_predictions, delete_contents, RoundedFloatEncoder, exclude_outlines
from post_performant import process_files_in_directory

from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, cv2
import geopandas as gpd
import shutil
import typing
from torch.amp import autocast
from shapely.geometry import box

gpd.options.display_precision = 2

def postprocess_files(config):
    """
    postprocess the files according to the configuration.
    """
    logger = config["logger"]
    logger.info("Postprocessing the predictions.")
    filename_pattern = (config.get('image_regex', "(\\d+)\\.tif"), config.get('height_data_regex', "(\\d+)\\.tif"))
    
    # 1. Filter with exclude outlines
    logger.info("Excluding Outlines.")
    exclude_outlines(config)

    # 2. Filter with post-processing rules 
    process_files_in_directory(os.path.join(config["output_directory"], 'geojson_predictions'), config['height_data_path'],\
                                confidence_threshold=config['confidence_threshold'], containment_threshold=config['containment_threshold'],\
                                parallel=False, filename_pattern=filename_pattern)

    # 4. Save the final predictions as gpkg in another folder 
    for file in os.listdir(os.path.join(config["output_directory"], 'geojson_predictions')):
        if not (file.endswith('.geojson') or file.endswith('.gpkg')) or file.startswith('processed_'):
            continue
        crowns = gpd.read_file(os.path.join(config["output_directory"], 'geojson_predictions', file))
        logger.debug(f" File {file}, # crowns {len(crowns)} ")
        crowns.to_file(os.path.join(config["output_directory"], file.replace('processed_', '')))

def predict_on_model(config, model_path, tiles_path, output_path, batch_size=50):
    """
    Predict the tiles according to the configuration using mixed precision and parallel inference.
    
    Args:
        cfg (CfgNode): Detectron Configuration node for the model.
        model_path (str): Path to the model.
        tiles_path (str): Path to the directory containing the tiles.
        output_path (str): Path to the output directory.
        batch_size (int): Number of images processed simultaneously.
        max_workers (int): Maximum number of threads for parallel inference.
    """
    logger = config.get("logger", None)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tiles_path):
        raise FileNotFoundError(f"Tiles directory not found: {tiles_path}")
    if not os.path.isdir(tiles_path):
        raise NotADirectoryError(f"Tiles directory is not a directory: {tiles_path}")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path) or not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    else:
        delete_contents(output_path, logger=logger)

    # Initialize the config with updated model weights and device
    cfg = setup_model_cfg(update_model=model_path, device=config["device"])

    # Initialize predictor
    predictor = DefaultPredictor(cfg)

    # Load dataset (assuming get_filenames is a helper function to load image paths)
    dataset_dicts = get_filenames(tiles_path)

    def process_image(d):
        """Helper function to process each image."""
        img = cv2.imread(d["file_name"])

        # Use mixed precision inference
        try:
            with autocast(device_type=config["device"]):
                outputs = predictor(img)
        except Exception as e:
            logger.error(f"Error processing {d['file_name']}: {e}")
            return f"Error processing: {d['file_name']}"

        # Generate output file name
        file_name_path = d["file_name"]

        # Extract the image name from the path
        image_name = os.path.basename(os.path.dirname(file_name_path))  # Get the folder name as imagename
        tile_name = os.path.basename(file_name_path).replace("png", "json")  # Replace .png with .json

        # Create corresponding subdirectory in output_path for predictions
        pred_subdir = os.path.join(output_path, image_name)  # Set output path to include imagename
        os.makedirs(pred_subdir, exist_ok=True)  # Ensure subdirectory exists

        output_file = os.path.join(pred_subdir, f"Prediction_{tile_name}")  # Full path for the output file

        # Save predictions as COCO JSON format
        evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), d["file_name"])
        with open(output_file, "w") as dest:
            json.dump(evaluations, dest, cls=RoundedFloatEncoder, separators=(',', ':'))
        return f"Processed: {file_name_path}"

    # Use ThreadPoolExecutor for parallel inference
    total_files = len(dataset_dicts)
    max_workers = config.get("num_workers", 1)
    logger.info(f"Predicting {total_files} images  from {tiles_path} using {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, d) for d in dataset_dicts]
        # Optionally: print progress as batches are processed
        for i, future in enumerate(futures, start=1):
            if i % batch_size == 0 and config.get("verbose", False) and logger:
                logger.debug(f"File {i}/{total_files}: {future.result()}")

    if config.get("verbose", False) and logger:
        logger.info(f"Processed {total_files}/{total_files} images")


def predict_tiles(config):
    """
    Predict the tiles according to the configuration.
    """
    logger = config["logger"]

    # 1. If urban model is available, predict the tiles using the urban model
    if config["urban_model"] and os.path.exists(config["urban_model"]) and \
            config["forrest_model"] and os.path.exists(config["forrest_model"]) and \
            config["forrest_outline"] and os.path.exists(config["forrest_outline"]):

        urban_fold = os.path.join(config["output_directory"], "urban_geojson")
        forrest_fold = os.path.join(config["output_directory"], "forrest_geojson")
        # Predict the tiles using the urban model
        predict_on_model(config, config["urban_model"], config["tiles_path"],
                         os.path.join(config["output_directory"], "urban_predictions"))
        # Predict the tiles using the forrest model
        predict_on_model(config, config["forrest_model"], config["tiles_path"],
                         os.path.join(config["output_directory"], "forrest_predictions"))
        # Project the predictions back to the geographic space
        project_to_geojson(config["tiles_path"], pred_fold=os.path.join(config["output_directory"], "urban_predictions"), \
                           output_fold=urban_fold, max_workers=config["num_workers"], \
                            logger=config["logger"], verbose=config["verbose"])
        project_to_geojson(config["tiles_path"], pred_fold=os.path.join(config["output_directory"], "forrest_predictions"), \
                           output_fold=forrest_fold, max_workers=config["num_workers"], \
                            logger=config["logger"], verbose=config["verbose"])
        
        logger.info("Predictions have been saved to the output directory. Begin stitching the tiles.")
        # Stitch the predictions together
        for top_folder in [urban_fold, forrest_fold]:
            forrest_folders = [f for f in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, f))]
            for folder in forrest_folders:
                output_path = os.path.join(top_folder, os.path.basename(folder) + ".gpkg")
                if os.path.isfile(output_path):
                    logger.debug(f"folder {folder} already processed for predicting tiles")
                    continue
                crowns = stitch_crowns(os.path.join(top_folder, folder), max_workers=config["num_workers"],
                                       logger=config["logger"], simplify_tolerance=config['simplify_tolerance'])
                # TODO Instead of using the clean crowns function use our own post-processing function
                # crowns = clean_crowns(crowns, iou_threshold=config["iou_threshold"], confidence=config["confidence_threshold_stitching"], logger=config["logger"])
                crowns.to_file(output_path, driver="GPKG")
        logger.info("Stitching has been completed. Begin fusing the predictions.")

        # Step 4: Fusion based on forest outline
        fuse_predictions(urban_fold, forrest_fold, config["forrest_outline"],
                         os.path.join(config["output_directory"], 'geojson_predictions'), logger=config["logger"])

        logger.info("Fusion based on forest outline has been completed.")

        # TODO: Save processed files for reprocessing avoidance
        # TODO Safe the filepaths of every processed file to the continue file to avoid reprocessing
    elif config["combined_model"] and os.path.exists(config["combined_model"]):
        predict_on_model(config, config["combined_model"], config["tiles_path"], config["output_directory"])
        geojson_fold = os.path.join(config["output_directory"], "geojson_fold")
        project_to_geojson(config["tiles_path"], pred_fold=config["output_directory"], output_fold=geojson_fold, max_workers=config["num_workers"], \
                            logger=config["logger"], verbose=config["verbose"])        
        folders = [f for f in os.listdir(geojson_fold) if os.path.isdir(os.path.join(geojson_fold, f))]
        for folder in folders:
            output_path = os.path.join(geojson_fold, os.path.basename(folder) + ".gpkg")
            if os.path.isfile(output_path):
                logger.debug(f"folder {folder} already processed for predicting tiles")
                continue
            crowns = stitch_crowns(os.path.join(geojson_fold, folder), max_workers=config["num_workers"],
                                    logger=config["logger"], simplify_tolerance=config['simplify_tolerance'])
        logger.info("Stitching has been completed.")
    else:
        raise FileNotFoundError(
            "No model available for prediction. Either urban model or forrest model + outline or combined model must be available.")


def preprocess_files(config):
    """
    Preprocess the files according to the configuration.
    """
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
    height_paths = [os.path.join(height_data_directory, f) for f in os.listdir(height_data_directory) if f.endswith('.tif')]

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

    # Validate height data availability
    missing_height_data = []
    for identifier, image_path in image_identifiers.items():
        if identifier not in height_data_identifiers:
            warnings.warn(f"Warning: No corresponding height data found for image file {image_path}")
            missing_height_data.append(image_path)

    if not images_paths:
        raise FileNotFoundError(f"No image TIF-files matching the pattern found in the directory: {images_directory} or all files have already been processed.")

    # Continue with tiling if there are valid images
    if images_paths:
        tile_data(images_paths, config["tiles_path"], config["buffer"], config["tile_width"], config["tile_height"], max_workers=config["num_workers"], logger=config["logger"])

    return images_paths


def process_files(config):
    """
    Process the files according to the configuration.
    """
    # Read the files and tile them
    #preprocess_files(config)

    # Predict the tiles
    #predict_tiles(config)

    # Post-process the predictions
    postprocess_files(config)

    shutil.rmtree(config["tiles_path"])  # Remove the tiles directory
    for folder in os.listdir(config["output_directory"]):
        folder = os.path.join(config["output_directory"], folder)
        keep_folders = ["processed_exclusions", "logs"]
        if os.path.isdir(folder) and os.path.basename(folder) not in keep_folders and not config.get('keep_intermediate', False):
            shutil.rmtree(folder)

    # Print stats about the processing


def profile_code(config, threshold = 0.05):
    """
    Profile the code to analyze performance using cProfile.

    
    20241024 Output:
        1    2.244    2.244   76.155   76.155 /home/jonas/TreeDetection/main.py:24(postprocess_files)
        1    0.000    0.000   19.579   19.579 /home/jonas/TreeDetection/main.py:211(preprocess_files)
        1    0.275    0.275  112.214  112.214 /home/jonas/TreeDetection/main.py:151(predict_tiles)

	    2    0.000    0.000   60.746   30.373 /home/jonas/TreeDetection/main.py:70(predict_on_model)
        2    0.001    0.000   26.859   13.429 /home/jonas/TreeDetection/helpers.py:266(stitch_crowns)        
	    1    0.000    0.000   19.579   19.579 /home/jonas/TreeDetection/tiling.py:109(tile_data)
        1    0.242    0.242   14.098   14.098 /home/jonas/TreeDetection/helpers.py:476(fuse_predictions)
        1    0.000    0.000   11.978   11.978 /home/jonas/TreeDetection/post_performant.py:511(process_files_in_directory)   
        2    0.001    0.000    9.288    4.644 /home/jonas/TreeDetection/helpers.py:71(project_to_geojson)
    """
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()

    # Start the processing
    process_files(config)

    pr.disable()

    s = io.StringIO()
    # Sort by cumulative time and apply the threshold
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(threshold)  # Only display functions above the cumulative time threshold

    logger = config.get("logger", None)
    if logger:
        logger.info(s.getvalue())
    else:
        print(s.getvalue())


if __name__ == "__main__":
    config = get_config("/home/jonas/TreeDetection/config.yml")

    # Print Information about the configuration

    # Start reading the files and validate the configuration

    # Start the processing
    #process_files(config)
    profile_code(config)
