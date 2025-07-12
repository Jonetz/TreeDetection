import os
import datetime
import time

from TreeDetection.helpers import process_and_stitch_predictions
from TreeDetection.config import Config
from TreeDetection.detection import postprocess_files, preprocess_files, predict_on_model

from SolarDetection.config import get_config
from SolarDetection.solar_postprocessing import process_files_in_directory

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
def predict_tiles(config):
    logger = config["logger"]
        
    logger.info("Only Combined Model is given. Starting prediction...")

    folder = os.path.join(config["output_directory"], "geojson_predictions")
    # Predict the tiles using the urban modelasyncio.run(predict_on_model(config, config["urban_model"], config["tiles_path"], config["output_path"]))
    logger.info(f'Starting prediction with model {config["combined_model"]}...')
    start = time.time()
    predict_on_model(config, config["combined_model"], config["tiles_path"],
                        os.path.join(config["output_directory"], "predictions"), batch_size=config["batch_size"], exclude_vars=["only_forest"])
    end = time.time()
    predict_on_model_duration = end - start

    # Process and stitch predictions for the urban model
    start = time.time()
    process_and_stitch_predictions(
        tiles_path=config["tiles_path"],
        pred_fold=os.path.join(config["output_directory"], "predictions"),
        output_path=folder,
        max_workers=config["num_workers"],
        shift=0,
        simplify_tolerance=config['simplify_tolerance'],
        logger=config["logger"]
    )
    end = time.time()
    process_and_stitch_predictions_duration = end - start

    logger.info("Predictions have been processed and stitched. Begin fusing the predictions.")

    logger.debug(f"Prediction took {predict_on_model_duration} seconds")
    logger.debug(f"process and stitch predictions for urban took {process_and_stitch_predictions_duration} seconds")
        
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
    config, _ = get_config("solar_config.yml")
    process_files(config)