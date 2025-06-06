import os, shutil, sys
from datetime import datetime
from detectree2.preprocessing.tiling import tile_data_train, to_traintest_folders
from detectree2.models.train import register_train_data, setup_cfg, MyTrainer
import rasterio
import geopandas as gpd
from glob import glob
from shapely.geometry import Polygon, MultiPolygon
import logging
import argparse
import json


def setup_logging(params):
    """
    Sets up the logging configuration.

    Args:
        params (dict): A dictionary containing parameters including 'site_path'
                       where the log file will be stored.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(params['site_path'], 'logs.log'),
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.DEBUG
    )
    return logger


def get_params(args):
    """
    Extracts and returns parameters from parsed command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary of parameters used throughout the script.
    """
    return {
        'site_path': args.site_path,
        'image_path': os.path.join(args.site_path, 'rgb'),
        'mask_path': os.path.join(args.site_path, 'masks'),
        'json_path': os.path.join(args.site_path, 'crowns'),
        'tile_path': os.path.join(args.site_path, 'tiles'),
        'buffer': args.buffer,
        'tile_width': args.tile_width,
        'tile_height': args.tile_height,
        'test_frac': 0.15,
        'val_fold': args.val,
        'threshold': 0.0,
        'update_model': args.update_model
    }


def preprocess_data(params, logger):
    """
    Preprocesses input image data by creating tiles for training.

    This function performs the following tasks:
    - Sets up the tile directory by removing any existing one.
    - Iterates through TIFF files in the image directory.
    - Finds the corresponding annotation file (JSON or GPKG) for each image.
    - Reads and filters geometries from the annotation file.
    - Tiles the image data if valid geometries exist.
    - Splits the tiled data into training and testing folders.

    Args:
        params (dict): Dictionary of parameters.
        logger (logging.Logger): Logger for logging debug and error messages.

    Raises:
        SystemExit: Exits the program if there is an error in setting up the tile directory.
    """
    try:
        # Remove and (re)create the tile directory
        if os.path.exists(params['tile_path']):
            shutil.rmtree(params['tile_path'])
        os.makedirs(params['tile_path'], exist_ok=True)
    except Exception as e:
        logger.exception("Failed to set up tile directory")
        print("Error setting up the tile directory. Please check the log file for details.")
        sys.exit(1)

    for rgb_file in glob(os.path.join(params['image_path'], '*.tif')):
        try:
            filename_withending = os.path.basename(rgb_file)
            filename = os.path.splitext(filename_withending)[0]

            # Locate the corresponding JSON or GPKG file
            if os.path.exists(os.path.join(params['json_path'], filename + '.gpkg')):
                json_file = os.path.join(params['json_path'], filename + '.gpkg')
            elif os.path.exists(os.path.join(params['json_path'], filename + '.json')):
                json_file = os.path.join(params['json_path'], filename + '.json')
            else:
                print('Could not find corresponding annotation file for:', filename)
                logger.warning(f"Annotation file not found for {filename}")
                continue

            # Read the TIFF file
            data = rasterio.open(rgb_file)

            # Read the crowns data and filter valid geometries
            crowns = gpd.read_file(json_file)
            crowns = crowns[~crowns.is_empty]
            crowns = crowns[crowns.is_valid]
            crowns = crowns[crowns.geometry.apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))]

            if not crowns.empty:
                tile_data_train(
                    data, params['tile_path'], params['buffer'],
                    params['tile_width'], params['tile_height'], crowns, params['threshold']
                )
            else:
                print("Skipping file due to no valid geometries:", rgb_file)
                logger.info(f"Skipping {rgb_file} as no valid geometries were found.")
                continue

            # Split data into training and testing sets
            if params['val_fold'] is not None and params['val_fold'] > 1:
                to_traintest_folders(
                    params['tile_path'], params['tile_path'], test_frac=params['test_frac'],
                    strict=True, folds=params['val_fold']
                )
            else:
                to_traintest_folders(
                    params['tile_path'], params['tile_path'], test_frac=params['test_frac'],
                    strict=True
                )
        except Exception as e:
            logger.exception(f"Error processing file {rgb_file}")
            print(f"An error occurred processing file {rgb_file}. Skipping this file.")
            continue


def train_model(params, logger):
    """
    Trains the model using the preprocessed tile data.

    This function performs the following tasks:
    - Registers the training data.
    - Sets up a unique output directory for the training run.
    - Configures the model based on whether an update model exists.
    - Initializes and runs the training process.

    Args:
        params (dict): Dictionary of parameters.
        logger (logging.Logger): Logger for logging debug and error messages.

    Raises:
        SystemExit: Exits the program if there is an error during model training.
    """
    try:
        out_path = os.path.join(params['tile_path'], "train")
        register_train_data(out_path, 'BW', params['val_fold'])
        register_train_data("BW/tiles/train", 'BW', params['val_fold'])

        # Define the model configuration
        base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        trains = ("BW_train",)  # Registered train data
        tests = ("BW_val",)  # Registered validation data

        base_out_dir = "train_output"
        current_date = datetime.now().strftime("%Y%m%d")
        run_number = 1

        # Create a unique output directory for this run
        while True:
            out_dir = os.path.join(base_out_dir, f"{current_date}_{run_number:02d}")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                break
            run_number += 1

        if params['update_model'] and os.path.exists(params['update_model']):
            cfg = setup_cfg(
                base_model, trains, tests, workers=16,
                update_model=params['update_model'], ims_per_batch=9, gamma=0.1213,
                backbone_freeze=3, warm_iter=72, batch_size_per_im=1209,
                base_lr=0.005, max_iter=2000, num_classes=1, eval_period=100, out_dir=out_dir
            )
        else:
            cfg = setup_cfg(
                base_model, trains, tests, workers=4, ims_per_batch=4, gamma=0.01,
                backbone_freeze=3, warm_iter=72, batch_size_per_im=512,
                base_lr=0.01, max_iter=2000, num_classes=1, eval_period=100, out_dir=out_dir
            )

        trainer = MyTrainer(cfg, patience=10)
        trainer.resume_or_load(resume=False)
        trainer.train()
    except Exception as e:
        logger.exception("Error during model training")
        print("An error occurred during model training. Please check the log file for details.")
        sys.exit(1)


def main(args):
    """
    Main entry point for the script.

    This function initializes parameters, logging, and executes the
    preprocessing and training steps. It catches unexpected errors and
    informs the user to check the log file for further details.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        SystemExit: Exits the program if an unexpected error occurs.
    """
    try:
        print('Starting Main')
        params = get_params(args)
        print('Loaded Params')
        logger = setup_logging(params)
        print('Setup Logging')
        preprocess_data(params, logger)
        print('Preprocessed Data')
        train_model(params, logger)
    except Exception as e:
        logging.exception("An unexpected error occurred in the main function")
        print("An unexpected error occurred. Please check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    """
    Parses command-line arguments and starts the training process.

    Exits the program if there is a critical error during argument parsing or execution.
    """
    try:
        parser = argparse.ArgumentParser(description="Tree Detection Training Script")
        parser.add_argument("--site_path", type=str, required=True, help="Path to the site directory")
        parser.add_argument("--buffer", type=int, default=0, help="Buffer size for tiling")
        parser.add_argument("--tile_width", type=int, default=50, help="Width of the tile")
        parser.add_argument("--tile_height", type=int, default=50, help="Height of the tile")
        parser.add_argument("--val", type=int, default=2, help="Validation Fold Count")
        parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for tiling")
        parser.add_argument("--update_model", type=str, default="", help="Path to an update model file")
        args = parser.parse_args()
        main(args)
    except Exception as e:
        print("A critical error occurred. Please check your input parameters and the log file for more details.")
        sys.exit(1)