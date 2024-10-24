import sys
import os
# Add the root project directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree_detection.config import get_config, setup_model_cfg
from tree_detection.tiling import tile_data
from tree_detection.helpers import get_filenames, project_to_geojson, stitch_crowns, clean_crowns, fuse_predictions, delete_contents
from tree_detection.post_performant import process_files_in_directory    

from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, cv2
import geopandas as gpd
import shutil
from torch.amp import autocast


def postprocess_files(config):
    """
    postprocess the files according to the configuration.
    """
    # 1. Filter with post-processing rules 
    process_files_in_directory(os.path.join(config["output_directory"], 'geojson_predictions'), config['height_data_path'],\
                                confidence_threshold=config['confidence_threshold'], containment_threshold=config['containment_threshold'],\
                                parallel=True)
    # 2. Filter with exclude outlines
    # 3. Filter with height threshold use post_performant
    # 4. Save the final predictions
    pass

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
            json.dump(evaluations, dest)
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
    # 1. If urban model is available, predict the tiles using the urban model
    if config["urban_model"] and os.path.exists(config["urban_model"]) and \
        config["forrest_model"] and os.path.exists(config["forrest_model"]) and \
        config["forrest_outline"] and os.path.exists(config["forrest_outline"]):
        
        logger = config["logger"]
        urban_fold = os.path.join(config["output_directory"], "urban_geojson")
        forrest_fold = os.path.join(config["output_directory"], "forrest_geojson")

        # Predict the tiles using the urban model
        predict_on_model(config, config["urban_model"], config["tiles_path"], os.path.join(config["output_directory"], "urban_predictions"))
        # Predict the tiles using the forrest model
        predict_on_model(config, config["forrest_model"], config["tiles_path"], os.path.join(config["output_directory"], "forrest_predictions"))

        # Project the predictions back to the geographic space
        project_to_geojson(config["tiles_path"], pred_fold=os.path.join(config["output_directory"], "urban_predictions"), \
                           output_fold=urban_fold, max_workers=config["num_workers"], \
                            logger=config["logger"], verbose=config["verbose"])
        project_to_geojson(config["tiles_path"], pred_fold=os.path.join(config["output_directory"], "forrest_predictions"), \
                           output_fold=forrest_fold, max_workers=config["num_workers"], \
                            logger=config["logger"], verbose=config["verbose"])
        
        # Delete the temporary files - includes predictions and tiles
        if not config['keep_intermediate'] and os.path.exists(os.path.join(config["output_directory"], "urban_predictions")):
            delete_contents(os.path.join(config["output_directory"], "urban_predictions"), logger=logger)
        if not config['keep_intermediate'] and os.path.exists(os.path.join(config["output_directory"], "forrest_predictions")):
            delete_contents(os.path.join(config["output_directory"], "forrest_predictions"), logger=logger)
        if not config['keep_intermediate'] and os.path.exists(config["tiles_path"]):
            delete_contents(config["tiles_path"], logger=logger)           

        logger.info("Predictions have been saved to the output directory. Begin stitching the tiles.")
        # Stitch the predictions together
        for top_folder in [urban_fold, forrest_fold]: 
            forrest_folders = [f for f in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, f))]
            for folder in forrest_folders:
                crowns = stitch_crowns(os.path.join(top_folder, folder), max_workers=config["num_workers"], logger=config["logger"])
                #TODO Instead of using the clean crowns function use our own post-processing function
                #crowns = clean_crowns(crowns, iou_threshold=config["iou_threshold"], confidence=config["confidence_threshold_stitching"], logger=config["logger"])
                basename = os.path.basename(folder)
                crowns.to_file(os.path.join(top_folder , basename + ".geojson"), driver="GeoJSON")
                if not config['keep_intermediate']:                    
                    delete_contents(os.path.join(top_folder, folder), logger=logger)

        logger.info("Stitching has been completed. Begin fusing the predictions.")

        # Step 4: Fusion based on forest outline
        fuse_predictions(urban_fold, forrest_fold, config["forrest_outline"], os.path.join(config["output_directory"], 'geojson_predictions'), logger=config["logger"])

        logger.info("Fusion based on forest outline has been completed.")

        # TODO: Save processed files for reprocessing avoidance
        # TODO Safe the filepaths of every processed file to the continue file to avoid reprocessing
    elif config["combined_model"] and os.path.exists(config["combined_model"]):
        pass
    else:
        raise FileNotFoundError("No model available for prediction. Either urban model or forrest model + outline or combined model must be available.")
def preprocess_files(config):
    """
    Preprocess the files according to the configuration.
    """
    # 1. Read from the dictionary   
    images_directory = config["image_directory"] 
    if not os.path.exists(images_directory):
        raise FileNotFoundError(f"Image directory not found: {images_directory}")
    if not os.path.isdir(images_directory):
        raise NotADirectoryError(f"Image directory is not a directory: {images_directory}")
    # Take all paths in the directory that end with .tif
    images_paths = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if f.endswith('.tif')]
    # Remove all paths that are in config["continue"], as they have already been processed
    if os.path.exists(config["continue"]):
        with open(config["continue"], 'r') as f:
            continue_files = f.read().splitlines()
        images_paths = [f for f in images_paths if f not in continue_files]
    if not images_paths:
        raise FileNotFoundError(f"No image TIF-files found in the directory: {images_directory} or all files have already been processed.")
    
    # 2. Tile the images
    tile_data(images_paths, config["tiles_path"], config["buffer"], config["tile_width"], config["tile_height"], max_workers=config["num_workers"], logger=config["logger"])

    # 3. Validate nDOM Availability
    #TODO: Implement nDOM availability check, warn if not available

    return

def process_files(config):
    """
    Process the files according to the configuration.
    """
    # Read the files and tile them
    preprocess_files(config)

    # Predict the tiles
    predict_tiles(config)

    # Post-process the predictions
    postprocess_files(config)

    shutil.rmtree(config["tiles_path"])  # Remove the tiles directory
    for folder in os.listdir(config["output_directory"]):        
        if os.path.exists(folder) and os.path.isdir(folder) and \
            (os.path.basename(folder) != "geojson_predictions" or os.path.basename(folder) != "logs"):
            shutil.rmtree(folder)

    # Print stats about the processing

if __name__ == "__main__":
    config = get_config("/home/jonas/tree_detection/tree_detection/config.yml")

    # Print Information about the configuration

    # Start reading the files and validate the configuration

    # Start the processing
    process_files(config)