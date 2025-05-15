import os
import yaml


def load_prediction_recovery_data(output_path, tiles_path, model_path, logger, exclude=None):
    """
    Loads the recovery data to check which files have been processed already.
    
    Args:
        output_path (str): Path to the output directory.
        tiles_path (str): Path to the directory containing the tiles.
        model_path (str): Path to the model used for prediction.
        logger (logging.Logger): Logger for logging information.
        
    Returns:
        (list, set): A list of unprocessed file paths and a set of processed files.
    """
    recovery_file = os.path.join(output_path, "prediction_recovery.yaml")
    processed_files = set()
    skipped_files = 0
    file_list = []

    if os.path.exists(recovery_file):
        #try:
        with open(recovery_file, "r") as f:
            recovery_data = yaml.safe_load(f)

        # Check if the model_path matches the one stored in the recovery file
        if recovery_data.get("model_path") != model_path:
            logger.warning(f"Model path does not match the one stored in the recovery file. Skipping recovery.")
            return file_list, processed_files

        # Validate the matching of output folder files and metadata in JSON
        for file_path, keys in recovery_data["files"].items():
            json_path = os.path.join(tiles_path, os.path.basename(file_path).replace('.tif', '.json'))
            if os.path.exists(json_path):
                # Validate number of files in the folder and the JSON keys match
                folder_dir = os.path.join(output_path, os.path.basename(file_path).replace('.tif', ''))
                folder_files = len(os.listdir(folder_dir))

                if folder_files == len(keys):  # Compare number of files with the number of keys
                    processed_files.add(file_path)
                    continue  # No need to load the JSON file if it matches
                
                # Only if it fails, load JSON and apply exclude logic
                with open(json_path, 'r') as jf:
                    json_data = yaml.safe_load(jf)

                if exclude:
                    valid_entries = [
                        k for k, v in json_data.items()
                        if not any(v.get(flag, False) for flag in exclude)
                    ]
                else:
                    valid_entries = list(json_data.keys())

                if folder_files == len(valid_entries):
                    processed_files.add(file_path)
                else:
                    logger.debug(f"Mismatch between output folder and JSON (after excludes) for {file_path}.")
            else:
                logger.debug(f"Missing JSON metadata for {file_path}. Skipping.")
                
        # Filter out already processed files from the list
        original_len = len(file_list)
        file_list = [f for f in file_list if f not in processed_files]
        skipped_files = original_len - len(file_list)
        if skipped_files > 0:
            logger.info(f"Skipped {skipped_files} files that were already processed.")
            
        #except Exception as e:
        #    logger.warning(f"Could not load prediction recovery file: {e}")
    return file_list, processed_files

def save_prediction_recovery_data(output_path, tiles_path, model_path, processed_files, file_list):
    """
    Saves the recovery state after the prediction process.
    
    Args:
        output_path (str): Path to the output directory.
        tiles_path (str): Path to the directory containing the tiles.
        model_path (str): Path to the model used for prediction.
        processed_files (set): A set of processed files.
        file_list (list): The original list of files to process.
    """
    recovery_file = os.path.join(output_path, "prediction_recovery.yaml")
    try:
        recovery_data = {
            "model_path": model_path,
            "files": {}
        }
        files = list(file_list) + list(processed_files)
        for file_path in files:
            json_filename = os.path.basename(file_path).replace('.tif', '.json')
            json_path = os.path.join(tiles_path, json_filename)

            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    json_data = yaml.safe_load(json_file)
                
                # Only store the keys (filenames) from the JSON data
                recovery_data["files"][file_path] = list(json_data.keys())

        with open(recovery_file, "w") as f:
            yaml.safe_dump(recovery_data, f, sort_keys=False)

    except Exception as e:
        print(f"Failed to save prediction recovery file: {e}")


def load_stitching_recovery(output_path, logger=None):
    recovery_file = os.path.join(output_path, "stitching_recovery.yaml")
    completed = set()

    if os.path.exists(recovery_file):
        try:
            with open(recovery_file, "r") as f:
                recovery_data = yaml.safe_load(f)
            if recovery_data and "completed_files" in recovery_data:
                completed = set(os.path.basename(path) for path in recovery_data["completed_files"])
            if logger:
                logger.info(f"Loaded {len(completed)} completed files from recovery.")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load stitching recovery: {e}")
    return completed

def save_stitching_recovery(output_path, results, logger=None):
    recovery_file = os.path.join(output_path, "stitching_recovery.yaml")
    try:
        valid_results = [
            os.path.splitext(os.path.basename(r))[0] for r in results if r is not None
        ]
        valid_results = list(set(valid_results))  # Remove duplicates
        recovery_data = {
            "completed_files": valid_results
        }
        with open(recovery_file, "w") as f:
            yaml.safe_dump(recovery_data, f, sort_keys=False)
        if logger:
            logger.info(f"Saved recovery with {len(valid_results)} files.")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save stitching recovery: {e}")
            
def load_prediction_recovery_data(output_path, tiles_path, model_path, logger, exclude=None):
    """
    Loads the recovery data to check which files have been processed already.
    
    Args:
        output_path (str): Path to the output directory.
        tiles_path (str): Path to the directory containing the tiles.
        model_path (str): Path to the model used for prediction.
        logger (logging.Logger): Logger for logging information.
        
    Returns:
        (list, set): A list of unprocessed file paths and a set of processed files.
    """
    recovery_file = os.path.join(output_path, "prediction_recovery.yaml")
    processed_files = set()
    skipped_files = 0
    file_list = []

    if os.path.exists(recovery_file):
        #try:
        with open(recovery_file, "r") as f:
            recovery_data = yaml.safe_load(f)

        # Check if the model_path matches the one stored in the recovery file
        if recovery_data.get("model_path") != model_path:
            logger.warning(f"Model path does not match the one stored in the recovery file. Skipping recovery.")
            return file_list, processed_files

        # Validate the matching of output folder files and metadata in JSON
        for file_path, keys in recovery_data["files"].items():
            json_path = os.path.join(tiles_path, os.path.basename(file_path).replace('.tif', '.json'))
            if os.path.exists(json_path):
                # Validate number of files in the folder and the JSON keys match
                folder_dir = os.path.join(output_path, os.path.basename(file_path).replace('.tif', ''))
                folder_files = len(os.listdir(folder_dir))

                if folder_files == len(keys):  # Compare number of files with the number of keys
                    processed_files.add(file_path)
                    continue  # No need to load the JSON file if it matches
                
                # Only if it fails, load JSON and apply exclude logic
                with open(json_path, 'r') as jf:
                    json_data = yaml.safe_load(jf)

                if exclude:
                    valid_entries = [
                        k for k, v in json_data.items()
                        if not any(v.get(flag, False) for flag in exclude)
                    ]
                else:
                    valid_entries = list(json_data.keys())

                if folder_files == len(valid_entries):
                    processed_files.add(file_path)
                else:
                    logger.debug(f"Mismatch between output folder and JSON (after excludes) for {file_path}.")
            else:
                logger.debug(f"Missing JSON metadata for {file_path}. Skipping.")
                
        # Filter out already processed files from the list
        original_len = len(file_list)
        file_list = [f for f in file_list if f not in processed_files]
        skipped_files = original_len - len(file_list)
        if skipped_files > 0:
            logger.info(f"Skipped {skipped_files} files that were already processed.")
            
        #except Exception as e:
        #    logger.warning(f"Could not load prediction recovery file: {e}")
    return file_list, processed_files

def save_prediction_recovery_data(output_path, tiles_path, model_path, processed_files, file_list):
    """
    Saves the recovery state after the prediction process.
    
    Args:
        output_path (str): Path to the output directory.
        tiles_path (str): Path to the directory containing the tiles.
        model_path (str): Path to the model used for prediction.
        processed_files (set): A set of processed files.
        file_list (list): The original list of files to process.
    """
    recovery_file = os.path.join(output_path, "prediction_recovery.yaml")
    try:
        recovery_data = {
            "model_path": model_path,
            "files": {}
        }
        files = list(file_list) + list(processed_files)
        for file_path in files:
            json_filename = os.path.basename(file_path).replace('.tif', '.json')
            json_path = os.path.join(tiles_path, json_filename)

            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    json_data = yaml.safe_load(json_file)
                
                # Only store the keys (filenames) from the JSON data
                recovery_data["files"][file_path] = list(json_data.keys())

        with open(recovery_file, "w") as f:
            yaml.safe_dump(recovery_data, f, sort_keys=False)

    except Exception as e:
        print(f"Failed to save prediction recovery file: {e}")
        
def load_fusion_recovery(output_dir, logger=None):
    recovery_file = os.path.join(output_dir, "fusion_recovery.yaml")
    completed = set()

    if os.path.exists(recovery_file):
        try:
            with open(recovery_file, "r") as f:
                recovery_data = yaml.safe_load(f)
            if recovery_data and "completed_files" in recovery_data:
                completed = set(os.path.basename(path) for path in recovery_data["completed_files"])
            if logger:
                logger.info(f"Loaded {len(completed)} completed fusion files from recovery.")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load fusion recovery: {e}")
    return completed

def save_fusion_recovery(output_dir, results, logger=None):
    recovery_file = os.path.join(output_dir, "fusion_recovery.yaml")
    try:
        valid_results = [
            os.path.splitext(os.path.basename(r))[0] for r in results if r is not None
        ]        
        valid_results = list(set(valid_results))  # Remove duplicates
        recovery_data = {
            "completed_files": valid_results
        }
        with open(recovery_file, "w") as f:
            yaml.safe_dump(recovery_data, f, sort_keys=False)
        if logger:
            logger.info(f"Saved fusion recovery with {len(valid_results)} files.")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save fusion recovery: {e}")
