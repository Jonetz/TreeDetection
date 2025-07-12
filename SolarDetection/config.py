import os

from TreeDetection.config import Config
from TreeDetection.config import load_config, set_device_configuration, setup_logging

def get_config(config_path: str):
    """
    Load the configuration from the specified path.
    """

    config = load_config(config_path)

    # 1. Check file handling paths
    assert config.get("image_directory") and os.path.exists(config.get("image_directory")), "Input path is missing from the configuration or path is incorrect."
    assert config.get("height_data_path") and os.path.exists(config.get("height_data_path")), "nDOM path is missing from the configuration or path is incorrect."
    assert config.get("combined_model") and os.path.exists(config.get("combined_model")), "Prediction model path is missing from the configuration or path is incorrect."
    assert str(config.get("combined_model")).endswith(('.h5', '.pth', '.pt')), "Prediction model path should end with .h5, .pth or .pt."
    
    config["combined_model"] = config.get("combined_model", None)    
    config["output_directory"] = config.get("output_directory", "./output")

    if not config["output_directory"]:
        os.makedirs(config["output_directory"], exist_ok=True)        
    config["tiles_path"] = config.get("tiles_path", "./tiles")
    if not config["tiles_path"]:
        os.makedirs(config["tiles_path"], exist_ok=True)

    # 3. Check the tiling parameters
    # tiles should be optimized to model training 
    # buffer should be at least 10 meters 
    # batch size best exerimentally, should be equivalent to 1.2 * Available GPU RAM
    config["tile_width"] = config.get("tile_width", 50)
    config["tile_height"] = config.get("tile_height", 50)
    config["buffer"] = config.get("buffer", 20)
    config["batch_size"] = config.get("batch_size", 10)

    # Overlapping tiles
    config["use_overlap"] = config.get("use_overlap", True)
    config["overlapping_tiles_width"] = config.get("overlapping_tiles_width", 3)
    config["overlapping_tiles_height"] = config.get("overlapping_tiles_height", 3)
    config["merged_path"] = config.get("merged_path", "merged")
    config["image_merged_regex"] = config.get("image_merged_regex", "FDOP20_(\\d+)_(\\d+)_(\\d+)_(\\d+)_(\\d+)\\.tif")
    config["height_data_merged_regex"] = config.get("height_data_merged_regex", "FDOP20_(\\d+)_(\\d+)\\.tif")


    # 4. Check the post-processing parameters
    # Stitching
    config['iou_threshold'] = config.get('iou_threshold', 0.5)
    config['confidence_threshold_stitching'] = config.get('confidence_threshold_stitching', 0.3)
    config['area_threshold'] = config.get('area_threshold', 6)

    # Special post-processing parameters
    config["exclude_files"] = config.get("exclude_files", [])
    config["confidence_threshold"] = config.get("confidence_threshold", 0.5)
    config["containment_threshold"] = config.get("containment_threshold", 0.9)

    config["height_threshold"] = config.get("height_threshold", 0.5)
    config["rectangularity_threshold"] = config.get("rectangularity_threshold", 0.6) # Only relevant if prediction is not in a building shape
    config["continue"] = config.get("continue", os.path.join(config["output_directory"], "continue.yml"))

    # 5. Other parameters
    raw_device = config.get("device", None)
    set_device_configuration(config, raw_device)
                
    config["parallel"] = config.get("parallel", True)
    config["num_workers"] = config.get("num_workers", None)
    config["verbose"] = config.get("verbose", False)
    config["debug"] = config.get("debug", False)
    config["logger"] = setup_logging(os.path.join(config["output_directory"], "logs"), config["debug"])
    config["keep_intermediate"] = config.get("keep_intermediate", False)
    config["timestamped_output_directory"] = config.get("timestamped_output_directory", False)
    config["simplify_tolerance"] = config.get("simplify_tolerance", 0.2)
    
    # 6. Solar panel specific parameters
    config["building_shapes"] = config.get("building_shapes", None)
    best_threshold = max((config["buffer"] + config["tile_width"]) * (config["buffer"] + config["tile_height"]) *  (1/30), 50)
    config["large_panels_threshold"] = config.get("large_panels_threshold", best_threshold)
    config["outside_area_threshold"] = config.get("outside_area_threshold", 20)

    config_obj = Config()
    config_obj._load_into_config(config)

    return config, config_obj