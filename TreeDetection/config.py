import logging
import os
from datetime import datetime
import warnings

import torch
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg


class Config:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.state = {}
        return cls._instance

    def _load_into_config(cls, config):
        for key, value in config.items():
            setattr(cls, key, value)

def setup_model_cfg(base_model="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", update_model=None, device="cpu"):
    """Set up config object for inference-only.
    
    Args:
        base_model: base pre-trained model from detectron2 model_zoo
        update_model: updated pre-trained model from detectree2 model_garden    
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    # Set the number of classes (only required if using custom weights with different class count)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # For a single class like trees
    
    # Set model weights (from pretrained or updated model)
    if update_model:
        cfg.MODEL.WEIGHTS = update_model
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)
    
    
    # Detectron2 only accepts 'cuda' or 'cpu' for MODEL.DEVICE
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()) or (isinstance(device, str) and device.startswith("cuda:")):
        #device = device.replace("cuda:", "")  # Remove 'cuda:' prefix if present
        gpu_id = int(device)
        if torch.cuda.is_available():
            print(f"Using GPU ID: {gpu_id}")
            torch.cuda.set_device(f'cuda:{gpu_id}')
            cfg.MODEL.DEVICE = "cuda"
        else:
            warnings.warn("CUDA not available, falling back to CPU.")
            cfg.MODEL.DEVICE = "cpu"
    elif device == "cuda":
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (trees)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Set threshold for predictions    
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.CUDNN_BENCHMARK = True
    cfg.SOLVER.AMP.ENABLED = True

    return cfg

def load_config(config_path: str):
    """Method to load  YAML config from a filepath

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dictionary: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_path: str, debug: bool):
    """Set up logging configuration."""
    # Create the log directory if it doesn't exist
    os.makedirs(log_path, exist_ok=True)
    
    # Log file with a timestamp
    log_filename = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(log_path, log_filename)
    
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    # Set up logging configuration
    logging.basicConfig(
        filename=log_file_path,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=log_level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger instance
    logger = logging.getLogger(__name__)
    
    # Optionally, also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def set_device_configuration(config, raw_device):
    if torch.cuda.is_available():
        if raw_device is not None:
            # Accept formats like "cuda:1", "1", or even "cuda"
            if isinstance(raw_device, int):
                device_str = raw_device
            elif isinstance(raw_device, str) and raw_device.startswith("cuda"):
                # Extract the device index from "cuda:X" format                
                if raw_device.replace("cuda:", "").isdigit():
                    device_str = raw_device.replace("cuda:", "")
                elif raw_device == "cuda":
                    device_str = "0"
            elif isinstance(raw_device, str) and raw_device.isdigit():
                device_str = raw_device
            else:
                device_str = "0"  # Default to first GPU if no valid device specified
            # Validate device index
            try:
                gpu_index = int(device_str)
                assert torch.cuda.device_count() > gpu_index, f"GPU index {gpu_index} is out of range."
            except (IndexError, ValueError):
                raise ValueError(f"Invalid CUDA device specification: {raw_device}")
            
            config["device"] = device_str
        else:
            config["device"] = "0"  # Default to first GPU
    else:
        if isinstance(raw_device, str) and raw_device.startswith("cuda"):
            warnings.warn(f"CUDA device '{raw_device}' requested but CUDA is not available. Falling back to CPU.")
            
        config["device"] = "cpu"
        
def get_config(config_path: str):
    """
    Load the configuration from the specified path.

    Args:
        config_path (str): Path to the configuration file. (should be a .yml file)

    Returns:
        dict: Configuration dictionary including the following keys:
        - image_directory (str): Path to the directory containing the input images.
        - height_data_path (str): Path to the nDOM data.
        - combined_model (str): Path to the combined model. (or forrest_outline/forrest_model/urban_model)
        - urban_model (str): Path to the urban model. (optional)
        - forrest_model (str): Path to the forrest model. (optional)
        - forrest_outline (str): Path to the forrest outline. (optional)    
        - output_directory (str): Path to the output directory. (default: ./output)
        - tiles_path (str): Path to the directory containing the tiles. (default: ./tiles)
        - continue_path (str): Path to the continue file. (default: ./output/continue.yml)
        - tile_width (int): Width of the tiles. (default: 50)
        - tile_height (int): Height of the tiles. (default: 50)
        - buffer (int): Buffer size for the tiles. (default: 0)
        - exclude_files (list): List of files that contain shapes to exclude from the predictions, such as water bodies, infrastructure, buildings, ... . (default: [])
        - confidence_threshold (float): Confidence threshold for the predictions. (default: 0.3)
        - containment_threshold (float): Containment threshold for the predictions. (default: 0.9)
        - height_threshold (float): Height threshold for the predictions. (default: 3)
    """
    config = load_config(config_path)

    # 1. Check file handling paths
    assert config.get("image_directory") and os.path.exists(config.get("image_directory")), "Input path is missing from the configuration or path is incorrect."
    assert config.get("height_data_path") and os.path.exists(config.get("height_data_path")), "nDOM path is missing from the configuration or path is incorrect."

    # 2. Check the models
    if not config.get("combined_model") or not os.path.exists(config.get("combined_model")):
        assert config.get("urban_model") and os.path.exists(config.get("urban_model")), "Urban model path is missing from the configuration or path is incorrect."
        assert config.get("forrest_model") and os.path.exists(config.get("forrest_model")), "Forrest model path is missing from the configuration."
        assert config.get("forrest_outline") and os.path.exists(config.get("forrest_outline")), "Forrest outline path is missing from the configuration."
    
    config["output_directory"] = config.get("output_directory", "./output")
    if not config["output_directory"]:
        os.makedirs(config["output_directory"], exist_ok=True)        
    config["tiles_path"] = config.get("tiles_path", "./tiles")
    if not config["tiles_path"]:
        os.makedirs(config["tiles_path"], exist_ok=True)
    config["continue"] = config.get("continue", os.path.join(config["output_directory"], "continue.yml"))

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
    config['area_threshold'] = config.get('area_threshold', 1)

    #Special post-processing parameters
    config["exclude_files"] = config.get("exclude_files", [])
    config["confidence_threshold"] = config.get("confidence_threshold", 0.3)
    config["containment_threshold"] = config.get("containment_threshold", 0.9)
    config["height_threshold"] = config.get("height_threshold", 3)

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
    
    config["building_shapes"] = config.get("building_shapes", None)

    config_obj = Config()
    config_obj._load_into_config(config)

    return config, config_obj