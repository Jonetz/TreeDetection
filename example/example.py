from TreeDetection.config import get_config
from TreeDetection.detection import process_files

if __name__ == "__main__":
    """
    With this script you can run the tree detection process on the provided example image of a 1 x 1 km square of Germany.
    """
    config, _ = get_config("/home/jonas/TreeDetection/example/config.yml")
    
    """
    For illustration purposes, we print the configuration, most of the configuration is set to default values, but can also be seen in the config files.
    
    Images are given by a filename that must contain a unique identifier to match it with the height data.
    The height data is given by a filename that must contain a unique identifier to match it with the images.
    
    Several models can be used to predict the trees, the combined model is used by default, but the urban and forest models can be used separately.
    """
    print("Configuration:")
    print(config)
    
    """
    With just one command the processing of the images in the directory can start, from there on the process is fully automated.
    If this is not wanted there are three highlevel functions that can be called separately:
    - preprocess_files(config)
    - predict_tiles(config)
    - postprocess_files(config)
    
    Each one works standalone, but they rely on being called in the correct order.
    """
    process_files(config)
