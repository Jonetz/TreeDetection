import os, shutil
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
#import wandb


  
def setup_logging(params):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(params['site_path'], 'logs.log'), format='%(asctime)s %(message)s', level=logging.DEBUG)
    return logger

def get_params(args):
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
    # Remove existing tile path directory if it exists
    if os.path.exists(params['tile_path']):
        shutil.rmtree(params['tile_path'])

    # Create tile path directory if it doesn't exist
    if not os.path.exists(params['tile_path']):
        os.makedirs(params['tile_path'])
        
    for rgb_file in glob(os.path.join(params['image_path'], '*.tif')):
        filename_withending = os.path.basename(rgb_file)
        filename = os.path.splitext(filename_withending)[0]
        
        # Locate the corresponding JSON or GPKG file
        if os.path.exists(os.path.join(params['json_path'], filename + '.gpkg')):
            json_file = os.path.join(params['json_path'], filename + '.gpkg')
        elif os.path.exists(os.path.join(params['json_path'], filename + '.json')):
            json_file = os.path.join(params['json_path'], filename + '.json')
        else:
            print('Could not find file:', filename)
            continue
        
        # Read in the tiff file
        data = rasterio.open(rgb_file)

        # Read in crowns (then filter by an attribute if required)
        crowns = gpd.read_file(json_file)
        
        # Filter out empty geometries
        crowns = crowns[~crowns.is_empty]

        # Filter out invalid geometries
        crowns = crowns[crowns.is_valid]

        # Filter to include only Polygons and MultiPolygons
        crowns = crowns[crowns.geometry.apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))]

        # Proceed only if crowns is not empty after filtering
        if not crowns.empty:
            tile_data_train(data, params['tile_path'], params['buffer'], params['tile_width'], params['tile_height'], crowns, params['threshold'])
        else:
            print("Skipping file, because no valid geometry is given:", rgb_file)
            continue
            
        # Split data into training and testing sets based on validation fold
        if params['val_fold'] is not None and params['val_fold'] > 1:
            to_traintest_folders(params['tile_path'], params['tile_path'], test_frac=params['test_frac'], strict=True, folds=params['val_fold'])
        else:
            to_traintest_folders(params['tile_path'], params['tile_path'], test_frac=params['test_frac'], strict=True)

            
def train_model(params):
    out_path = os.path.join(params['tile_path'], "train")
    register_train_data(out_path, 'BW', params['val_fold'])

    register_train_data("BW/tiles/train", 'BW', params['val_fold'])

    # Set the base (pre-trained) model from the detectron2 model_zoo
    base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    trains = ("BW_train", )  # Registered train data
    tests = ("BW_val")  # Registered validation data
    
    
    # Define the base output directory
    base_out_dir = "train_output"
    
    # Get the current date in YYYYMMDD format
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Initialize the run number
    run_number = 1
    
    # Find the next available run number for today
    while True:
        out_dir = os.path.join(base_out_dir, f"{current_date}_{run_number:02d}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            break
        run_number += 1
                    
    if params['update_model'] and  os.path.exists(params['update_model']):
        cfg = setup_cfg(base_model, trains, tests, workers=16, update_model=params['update_model'], ims_per_batch=9, gamma=0.1213, backbone_freeze=3, warm_iter=72, batch_size_per_im=1209, 
                      base_lr=0.005, max_iter=2000, num_classes=1, eval_period=100, out_dir=out_dir)  # update_model arg can be used to load in trained model
    else:
        #cfg = setup_cfg(base_model, trains, tests, workers=32, ims_per_batch=16, eval_period=100, max_iter=20000, out_dir=out_dir)  # update_model arg can be used to load in trained model
        cfg = setup_cfg( base_model, trains, tests, workers=4, ims_per_batch=4, gamma=0.01, backbone_freeze=3, warm_iter=72, batch_size_per_im=512, 
                      base_lr=0.01, max_iter=2000, num_classes=1, eval_period=100, out_dir=out_dir)
    
    trainer = MyTrainer(cfg, patience=10)
    trainer.resume_or_load(resume=False)
    
    trainer.train()
    
def main(args):
    print('Starting Main')
    params = get_params(args)
    print('Loaded Params')    
    logger = setup_logging(params)
    print('Setup Logging')
    preprocess_data(params, logger)
    
    print('Preprocessed data')
    train_model(params)

'''
def wandb_runs():
    # Initialize W&B run
    wandb.init(project="tree_detection", config={
        "buffer": 0,
        "tile_width": 50,
        "tile_height": 50,
        "val": 5,
        "threshold": 0.0,
        "ims_per_batch": 16,
        "max_iter": 6000
    })
    
    args = wandb.config
    print('Starting Main')
    params = get_params(args)
    print('Loaded Params')    
    logger = setup_logging(params)
    print('Setup Logging')
    preprocess_data(params, logger)
    print('Preprocessed data')
    train_model(params)
'''

if __name__ == "__main__":

    # Code for a simple run.
    parser = argparse.ArgumentParser(description="Tree Detection Training Script")
    parser.add_argument("--site_path", type=str, required=True, help="Path to the site directory")
    parser.add_argument("--buffer", type=int, default=0, help="Buffer size for tiling")
    parser.add_argument("--tile_width", type=int, default=50, help="Width of the tile")
    parser.add_argument("--tile_height", type=int, default=50, help="Height of the tile")
    #parser.add_argument("--box_threshold", type=float, default=0.5, help="Box threshold")
    #parser.add_argument("--iou_threshold", type=float, default=0.4, help="IoU threshold")
    parser.add_argument("--val", type=int, default=2, help="Validation Fold Count")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for tiling")
    parser.add_argument("--update_model", type=str, default="", help="Threshold for tiling")

    args = parser.parse_args()
    main(args)
    
    # Code for a W & B run.  
    #parser = argparse.ArgumentParser(description="Tree Detection Training Script")
    #parser.add_argument("--site_path", type=str, required=True, help="Path to the site directory")
    #args = parser.parse_args()
    #wandb.agent(args.site_path, function=wandb_runs)
