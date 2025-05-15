# Import necessary libraries
import os
from glob import glob

from sement_annotations_cambridge import process_image 
import geopandas as gpd
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import gc
import pandas as pd



from shapely.geometry import box

import logging
site_path = 'segmentation/cambridge_eval/'

'''Preprocessing'''
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(site_path, 'logs.log'), format='%(asctime)s %(message)s', encoding='utf-8', level=logging.INFO)

# Function to generate bounding boxes from polygons
def generate_bounding_boxes(polygons_gdf):
    """
    Generate bounding boxes from polygon annotations.

    Args:
        polygons_gdf (gpd.GeoDataFrame): GeoDataFrame containing the polygon annotations.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing bounding boxes.
    """
    bounding_boxes = polygons_gdf.copy()
    bounding_boxes['geometry'] = bounding_boxes['geometry'].apply(lambda x: x.bounds)
    bounding_boxes['geometry'] = bounding_boxes['geometry'].apply(lambda x: box(*x))
    return bounding_boxes

def predict_shapes(image, annotations_path, params, bounding_boxes, model="Sam"):
    """
    Predict shapes using bounding boxes.

    Args:
        image (str): Path to the image file.
        image (str): Path to the annotation file.
        bounding_boxes (gpd.GeoDataFrame): GeoDataFrame containing bounding boxes.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing predicted shapes.
    """
    
    if model=="SamHQ":
        from samgeo.hq_sam import SamGeo# Create a SAM object
        sam = SamGeo(
                model_type='vit_h',
                automatic = False,
                sam_kwargs=None
            )
    elif model=="Sam":
        from samgeo  import SamGeo
        sam = SamGeo(
                model_type='vit_h',
                sam_kwargs=None,
                automatic = False
            )
    
    # Get image filename
    filename = os.path.basename(image)

    # Dump the bounding boxes to a JSON file with the same name as the image file but with a .json extension in the annotations directory
    bounding_boxes.to_file(os.path.join(params['json_path'], os.path.splitext(filename)[0] + '.json'), driver='GeoJSON')
    
    # Process the image
    process_image(image, params, sam=sam)

    del sam
    gc.collect()
    
    # Read the predicted shapes from the output
    filename = os.path.splitext(os.path.basename(image))[0]
    out_path = os.path.join(params['crowns_path'])
    crowns_path = os.path.join(out_path, f"{filename}_T{params['tile_width']}_B{params['buffer']}_refined_cleaned.json")
    try:
        pred_gdf = gpd.read_file(crowns_path)
    except:
        return gpd.GeoDataFrame()
    return pred_gdf

# Function to calculate IoU for each pair of predicted and actual shapes
def calculate_iou(actual_shapes, predicted_shapes):
    """
    Calculate IoU for each pair of actual and predicted shapes.

    Args:
        actual_shapes (gpd.GeoDataFrame): GeoDataFrame containing actual polygon annotations.
        predicted_shapes (gpd.GeoDataFrame): GeoDataFrame containing predicted shapes.

    Returns:
        list: List of IoU values.
        dict: Dictionary containing counts of TP, FP, and FN.
    """
    iou_values = []
    tp, fp, fn = 0, 0, 0

    for _, actual in actual_shapes.iterrows():
        best_iou = 0
        for _, predicted in predicted_shapes.iterrows():
            intersection = actual.geometry.intersection(predicted.geometry).area
            union = actual.geometry.union(predicted.geometry).area
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
        if best_iou >= 0.5:
            iou_values.append(best_iou)
            tp += 1
        else:
            fn += 1

    for _, predicted in predicted_shapes.iterrows():
        best_iou = 0
        for _, actual in actual_shapes.iterrows():
            intersection = actual.geometry.intersection(predicted.geometry).area
            union = actual.geometry.union(predicted.geometry).area
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
        if best_iou < 0.5:
            fp += 1

    return iou_values, {'tp': tp, 'fp': fp, 'fn': fn}


# Function to plot IoU distribution
def plot_iou_distribution(iou_values, output_path):
    """
    Plot IoU distribution.

    Args:
        iou_values (list): List of IoU values.
        output_path (str): Path to save the plot.
    """
    plt.hist(iou_values, bins=20, range=(0, 1))
    plt.title('IoU Distribution')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()

def process_images(image_paths, annotations_path, output_path, params=None, model="Sam"):
    segmentation_path = os.path.join(output_path)
    if params is None:
        params = {
            'site_path': segmentation_path,
            'image_path': os.path.join(segmentation_path, 'rgb'),
            'mask_path': os.path.join(segmentation_path, 'masks'),
            'json_path': os.path.join(segmentation_path, 'json'),
            'tile_path': os.path.join(segmentation_path, 'tiles'),
            'crowns_path': os.path.join(segmentation_path, 'clean_crowns'),
            'buffer': 0,
            'tile_width': 140,
            'tile_height': 140,
            'box_threshold': 0.5,
            'iou_threshold': 0.3
        }

    annotations = gpd.read_file(annotations_path)
    all_iou_values = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for image_path in image_paths:
        with rio.open(image_path) as src:
            bounds = src.bounds
        image_extent = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        clipped_annotations = annotations[annotations.intersects(image_extent)].copy()
        clipped_annotations['geometry'] = clipped_annotations['geometry'].intersection(image_extent)
        temp_annotations_path = os.path.join(params['json_path'], 'temp_annotations.geojson')
        clipped_annotations.to_file(temp_annotations_path, driver='GeoJSON')
        bounding_boxes = generate_bounding_boxes(clipped_annotations)
        predicted_shapes = predict_shapes(image_path, temp_annotations_path, params, bounding_boxes, model=model)
        gc.collect()
        iou_values, counts = calculate_iou(clipped_annotations, predicted_shapes)
        all_iou_values.extend(iou_values)
        total_tp += counts['tp']
        total_fp += counts['fp']
        total_fn += counts['fn']
        gc.collect()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f'Image Directory: {image_directory}')
    logger.info(f'Annotations Path: {annotations_path}')
    logger.info(f'Output Path: {output_path}')
    logger.info('Parameters: %s', params)
    logger.info(f'Tile Width: {params["tile_width"]}')
    logger.info(f'Tile Height: {params["tile_height"]}')
    logger.info(f'Buffer: {params["buffer"]}')
    logger.info(f'Box Threshold: {params["box_threshold"]}')
    logger.info(f'IoU Threshold: {params["iou_threshold"]}')
    logger.info(f'Number of Images: {len(image_paths)}')
    logger.info(f'Number of Annotations: {len(annotations)}')
    logger.info(f'Number of Predictions: {len(all_iou_values)}')
    logger.info(f'TP: {total_tp}')
    logger.info(f'FP: {total_fp}')
    logger.info(f'FN: {total_fn}')
    logger.info(f'Total IoU: {np.mean(all_iou_values)}')
    logger.info(f'Model: {model}')
    logger.info(f'Precision: {precision}')
    logger.info(f'Recall: {recall}')
    logger.info(f'F1 Score: {f1}')

    plot_iou_distribution(all_iou_values, os.path.join(output_path, f'iou_dist_{model}_{params["tile_width"]}_{params["buffer"]}_{params["box_threshold"]}_{params["iou_threshold"]}.png'))

    # Return a dictionary of results for this iteration
    return {
        'Tile Width': params["tile_width"],
        'Tile Height': params["tile_height"],
        'Buffer': params["buffer"],
        'Box Threshold': params["box_threshold"],
        'IoU Threshold': params["iou_threshold"],
        'Number of Images': len(image_paths),
        'Number of Annotations': len(annotations),
        'Number of Predictions': len(all_iou_values),
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Total IoU': np.mean(all_iou_values),
        'Model': model,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

if __name__ == '__main__':
    image_directory = "segmentation/cambridge_eval/rgb/"
    annotations_path = "segmentation/cambridge_eval/json/cambridge_annotations.geojson"
    output_path = 'segmentation/cambridge_eval/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_paths = glob(os.path.join(image_directory, '*.tif'))

    segmentation_path = output_path
    params = {
        'site_path': segmentation_path,
        'image_path': os.path.join(segmentation_path, 'rgb'),
        'mask_path': os.path.join(segmentation_path, 'masks'),
        'json_path': os.path.join(segmentation_path, 'json'),
        'tile_path': os.path.join(segmentation_path, 'tiles'),
        'crowns_path': os.path.join(segmentation_path, 'clean_crowns'),
        'buffer': 0,
        'tile_width': 160,
        'tile_height': 160,
        'box_threshold': 0.5,
        'iou_threshold': 0.4
    }

    # List to accumulate results from all iterations
    results = []

    for buffer in [0, 20, 50]:
        params['buffer'] = buffer
        for tile_width in [50, 100, 160]:
            params['tile_width'] = tile_width
            params['tile_height'] = tile_width
            for box_threshold in [0.5, 0.4, 0.6]:
                params['box_threshold'] = box_threshold
                for io_threshold in [0.3, 0.2, 0.4]:
                    params['iou_threshold'] = io_threshold
                    if tile_width == 160:
                        params['buffer'] = 20
                    # Process images and collect results
                    iteration_result = process_images(image_paths, annotations_path, output_path, params=params, model="SamHQ")
                    results.append(iteration_result)

                    # Save all results as tabular data in a text file
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(os.path.join(output_path, 'evaluation_results.csv'), sep='\t', index=False)
