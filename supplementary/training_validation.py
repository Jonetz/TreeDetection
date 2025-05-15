import geopandas as gpd
import pandas as pd
import json
import os
import shutil
from shapely.geometry import box
from shapely.ops import unary_union
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterio.plot import show
from matplotlib.patches import Patch
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from datetime import datetime
import argparse

def load_data(ground_truth_path, prediction_path):
    """
    Load ground truth and prediction data from given paths.

    Parameters:
    - ground_truth_path (str): Path to the ground truth file.
    - prediction_path (str): Path to the prediction file.

    Returns:
    - ground_truth (GeoDataFrame): GeoDataFrame containing ground truth data.
    - predictions (GeoDataFrame): GeoDataFrame containing prediction data.
    """
    if os.path.exists(ground_truth_path):
        ground_truth = gpd.read_file(ground_truth_path)
    else:
        ground_truth = gpd.GeoDataFrame({'geometry': []})
    
    with open(prediction_path) as f:
        predictions = gpd.GeoDataFrame.from_features(json.load(f)['features'])
    
    # Ensure 'Confidence_score' is numeric
    predictions['Confidence_score'] = pd.to_numeric(predictions['Confidence_score'], errors='coerce')
    
    return ground_truth, predictions

def clip_with_buffer(ground_truth, buffer_size):
    """
    Create a buffered bounding box around the ground truth geometry.

    Parameters:
    - ground_truth (GeoDataFrame): GeoDataFrame containing ground truth data.
    - buffer_size (float): Buffer size in the same units as the ground truth geometry.

    Returns:
    - buffered_bbox (Polygon): Buffered bounding box or None if the ground truth is empty.
    """
    if ground_truth.empty:
        return None
    bbox = unary_union(ground_truth.geometry).bounds
    buffered_bbox = box(
        bbox[0] - buffer_size, bbox[1] - buffer_size,
        bbox[2] + buffer_size, bbox[3] + buffer_size
    )
    return buffered_bbox

def clip_predictions_to_bbox(ground_truth, predictions):
    """
    Clip predictions to the bounding box of the ground truth.

    Parameters:
    - ground_truth (GeoDataFrame): GeoDataFrame containing ground truth data.
    - predictions (GeoDataFrame): GeoDataFrame containing prediction data.

    Returns:
    - clipped_predictions (GeoDataFrame): Predictions clipped to the bounding box of the ground truth.
    """
    if ground_truth.empty:
        return predictions
    bbox = unary_union(ground_truth.geometry).bounds
    bbox_geom = box(*bbox)
    clipped_predictions = predictions[predictions.geometry.intersects(bbox_geom)]
    return clipped_predictions

def compute_metrics(ground_truth, predictions, confidence_threshold):
    """
    Compute precision, recall, and F1 score for the predictions.

    Parameters:
    - ground_truth (GeoDataFrame): GeoDataFrame containing ground truth data.
    - predictions (GeoDataFrame): GeoDataFrame containing prediction data.
    - confidence_threshold (float): Confidence threshold for filtering predictions.

    Returns:
    - precision (float): Precision score.
    - recall (float): Recall score.
    - f1 (float): F1 score.
    """
    filtered_predictions = predictions[predictions['Confidence_score'] >= confidence_threshold]
    
    y_true = []
    y_pred = []
    
    for gt in ground_truth.geometry:
        matched = False
        for pred in filtered_predictions.geometry:
            if gt.intersects(pred):
                matched = True
                break
        y_true.append(1)
        y_pred.append(1 if matched else 0)
    
    for pred in filtered_predictions.geometry:
        if not any(pred.intersects(gt) for gt in ground_truth.geometry):
            y_true.append(0)
            y_pred.append(1)
    
    if y_true and y_pred:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    return precision, recall, f1

def evaluate_model(params, confidence_levels):
    """
    Evaluate the model at different confidence levels and return results.

    Parameters:
    - params (dict): Dictionary containing paths and other parameters.
    - confidence_levels (numpy array): Array of confidence levels to evaluate.

    Returns:
    - results (DataFrame): DataFrame containing precision, recall, and F1 score at different confidence levels.
    - ground_truth (GeoDataFrame): GeoDataFrame containing ground truth data.
    - clipped_predictions (GeoDataFrame): Predictions clipped to the bounding box of the ground truth.
    """
    ground_truth, predictions = load_data(params['ground_truth_path'], params['prediction_path'])
    clipped_predictions = clip_predictions_to_bbox(ground_truth, predictions)
    
    results = []
    for conf in confidence_levels:
        precision, recall, f1 = compute_metrics(ground_truth, clipped_predictions, conf)
        results.append({
            'confidence': conf,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        print('confidence', conf, 'precision', precision, 'recall', recall, 'f1_score', f1)
    
    return pd.DataFrame(results), ground_truth, clipped_predictions

def plot_metrics(results):
    """
    Plot precision, recall, and F1 score at different confidence levels.

    Parameters:
    - results (DataFrame): DataFrame containing precision, recall, and F1 score at different confidence levels.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['confidence'], results['precision'], label='Precision')
    plt.plot(results['confidence'], results['recall'], label='Recall')
    plt.plot(results['confidence'], results['f1_score'], label='F1 Score')
    plt.xlabel('Confidence Level')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics at Different Confidence Levels')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_best_f1(ground_truth, predictions, confidence_threshold, geotiff_path):
    """
    Plot the ground truth and predictions at the best F1 score.

    Parameters:
    - ground_truth (GeoDataFrame): GeoDataFrame containing ground truth data.
    - predictions (GeoDataFrame): GeoDataFrame containing prediction data.
    - confidence_threshold (float): Confidence threshold for filtering predictions.
    - geotiff_path (str): Path to the GeoTIFF image.
    """
    best_predictions = predictions[predictions['Confidence_score'] >= confidence_threshold]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    with rasterio.open(geotiff_path) as src:
        show(src, ax=ax, title='Best F1 Score Scenario')
    
    if not ground_truth.empty:
        ground_truth.plot(ax=ax, facecolor='none', edgecolor='blue', label='Ground Truth')
    best_predictions.plot(ax=ax, facecolor='none', edgecolor='red', label='Predictions')
    
    legend_elements = [Patch(facecolor='none', edgecolor='blue', label='Ground Truth'),
                       Patch(facecolor='none', edgecolor='red', label='Predictions')]
    
    ax.legend(handles=legend_elements)
    plt.show()

def generate_model_out_path(base_path, prefix="model"):
    """
    Generate a unique output path for the model based on the current date and a counter.

    Parameters:
    - base_path (str): Base path for the output directory.
    - prefix (str): Prefix for the output directory name.

    Returns:
    - out_path (str): Unique output path.
    """
    date_str = datetime.now().strftime('%Y%m%d')
    counter = 1
    while True:
        out_path = os.path.join(base_path, f"{prefix}_{date_str}_{counter:02d}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            return out_path
        counter += 1

def process_image(params, confidence=None):
    """
    Process a single image file using a specified trained model.

    Parameters:
    - params (dict): Dictionary containing paths and other parameters.

    Returns:
    - outpath (str): Path to the GeoJSON file containing the cleaned crowns.
    - clean (GeoDataFrame): GeoDataFrame containing the cleaned crowns.
    """
    img_path = params['geotiff_path']
    annotation_file = params['ground_truth_path']
    trained_model = params['trained_model']
    tiles_path = params['tiles_path']
    model_out_path = params['model_out_path']
    buffer = params['buffer']
    tile_width = params['tile_width']
    tile_height = params['tile_height']

    print(f"Processing {img_path}")

    data = rasterio.open(img_path)
    img_crs = data.crs

    if os.path.exists(annotation_file):
        ground_truth = gpd.read_file(annotation_file)

        buffered_bbox = clip_with_buffer(ground_truth, buffer)
    
        if buffered_bbox is None:
            print(f"No ground truth data available for {img_path}. Skipping processing.")
            return None, None
        
        # Calculate the window transform only if the bounding box is valid
        bbox_window = data.window(*buffered_bbox.bounds)
        if bbox_window.width <= 0 or bbox_window.height <= 0:
            print(f"Buffered bounding box is invalid for {img_path}. Skipping processing.")
            return None, None
        
        tmp_path = "/tmp/tmp.tif"
        window = data.window(*buffered_bbox.bounds)
        transform = data.window_transform(window)
        cropped_image = data.read(window=window)
    
        with rasterio.open(
                tmp_path, 'w', driver='GTiff', height=cropped_image.shape[1],
                width=cropped_image.shape[2], count=cropped_image.shape[0],
                dtype=cropped_image.dtype, crs=img_crs, transform=transform
            ) as dst:
                dst.write(cropped_image)
        
        data = rasterio.open(tmp_path)
         
    tile_data(data, tiles_path, buffer, tile_width, tile_height)
    
    if os.path.exists(annotation_file):
        os.remove(tmp_path)
    
    cfg = setup_cfg(update_model=trained_model)
    cfg.MODEL.DEVICE = "cpu"
    predict_on_data(tiles_path + "/", predictor=DefaultPredictor(cfg), eval=False)
    project_to_geojson(tiles_path, os.path.join(tiles_path, "predictions"), os.path.join(tiles_path, "predictions_geo"))

    if not os.path.exists(os.path.join(tiles_path, "predictions_geo")):
        os.makedirs(os.path.join(tiles_path, "predictions_geo"))
        return
    elif list(Path(os.path.join(tiles_path, "predictions_geo")).glob("*geojson")) == []:
        print(f"No geojson files found for {os.path.join(tiles_path, 'predictions_geo')}")
        return

    crowns = stitch_crowns(os.path.join(tiles_path, "predictions_geo"), 1)
    clean = clean_crowns(crowns, 0.6, confidence=0)

    for root, dirs, files in os.walk(tiles_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    
    outpath = os.path.join(model_out_path, f"prediction.geojson")
    clean.to_file(outpath)
    
    if confidence is not None:        
        filtered_predictions = clean[clean['Confidence_score'] >= confidence]
        outpath = os.path.join(model_out_path, f"prediction_filtered.geojson")
        filtered_predictions.to_file(outpath)
    
    return outpath, clean

def save_metrics_to_file(results, image_path, output_path):
    """
    Save the evaluation metrics to a file.

    Parameters:
    - results (DataFrame): DataFrame containing evaluation metrics.
    - image_path (str): Path to the evaluated image.
    - output_path (str): Path to save the evaluation metrics.
    """
    with open(output_path, 'w') as f:
        f.write(f"Image Path: {image_path}\n")
        f.write(f"{'Confidence':<15}{'Precision':<15}{'Recall':<15}{'F1 Score':<15}\n")
        for row in results.itertuples(index=False):
            f.write(f"{row.confidence:<15.5f}{row.precision:<15.5f}{row.recall:<15.5f}{row.f1_score:<15.5f}\n")

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    - params (dict): Dictionary containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Tree Detection Evaluation Script")
    parser.add_argument('--ground_truth_path', type=str, default="", help='Path to the ground truth file.')
    parser.add_argument('--geotiff_path', type=str, required=True, help='Path to the GeoTIFF image file.')
    parser.add_argument('--trained_model', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--tiles_path', type=str, required=True, help='Path to save the tiles.')
    parser.add_argument('--output_base_path', type=str, required=True, help='Base path for the output directory.')
    parser.add_argument('--buffer', type=float, default=20, help='Buffer size in the same units as the ground truth geometry.')
    parser.add_argument('--tile_width', type=int, default=50, help='Width of the tiles.')
    parser.add_argument('--tile_height', type=int, default=50, help='Height of the tiles.')

    args = parser.parse_args()

    params = {
        'ground_truth_path': args.ground_truth_path,
        'geotiff_path': args.geotiff_path,
        'trained_model': args.trained_model,
        'tiles_path': args.tiles_path,
        'buffer': args.buffer,
        'tile_width': args.tile_width,
        'tile_height': args.tile_height,
        'model_out_path': generate_model_out_path(args.output_base_path)
    }

    return params

if __name__ == "__main__":
    params = parse_arguments()
    
    prediction_path, crowns = process_image(params)
    if params["ground_truth_path"] == "":
        do_eval = False
    else:
        do_eval = True
        
    if prediction_path and crowns is not None:
        params['prediction_path'] = prediction_path
        if do_eval:
            confidence_levels = np.linspace(0.05, 0.95, 19)
            results, ground_truth, clipped_predictions = evaluate_model(params, confidence_levels)
            plot_metrics(results)
        
            metric_path = os.path.join(params['model_out_path'], "evaluation_metrics.txt")
            save_metrics_to_file(results, params['geotiff_path'], output_path=metric_path)
        
            best_f1_index = results['f1_score'].idxmax()
            best_confidence = results.loc[best_f1_index, 'confidence']
            print(f"Best F1 Score: {results.loc[best_f1_index, 'f1_score']} at Confidence Level: {best_confidence}")
        
            crowns = crowns[crowns['Confidence_score'] >= best_confidence]
            
            plot_best_f1(ground_truth, clipped_predictions, best_confidence, params['geotiff_path'])
            
        outpath = os.path.join(params['model_out_path'], f"best_result.geojson")
        crowns.to_file(outpath)
        
