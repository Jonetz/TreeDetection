import os
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.strtree import STRtree
from rasterio.plot import show
from matplotlib.patches import Patch
import json

def load_annotations(annotation_dir, filter_properties=[("Area", 0.0, True)]):
    annotations = {}
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".gpkg"):
            identifier = filename.split('_')[2].split('.')[0]
            gdf = gpd.read_file(os.path.join(annotation_dir, filename))
            if gdf.empty:
                continue  # Skip empty files
            gdf = gdf[gdf.geometry.notnull() & gdf.is_valid]
            for property, threshold, above in filter_properties:
                gdf = gdf[gdf[property] > threshold] if above else gdf[gdf[property] < threshold]
            annotations[identifier] = gdf
    return annotations

def load_predictions(prediction_dir):
    predictions = {}
    files = [os.path.join(root, f) for root, _, files in os.walk(prediction_dir) for f in files]

    for file_path in files:
        filename = os.path.basename(file_path)
        if filename.endswith(".gpkg"):
            identifier = filename.split('_')[1]
            gdf = gpd.read_file(file_path)
            if gdf.empty:
                continue
            gdf = gdf[gdf.geometry.notnull() & gdf.is_valid]
            predictions[identifier] = gdf
    return predictions

def clip_predictions(predictions, annotations):
    clipped_preds = {}
    for identifier, pred in predictions.items():
        if identifier in annotations:
            ann_gdf = annotations[identifier]
            if ann_gdf.empty or pred.empty:
                continue
            ann_bounds = ann_gdf.total_bounds
            clip_box = box(*ann_bounds)
            clipped_preds[identifier] = pred[pred.intersects(clip_box)]
        else:
            print(f"No annotations found for {identifier}")
    return clipped_preds

def calculate_iou(pred, ann, iou_threshold=0.3):
    """
    Calculate intersection over union (IoU) between predictions and annotations.
    Ensures each ground truth annotation is matched only once.
    """
    ann_strtree = STRtree(ann.geometry)
    
    pred_matched = set()
    ann_matched = set()
    iou_scores = []

    for p_idx, pred_geom in enumerate(pred.geometry):
        candidates = ann_strtree.query(pred_geom)
        if not np.any(candidates):
            continue

        candidate_indices = ann.index[candidates]
        candidate_geoms = ann.geometry.loc[candidate_indices]

        intersections = candidate_geoms.intersection(pred_geom).area
        unions = candidate_geoms.union(pred_geom).area
        iou_values = intersections / unions

        if len(iou_values) == 0:
            continue

        # Filter out annotations that are already matched
        unmatched_mask = ~candidate_indices.isin(ann_matched)
        iou_values = iou_values[unmatched_mask]
        candidate_indices = candidate_indices[unmatched_mask]

        if len(iou_values) == 0:
            continue

        # Assign match only if IoU meets the threshold
        max_iou_idx = np.argmax(iou_values)
        max_iou = iou_values.iloc[max_iou_idx]

        if max_iou >= iou_threshold:
            pred_matched.add(p_idx)
            ann_matched.add(candidate_indices[max_iou_idx])
            iou_scores.append(max_iou)

    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    tp_list = np.zeros(len(pred), dtype=bool)
    fp_list = np.ones(len(pred), dtype=bool)
    fn_list = np.ones(len(ann), dtype=bool)

    for p_idx in pred_matched:
        tp_list[p_idx] = True
        fp_list[p_idx] = False

    for a_idx in ann_matched:
        fn_list[ann.index.get_loc(a_idx)] = False

    return list(tp_list), list(fp_list), list(fn_list), tp_list.sum(), fp_list.sum(), fn_list.sum(), mean_iou

def evaluate(predictions, annotations, confidence_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], iou_threshold=0.3):
    results = {}
    for identifier, ann in annotations.items():
        if identifier not in predictions or ann.empty:
            continue
        
        pred = predictions[identifier]
        if pred.empty:
            continue

        for threshold in confidence_thresholds:
            pred_filtered = pred[pred["Confidence_score"] >= threshold].reset_index(drop=True)
            if pred_filtered.empty:
                continue

            tp_list, fp_list, fn_list, tp, fp, fn, mean_iou = calculate_iou(pred_filtered, ann, iou_threshold=iou_threshold)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results.setdefault(identifier, []).append(
                (threshold, precision, recall, f1, mean_iou, tp_list, fp_list, fn_list, pred_filtered)
            )
    
    return results


def visualize_results(identifier, annotations, predictions, rgb_folder, save_path, tp_list, fp_list, fn_list, threshold):
    """
    Visualize results for a given identifier.
    
    Parameters
    ----------
    identifier : str    Identifier for the image
    annotations : dict  Dictionary of annotations
    predictions : dict  Dictionary of predictions
    rgb_folder : str    Path to RGB images
    save_path : str     Path to save visualizations
    tp_list : list      List of size as predictions, with True for true positives
    fp_list : list      List of size as predictions, with True for false positives
    fn_list : list      List of false negatives geometries
    threshold : float   Confidence threshold to visualize    
    """

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    image_path = os.path.join(rgb_folder, f"FDOP20_{identifier}_rgbi_epsg25832.tif")
    if not os.path.exists(image_path):
        print(f"No image found for {identifier}")
        return
    
    with rasterio.open(image_path) as src:
        show(src, ax=ax, title='Best F1 Score Scenario')
        
    # Extract the bounding box of annotations and predictions
    ann = annotations.get(identifier, gpd.GeoDataFrame())
    
    ann_bounds = ann.total_bounds  # [minx, miny, maxx, maxy]
    pred_bounds = predictions.total_bounds if not predictions.empty else ann_bounds  # Use pred_bounds if available
    
    # Combine all extents
    minx = min(ann_bounds[0], pred_bounds[0])
    miny = min(ann_bounds[1], pred_bounds[1])
    maxx = max(ann_bounds[2], pred_bounds[2])
    maxy = max(ann_bounds[3], pred_bounds[3])
    
    
    # Plot ground truth
    ann.plot(ax=ax, facecolor='none', edgecolor='blue', label='Ground Truth')    
    
    tps = predictions[predictions.index.isin(np.where(tp_list)[0])]
    fps = predictions[predictions.index.isin(np.where(fp_list)[0])]
    
    if not tps.empty:
        tps.plot(ax=ax, facecolor='none', edgecolor='red', label='True Positives')
    if not fps.empty:
        fps.plot(ax=ax, facecolor='none', edgecolor='green', label='False Positives')
        
    legend_elements = [Patch(facecolor='none', edgecolor='blue', label='Ground Truth'),
                       Patch(facecolor='none', edgecolor='red', label='True Positives'),
                       Patch(facecolor='none', edgecolor='green', label='False Positives')]
    
    # Set limits to the combined extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
        
    ax.legend(handles=legend_elements)
    
    plt.title(f"Visualization for {identifier}")
    plt.savefig(os.path.join(save_path, f"{identifier}_{int(threshold*10)}_results.png"))
    plt.close()
    
def main():    
    """
    Folder Structure:
    
    - rgb           "FDOP20_id_rgbi_epsg25832.tif"
    - nDSM          "nDSM_id_1km.tif"
    - annotations   anno_category_id.gpkg
    - predictions   
        - type
            - FDOP20_id1_id2_rgbi_epsg25832.gpkg
    """
    
    # for every folder in predictions folder (e.g. predictions\combined)
    root = r"predictions"
    listing = list(os.listdir(root))
    listing.reverse()
    for folder in listing:
        print(f"Evaluating {folder}")
        prediction_dir = os.path.join(root, folder)
        save_path = os.path.join("visualizations", folder)
        annotation_dir = "annotations/added_info"
        rgb_folder = "rgb"
        os.makedirs(save_path, exist_ok=True)
        
        # Filter annotations by criteria that we think every tree should meet
        filter_properties = [("Area", 1.0, True), ("TreeHeight", 3.0, True), ("MeanNDVI", 0.3, True)]
        annotations = load_annotations(annotation_dir, filter_properties=filter_properties)
        #annotations = load_annotations(annotation_dir)
        predictions = load_predictions(prediction_dir)
        clipped_predictions = clip_predictions(predictions, annotations)
        for iou_threshold in [0.3, 0.5, 0.7, 0.9]:
            results = evaluate(clipped_predictions.copy(), annotations.copy(), iou_threshold=iou_threshold)
            
            for identifier, scores in results.items():
                print(f"Results for {identifier}:")
                                
                for threshold, precision, recall, f1, mean_iou, tp_list, fp_list, fn_list, pred in scores:
                    
                    # write to txt file
                    with open(f"{save_path}/{identifier}_scores.txt", "a") as f:
                        f.write(f"IoU Threshold: {iou_threshold} | Confidence Threshold: {threshold} | "
                                f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | "
                                f"TP: {sum(tp_list)} | FP: {sum(fp_list)} | FN: {sum(fn_list)} | Mean IoU: {mean_iou:.3f}\n")
                    
                    print(f"IoU Threshold: {iou_threshold} | Confidence Threshold: {threshold} | "
                          f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | "
                          f"TP: {sum(tp_list)} | FP: {sum(fp_list)} | FN: {sum(fn_list)} | Mean IoU: {mean_iou:.3f}")

                    if iou_threshold == 0.5 and threshold == 0.3:
                        visualize_results(identifier, annotations.copy(), pred, rgb_folder, save_path, tp_list, fp_list, fn_list, threshold)


def main():    
    root = "predictions"
    listing = list(os.listdir(root))
    listing.reverse()
    
    for folder in listing:
        print(f"Evaluating {folder}")
        prediction_dir = os.path.join(root, folder)
        save_path = os.path.join("visualizations", folder)
        annotation_dir = "annotations/added_info"
        rgb_folder = "rgb"
        os.makedirs(save_path, exist_ok=True)
        
        filter_properties = [("Area", 1.0, True), ("TreeHeight", 3.0, True), ("MeanNDVI", 0.15, True)]
        annotations = load_annotations(annotation_dir, filter_properties=filter_properties)
        #annotations = load_annotations(annotation_dir)
        predictions = load_predictions(prediction_dir)
        clipped_predictions = clip_predictions(predictions, annotations)
        
        results_dict = {}
        
        for iou_threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            results = evaluate(clipped_predictions.copy(), annotations.copy(), iou_threshold=iou_threshold)
            
            for identifier, scores in results.items():
                if identifier not in results_dict:
                    results_dict[identifier] = []
                
                for threshold, precision, recall, f1, mean_iou, tp_list, fp_list, fn_list, pred in scores:
                    results_dict[identifier].append((iou_threshold, threshold, precision, f1))
                    
                    with open(f"{save_path}/{identifier}_scores.txt", "a") as f:
                        f.write(f"IoU Threshold: {iou_threshold} | Confidence Threshold: {threshold} | "
                                f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | "
                                f"TP: {sum(tp_list)} | FP: {sum(fp_list)} | FN: {sum(fn_list)} | Mean IoU: {mean_iou:.3f}\n")
                    
                    print(f"IoU Threshold: {iou_threshold} | Confidence Threshold: {threshold} | "
                          f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | "
                          f"TP: {sum(tp_list)} | FP: {sum(fp_list)} | FN: {sum(fn_list)} | Mean IoU: {mean_iou:.3f}")

                    if iou_threshold == 0.5 and threshold == 0.3:
                        visualize_results(identifier, annotations.copy(), pred, rgb_folder, save_path, tp_list, fp_list, fn_list, threshold)

        # Save results as JSON
        json_save_path = os.path.join(save_path, "evaluation_results.json")
        with open(json_save_path, "w") as json_file:
            json.dump(results_dict, json_file, indent=4)
        print(f"Saved JSON results to {json_save_path}")

# Main for evaluation towards different models and plots
def main():    
    """
    Folder Structure:
    
    - rgb           "FDOP20_id_rgbi_epsg25832.tif"
    - nDSM          "nDSM_id_1km.tif"
    - annotations   anno_category_id.gpkg
    - predictions   
        - type
            - FDOP20_id1_id2_rgbi_epsg25832.gpkg
    """
    
    # for every folder in predictions folder (e.g. predictions\combined)
    root = r"predictions"
    listing = os.listdir(root)
    for folder in listing:
        #if folder != "postprocessed":
        #    continue
        print(f"Evaluating {folder}")
        prediction_dir = os.path.join(root, folder)
        save_path = os.path.join("visualizations", folder)
        annotation_dir = "annotations/added_info"
        rgb_folder = "rgb"
        os.makedirs(save_path, exist_ok=True)
        
        #Filter annotations by criteria that wee think every tree should meet
        filter_properties = [("Area", 1.0, True), ("TreeHeight", 3.0, True), ("MeanNDVI", 0.15, True)]
        annotations = load_annotations(annotation_dir, filter_properties=filter_properties)
        predictions = load_predictions(prediction_dir)
        clipped_predictions = clip_predictions(predictions, annotations)
        for iou_threshold in [0.3, 0.5, 0.7, 0.9]:
            results = evaluate(clipped_predictions.copy(), annotations.copy(), iou_threshold=iou_threshold)
            
            for identifier, scores in results.items():
                print(f"Results for {identifier}:")
                for threshold, precision, recall, f1, tp_list, fp_list, fn_list, pred in scores:
                    # write to txt file
                    with open(f"{save_path}/{identifier}_scores.txt", "a") as f:
                        f.write(f"IoU Threshold: {iou_threshold} | Confidence Threshold: {threshold} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | TP: {sum(tp_list)} | FP: {sum(fp_list)} | FN: {sum(fn_list)}\n")
                    print(f"IoU Threshold: {iou_threshold} | Confidence Threshold: {threshold} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | TP: {sum(tp_list)} | FP: {sum(fp_list)} | FN: {sum(fn_list)}")
                    if iou_threshold == 0.5 and threshold == 0.3:
                        visualize_results(identifier, annotations.copy(), pred, rgb_folder, save_path, tp_list, fp_list, fn_list, threshold)

"""
Main for parameter evaluation

def main():
    root = r"postprocessing_finetuning"
    listing = os.listdir(root)
    
    f1_scores = {}  # Dictionary to store F1 scores for each model
    
    for folder in listing:
        print(f"Evaluating {folder}")
        prediction_dir = os.path.join(root, folder)
        annotation_dir = "annotations/added_info"
        
        # Filter annotations by criteria
        filter_properties = [("Area", 1.0, True), ("TreeHeight", 3.0, True), ("MeanNDVI", 0.15, True)]
        annotations = load_annotations(annotation_dir, filter_properties=filter_properties)
        predictions = load_predictions(prediction_dir)
        clipped_predictions = clip_predictions(predictions, annotations)
        
        # Evaluate at IoU 0.5 and Confidence 0.3
        results = evaluate(clipped_predictions, annotations, confidence_thresholds=[0.3], iou_threshold=0.5)
        
        # Extract F1 scores
        model_f1_scores = []
        for identifier, scores in results.items():
            for threshold, precision, recall, f1, *_ in scores:
                if threshold == 0.3:
                    model_f1_scores.append(f1)
        
        # Compute average F1 score for the model
        avg_f1 = sum(model_f1_scores) / len(model_f1_scores) if model_f1_scores else 0
        f1_scores[folder] = avg_f1
    
    for key, value in f1_scores.items():
        print(f"{key}: {value}")
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.bar(f1_scores.keys(), f1_scores.values(), color='skyblue')
    plt.xlabel("Model")
    plt.ylabel("F1 Score at IoU 0.5 & Confidence 0.3")
    plt.title("F1 Scores for Different Models")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
"""
    
if __name__ == "__main__":
    main()
