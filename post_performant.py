import os
import re
import warnings

import json
import os
from typing import Tuple
import sys
import time

import rasterio
from fiona.model import to_dict
import fiona

import shapely
import numba
import shapely
from rasterio._base import Affine
from rasterio.coords import BoundingBox
from shapely.geometry import shape, Point
from shapely import MultiPolygon
from shapely.geometry import shape, Polygon

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cupy as cp
import torch


from helpers import ndvi_array_from_rgbi


def convert_to_python_types(data):
    """
    Recursively convert NumPy data types to native Python types.
    """
    if isinstance(data, dict):
        return {key: convert_to_python_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert NumPy arrays to Python lists
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)  # Convert NumPy floats to Python floats
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)  # Convert NumPy ints to Python ints
    else:
        return data


# Assuming a function to check if GPU is available, this can be adjusted depending on your setup
def is_gpu_available():
    try:
        cp.cuda.Device(0).use()  # Check if the GPU is available and accessible
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False

def raster_to_geo(transform, row, col):
    """Convert raster indices to geographical coordinates."""
    # Extract elements from the 3x3 affine matrix
    a, b, c, d, e, f, g, h, i = transform
    # Apply the affine transformation to convert row, col to x, y
    x = a * col + b * row + c
    y = d * col + e * row + f
    return x, y


def geo_to_raster(transform, x, y):
    """Convert geographical coordinates to raster indices."""
    # Extract elements from the 3x3 affine matrix
    a, b, c, d, e, f, g, h, i = transform
    # Check for potential projective transformation and avoid division by zero
    if a == 0 or e == 0:
        raise ValueError(f"Affine transform scaling factors are zero: {a}, {e}, for Affine: {transform}")
        # Apply the inverse affine transformation (inverse of a 3x3 matrix)
    col = (x - c) / a
    row = (y - f) / e
    return int(row), int(col)

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def raster_to_geo_batch(transform, rows, cols):
    """Convert raster indices to geographical coordinates (batch version)."""
    a, b, c, d, e, f, g, h, i = transform
    x = a * cols + b * rows + c
    y = d * cols + e * rows + f
    return x, y


def geo_to_raster_batch(transform, x, y):
    """Convert geographical coordinates to raster indices (batch version)."""
    a, b, c, d, e, f, g, h, i = transform
    if a == 0 or e == 0:
        raise ValueError(f"Affine transform scaling factors are zero: {a}, {e}, for Affine: {transform}")
    col = (x - c) / a
    row = (y - f) / e
    return cp.floor(row).astype(cp.int32), cp.floor(col).astype(cp.int32)


def is_point_in_polygon_batch(center_x, center_y, radius, px, py):
    """
    Vectorized check if points are inside a circle approximated by the bounding box of the polygon.
    The center of the circle is the centroid of the bounding box, and the radius is the maximum distance
    from the center to any point in the bounding box.

    Arguments:
    - polygon_x: (num_vertices, ) x-coordinates of the polygon's vertices
    - polygon_y: (num_vertices, ) y-coordinates of the polygon's vertices
    - px, py: (batch_size, ) x and y coordinates of the points to check

    Returns:
    - inside_mask: (batch_size, ) Boolean array where True indicates the point is inside the circle
    """

    # Calculate the squared distance from each point to the circle center
    dist_squared = (px - center_x) ** 2 + (py - center_y) ** 2
    # Return True if the distance is less than or equal to the radius squared
    inside_mask = dist_squared <= radius ** 2

    return inside_mask


def get_height_within_polygon(polygon_x: np.ndarray, polygon_y: np.ndarray, height_data: np.ndarray,
                              transform: np.ndarray, width, height, bounds):
    """
    Find the maximum height within multiple polygons from raster height data using ray-casting, 
    processed in parallel for each polygon.
    
    Arguments:
    - polygon_x: (num_polygons, num_vertices) x-coordinates of the polygons' vertices
    - polygon_y: (num_polygons, num_vertices) y-coordinates of the polygons' vertices
    - height_data: (height, width) raster height data
    - transform: geo-to-raster transformation matrix
    - width, height: dimensions of the raster data
    - bounds: (minx, miny, maxx, maxy) bounding box for the area to consider
    
    Returns:
    - max_heights: (num_polygons,) maximum heights for each polygon
    - max_coordinates: (num_polygons, 2) coordinates of the highest point for each polygon
    """
    # Unpack bounds
    minx, miny, maxx, maxy = bounds
    
    # Determine raster coordinates for bounding box
    min_col, min_row = geo_to_raster(transform, minx, miny)
    max_col, max_row = geo_to_raster(transform, maxx, maxy)
    
    # Clamp values within valid raster bounds
    min_row, max_row = sorted([min(min_row, height - 1), max(max_row, 0)])
    min_col, max_col = sorted([min(min_col, width - 1), max(max_col, 0)])

    # Extract height data subset based on bounding box
    subset = height_data[min_row:max_row + 1, min_col:max_col + 1]
    if subset.size == 0:
        print("Error: The subset of height data is empty.")
        return -1, None    

    # Prepare points (x, y) for batch processing
    rows, cols = np.meshgrid(np.arange(subset.shape[0]), np.arange(subset.shape[1]), indexing='ij')
    rows, cols = rows.flatten(), cols.flatten()
    
    # Convert to geo-coordinates for all points
    x_coords, y_coords = raster_to_geo(transform, rows + min_row, cols + min_col)

    # Convert x_coords and y_coords to CuPy arrays for GPU processing
    x_coords_gpu, y_coords_gpu = cp.array(x_coords), cp.array(y_coords)
    
    # Prepare arrays for results
    num_polygons = polygon_x.shape[0]
    max_heights = cp.zeros(num_polygons, dtype=cp.float32)
    max_coordinates = cp.zeros((num_polygons, 2), dtype=cp.float32)

    centers = cp.zeros((num_polygons, 2), dtype=cp.float32)

    # Process each polygon in parallel
    for i in range(num_polygons):
        # Get current polygon coordinates
        polygon_x_i, polygon_y_i = polygon_x[i], polygon_y[i]
        
        valid_mask = ~cp.isnan(polygon_x_i) & ~cp.isnan(polygon_y_i)  # Mask to remove NaNs
        valid_x = polygon_x_i[valid_mask]
        valid_y = polygon_y_i[valid_mask]
        
        # Compute the center of the bounding box for the polygon (after filtering NaNs)
        min_x, max_x = valid_x.min(), valid_x.max()
        min_y, max_y = valid_y.min(), valid_y.max()
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        centers[i] = cp.array([center_x, center_y])

        # Compute the radius of the bounding box for the polygon
        radius = (max_x - min_x) + (max_y - min_y) / 4
        
        # Check if points are inside the polygon using a vectorized function
        inside_mask = is_point_in_polygon_batch(center_x, center_y, radius, x_coords_gpu, y_coords_gpu, )

        # Extract heights and coordinates where points are inside the polygon
        inside_heights = subset.flatten()[inside_mask]
        inside_coords = np.column_stack([x_coords_gpu[inside_mask], y_coords_gpu[inside_mask]])

        # Handle empty result (fallback to centroid)
        if inside_heights.size == 0:
            print(f"TODO No points found within polygon {i}. Implementing fallback to centroid.")
            max_coordinates[i] = cp.array([-1, -1])
            max_heights[i] = -1  # Placeholder value, can be adjusted
        else:
            # Find the maximum height and its coordinates
            max_index = cp.argmax(inside_heights)
            max_heights[i] = inside_heights[max_index]
            max_coordinates[i] = inside_coords[max_index]
    
    return max_heights, max_coordinates


def get_ndvi_within_polygon(polygon_x: np.ndarray, polygon_y: np.ndarray, ndvi_data: np.ndarray, transform: np.ndarray, width: int, height: int, bounds: BoundingBox):
    """
    Find the minimum, maximum, and mean NDVI values within a polygon from raster NDVI data.

    Args:
        polygon (shapely.geometry.Polygon): Polygon defining the area of interest.
        ndvi_data (numpy.ndarray): 2D array of NDVI data from the raster.
        transform (Affine): Transformation matrix for geo to raster / raster to geo coordinates.
        width (int): Width of the raster in pixels.
        height (int): Height of the raster in pixels.
        bounds (BoundingBox): Bounds of the raster.

    Returns:
        tuple: (min_ndvi, max_ndvi, mean_ndvi)
            - min_ndvi (float): Minimum NDVI value within the polygon.
            - max_ndvi (float): Maximum NDVI value within the polygon.
            - mean_ndvi (float): Mean NDVI value within the polygon.
    """
    minx, miny, maxx, maxy = bounds

    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])

    if maxx < bounds.left or minx > bounds.right or maxy < bounds.bottom or miny > bounds.top:
        print(f"Polygon bounds {minx, miny, maxx, maxy} are out of raster bounds.")
        return 0, 0, 0

    min_col, min_row = geo_to_raster(transform, minx, miny)
    max_col, max_row = geo_to_raster(transform, maxx, maxy)

    min_row, max_row = sorted([min_row, max_row])
    min_col, max_col = sorted([min_col, max_col])

    min_row, min_col = max(0, min_row), max(0, min_col)
    max_row, max_col = min(height - 1, max_row), min(width - 1, max_col)

    subset = ndvi_data[min_row:max_row + 1, min_col:max_col + 1]

    # Prepare points (x, y) for batch processing
    rows, cols = np.meshgrid(np.arange(subset.shape[0]), np.arange(subset.shape[1]), indexing='ij')
    rows, cols = rows.flatten(), cols.flatten()

    # Convert to geo-coordinates for all points
    x_coords, y_coords = raster_to_geo(transform, rows + min_row, cols + min_col)

    # Convert x_coords and y_coords to CuPy arrays for GPU processing
    x_coords_gpu, y_coords_gpu = cp.array(x_coords), cp.array(y_coords)

    # Prepare arrays for results
    num_polygons = polygon_x.shape[0]
    min_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)
    max_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)
    mean_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)

    centers = cp.zeros((num_polygons, 2), dtype=cp.float32)

    for i in range(num_polygons):
        # Get current polygon coordinates
        polygon_x_i, polygon_y_i = polygon_x[i], polygon_y[i]

        valid_mask = ~cp.isnan(polygon_x_i) & ~cp.isnan(polygon_y_i)  # Mask to remove NaNs
        valid_x = polygon_x_i[valid_mask]
        valid_y = polygon_y_i[valid_mask]

        # Compute the center of the bounding box for the polygon (after filtering NaNs)
        min_x, max_x = valid_x.min(), valid_x.max()
        min_y, max_y = valid_y.min(), valid_y.max()

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        centers[i] = cp.array([center_x, center_y])

        # Compute the radius of the bounding box for the polygon
        radius = (max_x - min_x) + (max_y - min_y) / 4

        # Check if points are inside the polygon using a vectorized function
        inside_mask = is_point_in_polygon_batch(center_x, center_y, radius, x_coords_gpu, y_coords_gpu, )

        # Extract heights and coordinates where points are inside the polygon
        inside_ndvi = subset.flatten()[inside_mask]

        mean_ndvi = cp.mean(inside_ndvi)

        # Handle empty result (fallback to centroid)
        min_index = cp.argmin(inside_ndvi)
        max_index = cp.argmax(inside_ndvi)

        min_ndvi_values[i] = inside_ndvi[min_index]
        max_ndvi_values[i] = inside_ndvi[max_index]
        mean_ndvi_values[i] = mean_ndvi

    return min_ndvi_values, max_ndvi_values, mean_ndvi_values

def calculate_iou(batch_boxes1, batch_boxes2):
    # Ensure batch_boxes1 and batch_boxes2 are 2D arrays with shape [N, 4]
    batch_boxes1 = cp.array(batch_boxes1).reshape(-1, 4)
    batch_boxes2 = cp.array(batch_boxes2).reshape(-1, 4)
    
    # Calculate the intersection of boxes using broadcasting
    xA = cp.maximum(batch_boxes1[:, 0][:, None], batch_boxes2[:, 0])  # Broadcast to compare all pairs
    yA = cp.maximum(batch_boxes1[:, 1][:, None], batch_boxes2[:, 1])  # Broadcast to compare all pairs
    xB = cp.minimum(batch_boxes1[:, 2][:, None], batch_boxes2[:, 2])  # Broadcast to compare all pairs
    yB = cp.minimum(batch_boxes1[:, 3][:, None], batch_boxes2[:, 3])  # Broadcast to compare all pairs

    # Calculate the area of intersection
    interArea = cp.maximum(0, xB - xA) * cp.maximum(0, yB - yA)
    
    # Calculate the area of each box
    box1Area = (batch_boxes1[:, 2] - batch_boxes1[:, 0]) * (batch_boxes1[:, 3] - batch_boxes1[:, 1])
    box2Area = (batch_boxes2[:, 2] - batch_boxes2[:, 0]) * (batch_boxes2[:, 3] - batch_boxes2[:, 1])
    
    # Calculate the area of union
    unionArea = box1Area[:, None] + box2Area - interArea  # Broadcast union area computation
    
    # Return IoU for all pairs
    return interArea / unionArea


def filter_polygons_by_iou_and_area(polygon_dict, id_to_area, confidence_scores, iou_threshold, area_threshold):
    """
    Filter polygons by IOU and area thresholds on the GPU, keeping the polygon with the highest confidence score.
    
    Args:
        polygon_dict (dict): A dictionary where keys are polygon ids and values are the polygon objects (assumed to have `.bounds`).
        id_to_area (dict): A dictionary mapping polygon ids to their areas.
        confidence_scores (dict): A dictionary mapping polygon ids to their confidence scores.
        iou_threshold (float): The threshold for Intersection over Union (IoU).
        area_threshold (float): The threshold for area difference.

    Returns:
        retained_ids (set): A set of ids of polygons that are retained after filtering.
    """
    ids = list(polygon_dict.keys())
    
    # Convert data to CuPy arrays
    bboxes = cp.array([polygon_dict[pid].bounds for pid in ids])
    confidences = cp.array([confidence_scores[pid] for pid in ids])
    areas = cp.array([id_to_area[pid] for pid in ids])
    
    retained_ids = set(ids)

    # Create an array to track which polygons to remove
    remove_indices = cp.zeros(len(ids), dtype=cp.bool_)

    # Vectorized computation of IoU and area differences
    iou_matrix = calculate_iou(bboxes, bboxes)  # IoU matrix for all pairs
    area_matrix = cp.abs(areas[:, None] - areas) / cp.maximum(areas[:, None], areas)  # Area difference matrix
    
    # Apply IoU and area thresholds to filter polygons
    mask = (iou_matrix > iou_threshold) & (area_matrix < area_threshold)  # Boolean mask for filtering
    
    # Loop through each polygon
    for i in range(len(ids)):
        if remove_indices[i]:
            continue
        
        # Get the indices of the polygons to compare with the current polygon i
        to_remove = cp.where(mask[i])[0]  # Get indices of polygons with which i has IoU > threshold
        
        # Exclude self-comparison (i.e., i == j)
        to_remove = to_remove[to_remove != i]
        
        for j in to_remove:
            if remove_indices[j]:  # Skip if j is already flagged for removal
                continue
            
            # Apply the condition: if IoU and area difference are above the thresholds
            if confidences[i] >= confidences[j]:
                remove_indices[j] = True  # Flag j for removal
            else:
                remove_indices[i] = True  # Flag i for removal
                break  # Once one is flagged, no need to compare further with this polygon

    # Determine which polygons to remove
    remove_ids = {ids[i] for i in range(len(ids)) if remove_indices[i]}
    retained_ids -= remove_ids

    return retained_ids

def calculate_area(polygon):
    """
    Calculate the area of a polygon.

    Args:
        polygon (shapely.geometry.Polygon): Polygon object.

    Returns:
        float: Area of the polygon.
    """
    return polygon.area

def get_centroids(polygon_x_gpu, polygon_y_gpu):
    """
    Compute the centroids for all polygons in the batch.
    
    Args:
        polygon_x_gpu (ndarray): x-coordinates of all polygons.
        polygon_y_gpu (ndarray): y-coordinates of all polygons.
    
    Returns:
        centroids (ndarray): The centroids of all polygons.
    """
    # Calculate centroids in parallel for all polygons
    centroid_x = cp.mean(polygon_x_gpu, axis=1)
    centroid_y = cp.mean(polygon_y_gpu, axis=1)
    
    return cp.stack((centroid_x, centroid_y), axis=1)

def round_coordinates(polygon, decimals=3):
    """
    Round the coordinates of a polygon to a specified number of decimal places.

    Args:
        polygon (list): Polygon coordinates.
        decimals (int): Number of decimal places to round to (default is 3).

    Returns:
        list: Rounded polygon coordinates.
    """
    factor = 10 ** decimals
    try:
        return [[[round(coord * factor) / factor for coord in point] for point in ring] for ring in polygon]
    except:
        return [[[coord for coord in point] for point in ring] for ring in polygon]

def process_containment_features(features, polygon_ids, polygon_bounds, containment_threshold=0.6):
    """
    Efficiently process features to calculate polygon containment using GPU, excluding self-containment.

    Args:
        features (list): List of features with GeoJSON-like format.
        containment_threshold (float): Percentage of bounding box overlap to determine containment.

    Returns:
        list: Updated features with 'is_contained' and 'num_contained' properties added.
    """
    # Convert bounding boxes to CuPy arrays
    bounds_gpu = cp.array(polygon_bounds, dtype=cp.float32)  # Shape: (num_polygons, 4)

    # Step 2: Compute overlap for all pairs of polygons
    num_polygons = bounds_gpu.shape[0]
    min_x_outer = bounds_gpu[:, 0][:, None]
    min_y_outer = bounds_gpu[:, 1][:, None]
    max_x_outer = bounds_gpu[:, 2][:, None]
    max_y_outer = bounds_gpu[:, 3][:, None]

    min_x_inner = bounds_gpu[:, 0]
    min_y_inner = bounds_gpu[:, 1]
    max_x_inner = bounds_gpu[:, 2]
    max_y_inner = bounds_gpu[:, 3]

    # Calculate intersection dimensions
    inter_min_x = cp.maximum(min_x_outer, min_x_inner[None, :])
    inter_min_y = cp.maximum(min_y_outer, min_y_inner[None, :])
    inter_max_x = cp.minimum(max_x_outer, max_x_inner[None, :])
    inter_max_y = cp.minimum(max_y_outer, max_y_inner[None, :])

    # Compute intersection areas
    inter_width = cp.maximum(0, inter_max_x - inter_min_x)
    inter_height = cp.maximum(0, inter_max_y - inter_min_y)
    intersection_area = inter_width * inter_height

    # Compute areas of the inner polygons
    inner_areas = (max_x_inner - min_x_inner) * (max_y_inner - min_y_inner)

    # Calculate containment ratio
    containment_ratios = intersection_area / inner_areas[None, :]  # Shape: (num_polygons, num_polygons)

    # Step 3: Determine containment based on the threshold
    is_contained = containment_ratios >= containment_threshold

    # Exclude self-containment (set diagonal to False)
    is_contained[cp.arange(num_polygons), cp.arange(num_polygons)] = False

    num_contained = cp.sum(is_contained, axis=1).get()  # Convert to NumPy array for serialization

    # Step 4: Update features with containment information
    updated_features = []
    for i, feature in enumerate(features):
        if polygon_ids[i] not in polygon_ids:
            continue  # Skip invalid features
        new_properties = dict(feature['properties'])
        new_properties['is_contained'] = bool(cp.any(is_contained[:, i]).get())  # Is this polygon contained?
        new_properties['num_contained'] = int(num_contained[i])  # How many polygons contain this one?

        updated_features.append({
            'type': 'Feature',
            'properties': new_properties,
            'geometry': feature['geometry']
        })

    return updated_features


def process_features(features, polygon_dict, id_to_area, containment_threshold, height_data, height_transform, ndvi_data, ndvi_transform, width, height, height_bounds, ndvi_bounds):
    polygon_x_all = []
    polygon_y_all = []
    ids_all = []

    # Collect all polygons' coordinates and IDs for batch processing
    max_points = 0  # Track the maximum number of points in any polygon
    polygon_bounds = []
    for feature in features:
        polygon = shape(feature['geometry'])

        # If the polygon is a MultiPolygon, merge it into a single Polygon
        if isinstance(polygon, MultiPolygon):
            polygon = list(polygon.geoms)[0]
        if isinstance(polygon, Polygon):
            # Handle single Polygon
            polygon_x, polygon_y = polygon.exterior.xy
            polygon_x_all.append(polygon_x)
            polygon_y_all.append(polygon_y)
        else:
            print(f'Other type than polygon encountered: {type(polygon)}')
        polygon_bounds.append(polygon.bounds)
        ids_all.append(feature['properties']['poly_id'])
        max_points = max(max_points, len(polygon_x))  # Update max_points

        def pad_polygon_coords(polygon_coords_all, max_length):
            """
            Pads a batch of polygons with NaN values to ensure all have the same length.

            Args:
                polygon_coords_all (list of cupy.ndarray): List of 1D arrays of polygon coordinates.
                max_length (int): The maximum length to which each polygon is padded.

            Returns:
                cupy.ndarray: 2D array where each row represents a padded polygon.
            """
            # Calculate the batch size
            batch_size = len(polygon_coords_all)

            # Create an empty array filled with NaN of shape (batch_size, max_length)
            padded_array = cp.full((batch_size, max_length), cp.nan)

            # Fill the padded array with the polygon coordinates
            for i, coords in enumerate(polygon_coords_all):
                padded_array[i, :len(coords)] = cp.array(coords)

            return padded_array

    # Now use this function to pad the coordinates
    polygon_x_all_padded = pad_polygon_coords(polygon_x_all, max_points)
    polygon_y_all_padded = pad_polygon_coords(polygon_y_all, max_points)

    # Convert the padded lists into CuPy arrays
    polygon_x_gpu = cp.array(polygon_x_all_padded, dtype=cp.float32)  # Shape: (num_polygons, max_points)
    polygon_y_gpu = cp.array(polygon_y_all_padded, dtype=cp.float32)  # Shape: (num_polygons, max_points)

    height_data_gpu = cp.array(height_data, dtype=cp.float32)
    ndvi_data_gpu = cp.array(ndvi_data, dtype=cp.float32)

    # Compute centroids for all polygons on GPU (using the batch processing function)
    centroids = get_centroids(polygon_x_gpu, polygon_y_gpu)

    # Perform height data lookups for all polygons at once on the GPU
    heights, highest_points = get_height_within_polygon(polygon_x_gpu, polygon_y_gpu, height_data_gpu, height_transform, width,
                                                        height, height_bounds)

    # Perform NDVI data lookups for all polygons at once on the GPU, similar to height data lookup
    min_ndvi, max_ndvi, mean_ndvi = get_ndvi_within_polygon(polygon_x_gpu,
                                                            polygon_y_gpu,
                                                            ndvi_data_gpu,
                                                            ndvi_transform,
                                                            ndvi_data.shape[0],
                                                            ndvi_data.shape[1],
                                                            ndvi_bounds)

    # Call process_containment_features and retrieve attributes
    polygon_bounds = cp.array(polygon_bounds, dtype=cp.float32)
    containment_results = process_containment_features(features, ids_all, polygon_bounds, containment_threshold)

    # Extract containment results
    containment_info = {feature['properties']['poly_id']: {'is_contained': feature['properties']['is_contained'],
                                                           'num_contained': feature['properties']['num_contained']}
                        for feature in containment_results}

    # Step to further select the polygons based on containment results
    selected_features = []

    for feature in features:
        polygon_id = feature['properties']['poly_id']
        containment_data = containment_info.get(polygon_id, {'is_contained': False, 'num_contained': 0})

        if containment_data['num_contained'] >= 3:
            # Case 1: Contains at least three other polygons, discard it
            continue
        elif containment_data['num_contained'] == 2:
            # Case 2: Contains two other polygons
            contained_polygons = [f for f in features if polygon_id != f['properties']['poly_id'] and containment_info[f['properties']['poly_id']]['is_contained']]
            if len(contained_polygons) == 2:
                # Case 2a: Check mutual containment
                if polygon_id in [f['properties']['poly_id'] for f in contained_polygons]:
                    # Remove one of the polygons
                    continue
        elif containment_data['num_contained'] == 1:
            # Case 3: Contains one other polygon, apply sorting strategies
            other_polygon_id = [f['properties']['poly_id'] for f in features if containment_info[f['properties']['poly_id']]['is_contained']][0]
            other_polygon = next(f for f in features if f['properties']['poly_id'] == other_polygon_id)

            # Sorting logic:
            # TODO Refine sorting logic
            if heights[features.index(feature)] > heights[features.index(other_polygon)] :
                selected_features.append(feature)
            else:
                # Handle cases where NDVI or area is preferred
                pass
        else:
            # Case 4: Does not contain anything, no problem, we keep it
            selected_features.append(feature)

    # Convert CuPy arrays to NumPy arrays for serialization
    heights = heights.get()  # Convert to NumPy array
    highest_points = highest_points.get()  # Convert to NumPy array

    min_ndvi = min_ndvi.get()
    max_ndvi = max_ndvi.get()
    mean_ndvi = mean_ndvi.get()

    updated_features = []

    for i, feature in enumerate(selected_features):
        polygon_id = feature['properties']['poly_id']
        area = id_to_area.get(polygon_id, None)

        # Check if the feature is within the containment threshold and has valid data
        height = heights[i] if highest_points[i] is not None else -1
        centroid = centroids[i].get()  # Convert centroid from CuPy to NumPy

        try:
            rounded_coords = round_coordinates(feature['geometry']['coordinates'])
        except Exception as e:
            print(f"Error rounding coordinates for polygon {polygon_id}: {e}")

         # Create a new properties dictionary to avoid direct mutation
        containment_data = containment_info.get(polygon_id, {'is_contained': False, 'num_contained': 0})
        new_properties = dict(feature['properties'])
        new_properties.update({
            'poly_id': polygon_id,
            'Area': area,
            'TreeHeight': height,
            'Centroid': {'x': float(centroid[0]), 'y': float(centroid[1])},  # Ensure JSON compatibility
            'is_contained': containment_data['is_contained'],
            'num_contained': containment_data['num_contained'],
            'MeanNDVI': mean_ndvi[i],
            'MaxNDVI': max_ndvi[i],
            'MinNDVI': min_ndvi[i]
        })

        new_feature = {
            'type': 'Feature',
            'properties': new_properties,
            'geometry': {
                'type': feature['geometry']['type'],
                'coordinates': rounded_coords
            }
        }

        updated_features.append(new_feature)
    return updated_features

def get_centroid_gpu(polygon_x_gpu, polygon_y_gpu):
    # Example: Calculate the centroid using CuPy (replace with your actual method)
    x_mean = cp.mean(polygon_x_gpu)
    y_mean = cp.mean(polygon_y_gpu)
    return cp.array([x_mean, y_mean])

def process_geojson(data, confidence_threshold, containment_threshold, height_data_path, rgbi_data_path, area_threshold=3):
    """
    Process a GeoJSON object to update features with additional properties based on containment and height data.

    Args:
        data (dict): GeoJSON object with features to process.
        confidence_threshold (float): Minimum confidence score required to include a feature.
        containment_threshold (float): Threshold for polygon containment.
        height_data_path (str): Path to the raster file containing height data.

    Returns:
        dict: Updated GeoJSON object with additional properties.
    """
    features = data['features']

    # 1. Filter features based on confidence score
    filtered_features = [
        feature for feature in features
        if feature['properties'].get('Confidence_score') is not None and
           float(feature['properties'].get('Confidence_score', 0)) >= confidence_threshold
    ]

    # 2. Add polygon IDs and calculate polygon areas
    id_to_area = {}
    i = 0
    for feature in filtered_features:
        polygon = shape(feature['geometry']).simplify(0.5)
        area = calculate_area(polygon)
        polygon_id = str(i)
        feature['properties']['poly_id'] = polygon_id  # TODO modify this to use a deepcopy
        id_to_area[polygon_id] = area
        i += 1
    polygon_dict = {
        feature['properties']['poly_id']: shape(feature['geometry'])
        for feature in filtered_features
    }

    # Exit early if no polygons exist
    if not polygon_dict:
        return {'type': 'FeatureCollection', 'features': []}

    # 2.1 Filter out small polygons
    new_features = [feature for feature in filtered_features if id_to_area[feature['properties']['poly_id']] >= area_threshold]
    filtered_features = new_features

    # Precompute the set of poly_ids from filtered_features
    filtered_ids = {feature['properties']['poly_id'] for feature in filtered_features}

    # Now filter polygon_dict based on the precomputed set of poly_ids
    polygon_dict = {key: value for key, value in polygon_dict.items() if key in filtered_ids}

    # Prepare confidence scores for selection
    confidence_scores = {feature['properties']['poly_id']: feature['properties']['Confidence_score'] for feature in filtered_features}

    # TODO Make this config parameters
    iou_threshold = 0.7
    area_threshold = 0.5

    # 3. Apply filtering to keep only selected polygons
    retained_ids = filter_polygons_by_iou_and_area(polygon_dict, id_to_area, confidence_scores, iou_threshold, area_threshold)
    filtered_features = [feature for feature in filtered_features if feature['properties']['poly_id'] in retained_ids]
    # Continue with remaining processing steps
    with rasterio.open(height_data_path) as src:
        height_data = src.read(1)
        height_transform = src.transform
        height_width_tif, height_height_tif = src.width, src.height
        height_bounds = src.bounds

    with rasterio.open(rgbi_data_path) as src:
        rgbi_data = src.read()
        ndvi_data = ndvi_array_from_rgbi(rgbi_data)
        ndvi_transform = src.transform
        ndvi_bounds = src.bounds



    # 4. Filter polygons more complex based on containment and calculate height data for each polygon
    new_features = process_features(filtered_features, polygon_dict, id_to_area, containment_threshold, height_data, height_transform, ndvi_data, ndvi_transform, height_width_tif, height_height_tif, height_bounds, ndvi_bounds)
    data['features'] = new_features
    return data


def order_properties(feature, schema_properties):
    """
    Order the properties of a feature to match the schema.

    Args:
        feature (dict): Feature whose properties need to be ordered.
        schema_properties (dict): Schema properties defining the correct order.

    Returns:
        dict: Feature with ordered properties.
    """
    ordered_properties = {key: feature['properties'].get(key, None) for key in schema_properties.keys()}
    feature['properties'] = ordered_properties
    return feature


def process_single_file(file_path, processed_file_path, confidence_threshold, containment_threshold, height_data_path, rgbi_data_path):
    """
    Process a single GeoJSON file and save the results to a new file.

    Args:
        file_path (str): Path to the input GeoJSON file.
        processed_file_path (str): Path to save the processed GeoJSON file.
        confidence_threshold (float): Minimum confidence score required to include a feature.
        containment_threshold (float): Threshold for polygon containment.
        height_data_path (str): Path to the raster file containing height data.
    """
    with fiona.open(file_path, 'r') as source:
        features = [to_dict(feature) for feature in source]
        schema = source.schema
        crs = source.crs.to_string()

    # TODO: INTRODUCE BATCHING IN POSTPROCESSING
    features = np.array(features)[:1000]

    data = {
        "type": "FeatureCollection",
        "features": features
    }
    processed_data = process_geojson(data, confidence_threshold, containment_threshold, height_data_path, rgbi_data_path)

    new_schema = schema.copy()
    new_properties_schema = {
        'Confidence_score': 'float',
        'poly_id': 'str',
        'Area': 'float',
        'ContainedCount': 'int',
        'TreeHeight': 'float',
        'Centroid': 'str',
        'is_contained': 'str',
        'num_contained': 'int',
        'MeanNDVI': 'float',
        'MaxNDVI': 'float',
        'MinNDVI': 'float'
    }
    new_schema['properties'] = new_properties_schema

    # Filter features based on the provided conditions
    filtered_features = []
    for feature in processed_data["features"]:
        properties = feature['properties']

        # Convert 'Centroid' to a JSON string if it exists
        if 'Centroid' in feature['properties']:
            feature['properties']['Centroid'] = json.dumps(feature['properties']['Centroid'])

        # Convert all NumPy types to native Python types
        feature['properties'] = convert_to_python_types(feature['properties'])

        # Ensure properties are ordered correctly
        feature = order_properties(feature, new_properties_schema)

        filtered_features.append(feature)

    # Write the filtered features to the new GeoJSON file
    with fiona.open(processed_file_path, 'w', driver='GPKG', schema=new_schema, crs=crs) as dest:
        for feature in filtered_features:
            dest.write(feature)


def process_files_in_directory(directory, height_directory, image_directory, confidence_threshold, containment_threshold, parallel=True, filename_pattern=None):
    """
    Process all GeoJSON files in a directory and save the results.

    Args:
        directory (str): Directory containing GeoJSON files to process.
        height_directory (str): Directory containing corresponding height data files.
        confidence_threshold (float): Minimum confidence score required to include a feature.
        containment_threshold (float): Threshold for polygon containment.
        parallel (bool): Whether to process files in parallel (default is True).
    """
    geojson_files = [f for f in os.listdir(directory) if f.endswith('.gpkg')]

    if filename_pattern is None:
        height_data_pattern = "(\\d+)\\.tif"
        image_pattern = "(\\d+)\\.tif"
    else:
        image_pattern, height_data_pattern = filename_pattern

    if height_data_pattern is None:
        height_data_pattern = "(\\d+)\\.tif"
    if image_pattern is None:
        image_pattern = "(\\d+)\\.tif"
    image_pattern = re.compile(image_pattern)
    height_data_pattern = re.compile(height_data_pattern)


    def find_matching_height_file(base_name):
        """Find a matching height data file based on regex groups from the base name."""
        geojson_match = image_pattern.match(base_name + ".tif")
        if geojson_match:
            geojson_groups = geojson_match.groups()  # Capture groups for matching
            geojson_concat = ''.join(geojson_groups)
            for height_file in os.listdir(height_directory):
                height_match = height_data_pattern.match(height_file)
                if height_match:
                    height_groups = height_match.groups()
                    height_concat = ''.join(
                        height_groups[:len(geojson_groups)])  # Concatenate height groups for comparison
                    # Check if height groups start with geojson groups
                    if height_concat == geojson_concat:
                        return os.path.join(height_directory, height_file)
        return None

    def find_matching_image_file(base_name):
        """Find a matching height data file based on regex groups from the base name."""
        geojson_match = image_pattern.match(base_name + ".tif")
        if geojson_match:
            geojson_groups = geojson_match.groups()  # Capture groups for matching
            geojson_concat = ''.join(geojson_groups)
            for height_file in os.listdir(image_directory):
                height_match = height_data_pattern.match(height_file)
                if height_match:
                    height_groups = height_match.groups()
                    height_concat = ''.join(
                        height_groups[:len(geojson_groups)])  # Concatenate height groups for comparison
                    # Check if height groups start with geojson groups
                    if height_concat == geojson_concat:
                        return os.path.join(image_directory, height_file)
        return None

    if not parallel:
        # Sequential processing
        for filename in geojson_files:
            file_path = os.path.join(directory, filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            height_file_path = find_matching_height_file(base_name)
            image_file_path = find_matching_image_file(base_name)

            if height_file_path and image_file_path:
                processed_file_path = os.path.join(directory, f"processed_{filename}")
                process_single_file(file_path, processed_file_path, confidence_threshold, containment_threshold, height_file_path, image_file_path)
                torch.cuda.empty_cache()
            else:
                warnings.warn(
                    f"Height data file not found for: {filename}, searched pattern for base name: {base_name}")
    else:
        # Parallel processing
        with ThreadPoolExecutor() as executor:

            futures = []
            for filename in geojson_files:
                if filename.startswith("processed_"):
                    continue
                file_path = os.path.join(directory, filename)
                base_name = os.path.splitext(os.path.basename(filename))[0]
                height_file_path = find_matching_height_file(base_name)
                image_file_path = find_matching_image_file(base_name)

                if height_file_path and image_file_path:
                    processed_file_path = os.path.join(directory, f"processed_{filename}")
                    futures.append(executor.submit(process_single_file, file_path, processed_file_path, confidence_threshold, containment_threshold, height_file_path, image_file_path))
                else:
                    warnings.warn(
                        f"Height data file not found for: {filename}, searched pattern for base name: {base_name}")

            # Ensure all futures complete
            for future in futures:
                future.result()


def profile_code():
    """
    Profile the code to analyze performance using cProfile.
    """
    import cProfile
    import pstats
    import io
    pr = cProfile.Profile()
    
    CONFIDENCE_THRESHOLD = 0.3
    CONTAINMENT_THRESHOLD = 0.6

    geojson_directory = '/output/geojson_predictions'
    height_directory = '/data/nDSM_anno'

    pr.enable()
    process_files_in_directory(geojson_directory, height_directory, CONFIDENCE_THRESHOLD, CONTAINMENT_THRESHOLD, parallel=True, filename_pattern=("FDOP20_(\\d+)_(\\d+)_rgbi\\.tif","nDSM_(\\d+)_1km\\.tif"))

    pr.disable()

    s = io.StringIO()
    # Sort by cumulative time and apply the threshold
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(0.1)  # Only display functions above the cumulative time threshold

    print(s.getvalue())


if __name__ == "__main__":
    profile_code()
    exit()
    CONFIDENCE_THRESHOLD = 0.3
    CONTAINMENT_THRESHOLD = 0.9

    geojson_directory = 'output/geojson_predictions'
    height_dir = 'data/nDSM'
    image_directory = 'data/rgb'

    process_files_in_directory(geojson_directory, height_dir, image_directory, CONFIDENCE_THRESHOLD, CONTAINMENT_THRESHOLD,
                               parallel=False)
