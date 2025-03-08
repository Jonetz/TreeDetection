import os
import re
import warnings

import json
import os
from pathlib import Path
from typing import Tuple
import sys
import rasterio
from fiona.model import to_dict
import fiona

from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from shapely import MultiPolygon
from shapely.geometry import shape, Polygon

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cupy as cp
import torch

from config import get_config, Config
from helpers import ndvi_array_from_rgbi, check_similarity_bounds


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
            print(f"TODO No height points found within polygon {i}. Implementing fallback to centroid.")
            max_coordinates[i] = cp.array([-1, -1])
            max_heights[i] = -1  # Placeholder value, can be adjusted
        else:
            # Find the maximum height and its coordinates
            max_index = cp.argmax(inside_heights)
            max_heights[i] = inside_heights[max_index]
            max_coordinates[i] = inside_coords[max_index]
    
    return max_heights, max_coordinates


def get_ndvi_within_polygon(polygon_x: np.ndarray, polygon_y: np.ndarray, ndvi_data: np.ndarray,
                            transform: np.ndarray, width: int, height: int, bounds: BoundingBox):
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
    # Unpack bounds
    minx, miny, maxx, maxy = bounds

    # Determine raster coordinates for bounding box
    min_col, min_row = geo_to_raster(transform, minx, miny)
    max_col, max_row = geo_to_raster(transform, maxx, maxy)

    # Clamp values within valid raster bounds
    min_row, max_row = sorted([min(min_row, height - 1), max(max_row, 0)])
    min_col, max_col = sorted([min(min_col, width - 1), max(max_col, 0)])

    # Extract height data subset based on bounding box
    subset = ndvi_data[min_row:max_row + 1, min_col:max_col + 1]
    if subset.size == 0:
        print("Error: The subset of height data is empty.")
        return -1, None

        # Prepare points (x, y) for batch processing
    rows, cols = np.meshgrid(np.arange(subset.shape[0]), np.arange(subset.shape[1]), indexing='ij')
    rows, cols = rows.flatten(), cols.flatten()

    # Convert to geo-coordinates for all points
    x_coords, y_coords = raster_to_geo(transform, rows + min_row, cols + min_col)

    # Convert x_coords and y_coords to CuPy arrays for GPU processing
    x_coords_gpu, y_coords_gpu = cp.array(x_coords, dtype=cp.float32), cp.array(y_coords, dtype=cp.float32)

    # Prepare arrays for results
    num_polygons = polygon_x.shape[0]

    min_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)
    max_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)
    mean_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)
    var_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)

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

        scaling_factor = 0.5
        # Compute the radius of the bounding box for the polygon
        radius = ((max_x - min_x) + (max_y - min_y) / 4) * scaling_factor

        # Check if points are inside the polygon using a vectorized function
        inside_mask = is_point_in_polygon_batch(center_x, center_y, radius, x_coords_gpu, y_coords_gpu, )

        # Extract heights and coordinates where points are inside the polygon
        inside_ndvi = subset.flatten()[inside_mask]

        # Increase the radius until we have at least 50 points (or the radius is too large)
        #radius = ((max_x - min_x) + (max_y - min_y) / 4) * scaling_factor
        #inside_mask = is_point_in_polygon_batch(center_x, center_y, radius, x_coords_gpu, y_coords_gpu, )
        #inside_ndvi = subset.flatten()[inside_mask]
        
        if inside_ndvi.shape[0] == 0:
            print(f"TODO No NDVI points found within polygon {i}. Implementing fallback to centroid.")
            min_ndvi_values[i] = -1
            max_ndvi_values[i] = -1
            mean_ndvi_values[i] = -1
            var_ndvi_values[i] = -1
        else:
            mean_ndvi = cp.mean(inside_ndvi)
            var_ndvi = cp.var(inside_ndvi)

            # Handle empty result (fallback to centroid)
            min_index = cp.argmin(inside_ndvi)
            max_index = cp.argmax(inside_ndvi)

            min_ndvi_values[i] = inside_ndvi[min_index]
            max_ndvi_values[i] = inside_ndvi[max_index]
            mean_ndvi_values[i] = mean_ndvi
            var_ndvi_values[i] = var_ndvi

    return min_ndvi_values, max_ndvi_values, mean_ndvi_values, var_ndvi_values

def get_metadata_within_polygon(polygon_x: np.ndarray, polygon_y: np.ndarray, ndvi_data: np.ndarray, height_data: np.ndarray,
                            transform: np.ndarray, width: int, height: int, bounds: BoundingBox):
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
    # Unpack bounds
    minx, miny, maxx, maxy = bounds

    # Determine raster coordinates for bounding box
    min_col, min_row = geo_to_raster(transform, minx, miny)
    max_col, max_row = geo_to_raster(transform, maxx, maxy)

    # Clamp values within valid raster bounds
    min_row, max_row = sorted([min(min_row, height - 1), max(max_row, 0)])
    min_col, max_col = sorted([min(min_col, width - 1), max(max_col, 0)])

    # Extract ndvi & height data subset based on bounding box
    ndvi_subset = ndvi_data[min_row:max_row + 1, min_col:max_col + 1]
    height_subset = height_data[min_row:max_row + 1, min_col:max_col + 1]

    if ndvi_subset.size == 0:
        print("Error: The subset of height data is empty.")
        return -1, None

    # Prepare points (x, y) for batch processing
    rows, cols = np.meshgrid(np.arange(ndvi_subset.shape[0]), np.arange(ndvi_subset.shape[1]), indexing='ij')
    rows, cols = rows.flatten(), cols.flatten()

    # Convert to geo-coordinates for all points
    x_coords, y_coords = raster_to_geo(transform, rows + min_row, cols + min_col)

    # Convert x_coords and y_coords to CuPy arrays for GPU processing
    x_coords_gpu, y_coords_gpu = cp.array(x_coords, dtype=cp.float32), cp.array(y_coords, dtype=cp.float32)

    # Prepare arrays for results
    num_polygons = polygon_x.shape[0]

    min_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)
    max_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)
    mean_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)
    var_ndvi_values = cp.zeros(num_polygons, dtype=cp.float32)

    max_heights = cp.zeros(num_polygons, dtype=cp.float32)
    max_coordinates = cp.zeros((num_polygons, 2), dtype=cp.float32)

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

        scaling_factor = 0.5
        # Compute the radius of the bounding box for the polygon
        radius = ((max_x - min_x) + (max_y - min_y) / 4)

        # Check if points are inside the polygon using a vectorized function
        inside_mask = is_point_in_polygon_batch(center_x, center_y, radius * scaling_factor, x_coords_gpu, y_coords_gpu, )

        # Extract ndvi values where points are inside the polygon
        inside_ndvi = ndvi_subset.flatten()[inside_mask]

        # Check if points are inside the polygon using a vectorized function
        inside_mask = is_point_in_polygon_batch(center_x, center_y, radius, x_coords_gpu,
                                                y_coords_gpu, )

        # Extract height values where points are inside the polygon
        inside_heights = height_subset.flatten()[inside_mask]

        inside_coords = np.column_stack([x_coords_gpu[inside_mask], y_coords_gpu[inside_mask]])

        if inside_ndvi.shape[0] == 0:
            print(f"TODO No points found within polygon {i}. Implementing fallback to centroid.")
            min_ndvi_values[i] = -1
            max_ndvi_values[i] = -1
            mean_ndvi_values[i] = -1
            var_ndvi_values[i] = -1
        else:
            mean_ndvi = cp.mean(inside_ndvi)
            var_ndvi = cp.var(inside_ndvi)

            # Handle empty result (fallback to centroid)
            min_index = cp.argmin(inside_ndvi)
            max_index = cp.argmax(inside_ndvi)

            min_ndvi_values[i] = inside_ndvi[min_index]
            max_ndvi_values[i] = inside_ndvi[max_index]
            mean_ndvi_values[i] = mean_ndvi
            var_ndvi_values[i] = var_ndvi

        if inside_heights.size == 0:
            print(f"TODO No points found within polygon {i}. Implementing fallback to centroid.")
            max_coordinates[i] = cp.array([-1, -1])
            max_heights[i] = -1  # Placeholder value, can be adjusted
        else:
            # Find the maximum height and its coordinates
            max_index = cp.argmax(inside_heights)
            max_heights[i] = inside_heights[max_index]
            max_coordinates[i] = inside_coords[max_index]

    height_values = [max_heights, max_coordinates]
    ndvi_values = [min_ndvi_values, max_ndvi_values, mean_ndvi_values, var_ndvi_values]
    return height_values, ndvi_values


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
    confidences = cp.array([confidence_scores[pid] for pid in ids], dtype=cp.float16)
    areas = cp.array([id_to_area[pid] for pid in ids], dtype=cp.float16)
    
    retained_ids = set(ids)

    # Create an array to track which polygons to remove
    remove_indices = cp.zeros(len(ids), dtype=cp.bool_)

    # Vectorized computation of IoU and area differences
    iou_matrix = calculate_iou(bboxes, bboxes)  # IoU matrix for all pairs
    area_matrix = cp.abs(areas[:, None] - areas) / cp.maximum(areas[:, None], areas)  # Area difference matrix
    
    # Apply IoU and area thresholds to filter polygons
    mask = (iou_matrix > iou_threshold) & (area_matrix < area_threshold)  # Boolean mask for filtering
    
    # Loop through each polygon and retain only the one with the highest confidence per group
    for i in range(len(ids)):
        if remove_indices[i]:
            continue  # Skip already marked polygons

        # Get indices of polygons connected to the current polygon (including self)
        connected = cp.where(mask[i])[0]  # Indices with IoU > threshold and area < threshold

        # Include self in the connected group
        connected = cp.append(connected, i)

        # Find the polygon with the highest confidence in the group
        best_idx = connected[cp.argmax(confidences[connected])]

        # Flag all other polygons in the group for removal
        for j in connected:
            if j != best_idx:
                remove_indices[j] = True  # Flag for removal

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
    Compute the centroids for all polygons in the batch, ignoring NaN values within polygons.
    
    Args:
        polygon_x_gpu (ndarray): x-coordinates of all polygons (padded with NaNs).
        polygon_y_gpu (ndarray): y-coordinates of all polygons (padded with NaNs).
    
    Returns:
        centroids (ndarray): The centroids of all polygons.
    """
    # Compute centroids, ignoring NaNs within each polygon
    centroid_x = cp.nanmean(polygon_x_gpu, axis=1)
    centroid_y = cp.nanmean(polygon_y_gpu, axis=1)

    # Combine centroids into a single array
    centroids = cp.stack((centroid_x, centroid_y), axis=1)
    return centroids

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

def process_containment_features(features, polygon_ids, polygon_bounds, containment_threshold):
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
    j = 0
    for feature in features:
        i =  int(feature['properties']['poly_id'])
        if str(i) not in polygon_ids:
            continue  # Skip invalid features
        new_properties = dict(feature['properties'])
        new_properties['containment_ratio'] = float(containment_ratios[:, j].max().get())  # Maximum containment ratio
        new_properties['is_contained'] = bool(cp.any(is_contained[:, j]).get())  # Is this polygon contained?
        new_properties['num_contained'] = int(num_contained[j])  # How many polygons contain this one?
        j += 1
        updated_features.append({
            'type': 'Feature',
            'properties': new_properties,
            'geometry': feature['geometry']
        })
    return updated_features


def process_features(features, polygon_dict, id_to_area, height_data, height_transform, height_bounds, ndvi_data, ndvi_transform, ndvi_bounds):
    config = Config()

    def preprocess_features(features):
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

        return polygon_x_all, polygon_y_all, ids_all, max_points, polygon_bounds

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

    polygon_x_all, polygon_y_all, ids_all, max_points, polygon_bounds = preprocess_features(features)

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

    if height_transform.almost_equals(ndvi_transform) and check_similarity_bounds(height_bounds, ndvi_bounds):
        print("Process NDVI and Height information together.")
        height_values, ndvi_values = get_metadata_within_polygon(polygon_x_gpu, polygon_y_gpu, ndvi_data_gpu, height_data_gpu, ndvi_transform, height_data.shape[0], height_data.shape[1], ndvi_bounds)

        min_ndvi, max_ndvi, mean_ndvi, var_ndvi = ndvi_values
        heights, highest_points = height_values
    else:
        # Perform height data lookups for all polygons at once on the GPU
        print("Process NDVI and Height information separately.")
        heights, highest_points = get_height_within_polygon(polygon_x_gpu, polygon_y_gpu, height_data_gpu, height_transform, height_data.shape[0],
                                                                        height_data.shape[1], height_bounds)

        # Perform NDVI data lookups for all polygons at once on the GPU, similar to height data lookup
        min_ndvi, max_ndvi, mean_ndvi, var_ndvi = get_ndvi_within_polygon(polygon_x_gpu,
                                                                        polygon_y_gpu,
                                                                        ndvi_data_gpu,
                                                                        ndvi_transform,
                                                                        ndvi_data.shape[0],
                                                                        ndvi_data.shape[1],
                                                                        ndvi_bounds)

    # Preselect features based on the heights and NDVI values to speed up containment processing
    preselected_features = []
    for i, feature in enumerate(features):
        if heights[i] < config.height_threshold:
            # Height is too small, discard it
            continue
        if mean_ndvi[i] < config.ndvi_mean_threshold or var_ndvi[i] > config.ndvi_var_threshold:
            continue
        preselected_features.append(feature)

    #polygon_x_all, polygon_y_all, ids_all, max_points, polygon_bounds = preprocess_features(preselected_features)

    # Call process_containment_features and retrieve attributes
    polygon_bounds = cp.array(polygon_bounds, dtype=cp.float32)
    containment_results = process_containment_features(features, ids_all, polygon_bounds, config.containment_threshold)

    # Extract containment results
    containment_info = {feature['properties']['poly_id']: {'is_contained': feature['properties']['is_contained'],
                                                           'num_contained': feature['properties']['num_contained'],
                                                           'containment_ratio': feature['properties']['containment_ratio']}
                        for feature in containment_results}

    min_ndvi = min_ndvi.get()
    max_ndvi = max_ndvi.get()
    mean_ndvi = mean_ndvi.get()
    var_ndvi = var_ndvi.get()

    # Step to further select the polygons based on containment results
    selected_features = []

    for i, feature in enumerate(preselected_features):
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
            # Case 3.1: If NDVI Values differ by more than 0.05, keep the one with less variance
            if abs(mean_ndvi[features.index(feature)] - mean_ndvi[features.index(other_polygon)]) > 0.05:
                if var_ndvi[i] < var_ndvi[features.index(other_polygon)]:
                    selected_features.append([i, feature])
                else:
                    selected_features.append([int(other_polygon_id), other_polygon])            
            # Case 3.2: Fallback Big area
            elif id_to_area.get(polygon_id, 0) > id_to_area.get(int(other_polygon_id), 0):
                selected_features.append([i, feature])            
        else:
            # Case 4: Does not contain anything, no problem, we keep it
            selected_features.append([i, feature])

    # Convert CuPy arrays to NumPy arrays for serialization
    heights = heights.get()  # Convert to NumPy array
    highest_points = highest_points.get()  # Convert to NumPy array

    updated_features = []

    # We need to save the indices of selected features, to get the right ndvi indices / heights later
    for i, feature in selected_features:
        polygon_id = feature['properties']['poly_id']
        area = id_to_area.get(polygon_id, None)
        # Check if the feature is within the containment threshold and has valid data
        if highest_points[i] is not None:
            height = heights[i]  
        else:
            height = -1

        centroid = centroids[i].get()  # Convert centroid from CuPy to NumPy

        try:
            rounded_coords = round_coordinates(feature['geometry']['coordinates'])
        except Exception as e:
            print(f"Error rounding coordinates for polygon {polygon_id}: {e}")

         # Create a new properties dictionary to avoid direct mutation
        containment_data = containment_info.get(polygon_id, {'is_contained': False, 'num_contained': -1, 'containment_ratio': 0.0})
        new_properties = dict(feature['properties'])
        new_properties.update({
            'poly_id': polygon_id,
            'Area': area,
            'TreeHeight': height,
            'Centroid': {'x': float(centroid[0]), 'y': float(centroid[1])},  # Ensure JSON compatibility
            'is_contained': containment_data['is_contained'],
            'num_contained': containment_data['num_contained'],
            'containment_ratio': containment_data['containment_ratio'],
            'MeanNDVI': mean_ndvi[i],
            'VarNDVI': var_ndvi[i],
            'MaxNDVI': max_ndvi[i],
            'MinNDVI': min_ndvi[i],
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

def process_geojson(data, confidence_threshold, containment_threshold, iou_threshold, area_threshold, height_data_path, rgbi_data_path):
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
    config = Config()
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

    # Continue with remaining processing steps
    height_scaling_factor = config.height_scaling_factor
    with rasterio.open(height_data_path) as src:
        height_data = src.read(
            1,
            out_shape=(1, int(src.height * height_scaling_factor), int(src.width * height_scaling_factor)),
            resampling=Resampling.bilinear
        )
        height_transform = src.transform * src.transform.scale((src.width / height_data.shape[-1]),
                                                             (src.height / height_data.shape[-2]))
        height_width_tif, height_height_tif = src.width, src.height
        height_bounds = src.bounds

    ndvi_scaling_factor = config.ndvi_scaling_factor
    with rasterio.open(rgbi_data_path) as src:
        rgbi_data = src.read(
            out_shape=(src.count, int(src.height * ndvi_scaling_factor), int(src.width * ndvi_scaling_factor)),
            resampling=Resampling.bilinear
        )
        ndvi_data = ndvi_array_from_rgbi(rgbi_data)
        ndvi_transform = src.transform * src.transform.scale((src.width / ndvi_data.shape[-1]), (src.height / ndvi_data.shape[-2]))
        ndvi_bounds = src.bounds

    # 3. Apply filtering to keep only selected polygons
    retained_ids = filter_polygons_by_iou_and_area(polygon_dict, id_to_area, confidence_scores, iou_threshold, area_threshold)
    filtered_features = [feature for feature in filtered_features if feature['properties']['poly_id'] in retained_ids]

    # 4. Filter polygons more complex based on containment and calculate height data for each polygon
    new_features = process_features(filtered_features, polygon_dict, id_to_area, height_data, height_transform, height_bounds, ndvi_data, ndvi_transform, ndvi_bounds)
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


def process_single_file(file_path, processed_file_path, height_data_path, rgbi_data_path):
    """
    Process a single GeoJSON file and save the results to a new file.

    Args:
        file_path (str): Path to the input GeoJSON file.
        processed_file_path (str): Path to save the processed GeoJSON file.
        confidence_threshold (float): Minimum confidence score required to include a feature.
        containment_threshold (float): Threshold for polygon containment.
        height_data_path (str): Path to the raster file containing height data.
    """
    config = Config()

    with fiona.open(file_path, 'r') as source:
        features = [to_dict(feature) for feature in source]
        schema = source.schema
        crs = source.crs.to_string()

    data = {
        "type": "FeatureCollection",
        "features": features
    }
    processed_data = process_geojson(data, config.confidence_threshold, config.containment_threshold, config.iou_threshold, config.area_threshold, height_data_path, rgbi_data_path)

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
        'MinNDVI': 'float',
        'VarNDVI': 'float'
    }
    new_schema['properties'] = new_properties_schema

    # Filter features based on the provided conditions
    filtered_features = []
    for feature in processed_data["features"]:
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


def process_files_in_directory(directory, height_directory, image_directory, parallel=True, filename_pattern=None):
    """
    Process all GeoJSON files in a directory and save the results.

    Args:
        directory (str): Directory containing GeoJSON files to process.
        height_directory (str): Directory containing corresponding height data files.
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

    def find_matching_file(base_name, geojson_pattern, search_pattern, directory):
        """Find a matching height data file based on regex groups from the base name."""
        geojson_match = geojson_pattern.match(base_name + ".tif")
        print("BASENAME ", base_name, geojson_match, search_pattern, directory, filename_pattern)
        if geojson_match:
            geojson_groups = geojson_match.groups()  # Capture groups for matching
            geojson_concat = ''.join(geojson_groups)
            for file in os.listdir(directory):
                search_match = search_pattern.match(file)
                if search_match:
                    search_groups = search_match.groups()
                    search_concat = ''.join(
                        search_groups[:len(geojson_groups)])  # Concatenate height groups for comparison
                    # Check if height groups start with geojson groups
                    if search_concat == geojson_concat:
                        return os.path.join(directory, file)
        return None

    if not parallel:
        # Sequential processing
        for filename in geojson_files:
            file_path = os.path.join(directory, filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            height_file_path = find_matching_file(base_name, image_pattern, height_data_pattern, height_directory)#find_matching_height_file(base_name)
            image_file_path = find_matching_file(base_name, image_pattern, image_pattern, image_directory)#find_matching_image_file(base_name)

            if height_file_path and image_file_path:
                processed_file_path = os.path.join(directory, f"processed_{filename}")
                process_single_file(file_path, processed_file_path, height_file_path, image_file_path)
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
                height_file_path = find_matching_file(base_name, image_pattern, height_data_pattern, height_directory)
                image_file_path = find_matching_file(base_name, image_pattern, image_pattern, image_directory)

                if height_file_path and image_file_path:
                    processed_file_path = os.path.join(directory, f"processed_{filename}")
                    futures.append(executor.submit(process_single_file, file_path, processed_file_path, height_file_path, image_file_path))
                else:
                    warnings.warn(
                        f"Height data file not found for: {filename}, searched pattern for base name: {base_name}")

            # Ensure all futures complete
            for future in futures:
                future.result()