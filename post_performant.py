import json
import os
from typing import Tuple
import rasterio
import fiona
import numpy as np
import re
import warnings

from shapely import MultiPolygon
from shapely.geometry import shape, Point, Polygon
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from rtree import index

import cProfile
import pstats
import io

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from numba import njit, prange
import cupy as cp

from cupy.cuda import stream

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

# Decorator to automatically select whether to use GPU or CPU based on availability
def run_on_device(func):
    def wrapper(*args, **kwargs):
        # Check if GPU is available
        if is_gpu_available():
            # Transfer data to GPU
            args = tuple(cp.array(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
            return func(*args, **kwargs)
        else:
            # Fall back to CPU using NumPy
            args = tuple(np.array(arg) if isinstance(arg, cp.ndarray) else arg for arg in args)
            return func(*args, **kwargs)
    return wrapper

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

def is_point_in_polygon(polygon_x, polygon_y, px, py):
    """
    Use the ray-casting algorithm to determine if a point (px, py) is inside a polygon.
    
    Arguments:
    - polygon_x: List of x-coordinates of the polygon's vertices
    - polygon_y: List of y-coordinates of the polygon's vertices
    - px, py: The point to check
    
    Returns:
    - True if the point is inside the polygon, otherwise False.
    """
    num_vertices = len(polygon_x)
    inside = False
    xinters = 0.0
    p1x, p1y = polygon_x[0], polygon_y[0]
    for i in range(num_vertices + 1):
        p2x, p2y = polygon_x[i % num_vertices], polygon_y[i % num_vertices]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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

def is_point_in_polygon_batch(polygon_x, polygon_y, px, py):
    """
    Vectorized check if points are inside the polygon using ray-casting for batches of points.
    Returns a boolean mask where True indicates the point is inside the polygon.
    
    Arguments:
    - polygon_x: (num_vertices, ) x-coordinates of the polygon's vertices
    - polygon_y: (num_vertices, ) y-coordinates of the polygon's vertices
    - px, py: (batch_size, ) x and y coordinates of the points to check
    
    Returns:
    - inside_mask: (batch_size, ) Boolean array where True indicates the point is inside the polygon
    """
    num_vertices = polygon_x.shape[0]
    batch_size = px.shape[0]
    
    # Initialize inside mask
    inside_mask = cp.zeros(batch_size, dtype=cp.bool_)

    # Loop over each edge of the polygon
    for i in range(num_vertices):
        p1x, p1y = polygon_x[i], polygon_y[i]
        p2x, p2y = polygon_x[(i + 1) % num_vertices], polygon_y[(i + 1) % num_vertices]
        
        # Ensure px and py are broadcasted to match the shape of the polygon vertices
        condition = (py[:, None] > cp.minimum(p1y, p2y)) & (py[:, None] <= cp.maximum(p1y, p2y)) & (px[:, None] <= cp.maximum(p1x, p2x))
        
        # Calculate the intersection of the ray with the edge
        xinters = (py[:, None] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
        
        # Update the inside mask by toggling with XOR (ray-casting)
        inside_mask = inside_mask ^ (condition & (px[:, None] <= xinters)).any(axis=1)
    
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
    min_row, max_row = sorted([min(min_row, height-1), max(max_row, 0)])
    min_col, max_col = sorted([min(min_col, width-1), max(max_col, 0)])

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

    # Process each polygon in parallel
    for i in range(num_polygons):
        # Get current polygon coordinates
        polygon_x_i, polygon_y_i = polygon_x[i], polygon_y[i]
        
        # Check if points are inside the polygon using a vectorized function
        inside_mask = is_point_in_polygon_batch(polygon_x_i, polygon_y_i, x_coords_gpu, y_coords_gpu)
        
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

def get_centroid(polygon):
    """
    Calculate the centroid of a polygon.

    Args:
        polygon (shapely.geometry.Polygon): Polygon object.

    Returns:
        tuple: (x, y) Coordinates of the centroid.
    """
    centroid = polygon.centroid
    return (centroid.x, centroid.y)

def get_raster_metadata(raster_file):
    """
    Extract metadata and data from a raster file.

    Args:
        raster_file (str): Path to the raster file.

    Returns:
        tuple: (transform, crs, width, height, dtype, data)
            - transform (Affine): Transformation matrix of the raster.
            - crs (CRS): Coordinate Reference System of the raster.
            - width (int): Width of the raster in pixels.
            - height (int): Height of the raster in pixels.
            - dtype (numpy.dtype): Data type of the raster values.
            - data (numpy.ndarray): 2D array of raster values.
    """
    with rasterio.open(raster_file) as src:
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        dtype = src.dtypes[0]
        data = src.read(1)
    return transform, crs, width, height, dtype, data

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

def calculate_area(polygon):
    """
    Calculate the area of a polygon.

    Args:
        polygon (shapely.geometry.Polygon): Polygon object.

    Returns:
        float: Area of the polygon.
    """
    return polygon.area

def create_spatial_index(polygon_dict):
    """
    Create an Rtree spatial index for the polygons.

    Args:
        polygon_dict (dict): Dictionary with polygon IDs as keys and shapely.geometry.Polygon objects as values.

    Returns:
        index.Index: Rtree spatial index of the polygons.
    """
    idx = index.Index()
    for poly_id, poly in polygon_dict.items():
        bbox = poly.bounds
        idx.insert(id=int(poly_id), coordinates=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
    return idx

def is_contained(polygon, polygon_dict, id_to_area, polygon_id, threshold=0.9, spatial_idx=None):
    """
    Determine which polygons are contained within the given polygon based on area threshold.

    Args:
        polygon (shapely.geometry.Polygon): Polygon to check for containment.
        polygon_dict (dict): Dictionary of polygons with their IDs.
        id_to_area (dict): Dictionary mapping polygon IDs to their areas.
        polygon_id (str): ID of the polygon to check.
        threshold (float): Area threshold for containment (default is 0.9).
        spatial_idx (index.Index, optional): Rtree spatial index for spatial queries.

    Returns:
        list: List of IDs of polygons that are contained within the given polygon.
    """
    contained_ids = []
    polygon_area = id_to_area.get(polygon_id, None)
    if polygon_area is None:
        return contained_ids

    polygon_bbox = polygon.bounds

    if spatial_idx:
        possible_matches = list(spatial_idx.intersection(polygon_bbox))
    else:
        possible_matches = polygon_dict.keys()

    for other_id in possible_matches:
        if str(other_id) != str(polygon_id):
            other = polygon_dict.get(str(other_id))
            if other is None:
                continue

            other_area = id_to_area.get(str(other_id))
            if other_area is None:
                continue

            other_bbox = other.bounds
            if (polygon_bbox[0] > other_bbox[2] or
                polygon_bbox[2] < other_bbox[0] or
                polygon_bbox[1] > other_bbox[3] or
                polygon_bbox[3] < other_bbox[1]):
                continue

            intersection_area = polygon.intersection(other).area
            if intersection_area / other_area > threshold:
                contained_ids.append(str(other_id))

    return contained_ids

def is_surrounded(contained_ids, polygon_id):
    """
    Count how many of the contained polygons surround the given polygon.

    Args:
        contained_ids (list): List of IDs of polygons contained within the given polygon.
        polygon_id (str): ID of the polygon to check.

    Returns:
        int: Count of polygons that surround the given polygon.
    """
    count = 0
    surrounding_ids = []
    if not contained_ids:
        return count, surrounding_ids
    if contained_ids is None or None in contained_ids:
        return count, surrounding_ids
    count = contained_ids.count(str(polygon_id))
    return int(count)

def process_features(features, polygon_dict, id_to_area, containment_threshold, height_data, transform, width, height, bounds):
    polygon_x_all = []
    polygon_y_all = []
    ids_all = []
    
    # Collect all polygons' coordinates and IDs for batch processing
    max_points = 0  # Track the maximum number of points in any polygon
    for feature in features:
        polygon = shape(feature['geometry'])

        # If the polygon is a MultiPolygon, merge it into a single Polygon
        if isinstance(polygon, MultiPolygon):
            polygon =list(polygon.geoms)[0]
        if isinstance(polygon, Polygon):
            # Handle single Polygon
            polygon_x, polygon_y = polygon.exterior.xy
            polygon_x_all.append(polygon_x)
            polygon_y_all.append(polygon_y)
        else:
            print(f'Other type than polygon encountered: {type(polygon)}')
        ids_all.append(feature['properties']['poly_id'])
        max_points = max(max_points, len(polygon_x))  # Update max_points

    def pad_polygon_coords(polygon_coords, max_length):
        # Ensure that polygon_coords is a CuPy array
        polygon_coords = cp.asarray(polygon_coords)
        
        # Pad with NaN to match the required length
        return cp.concatenate([polygon_coords, cp.full((max_length - len(polygon_coords),), cp.nan)])

    # Now use this function to pad the coordinates
    polygon_x_all_padded = [pad_polygon_coords(polygon_x, max_points) for polygon_x in polygon_x_all]
    polygon_y_all_padded = [pad_polygon_coords(polygon_y, max_points) for polygon_y in polygon_y_all]

    # Convert the padded lists into CuPy arrays
    polygon_x_gpu = cp.array(polygon_x_all_padded, dtype=cp.float32)  # Shape: (num_polygons, max_points)
    polygon_y_gpu = cp.array(polygon_y_all_padded, dtype=cp.float32)  # Shape: (num_polygons, max_points)

    height_data_gpu = cp.array(height_data, dtype=cp.float32)

    # Compute centroids for all polygons on GPU (using the batch processing function)
    centroids = get_centroids(polygon_x_gpu, polygon_y_gpu)
    
    # Perform height data lookups for all polygons at once on the GPU
    heights, highest_points = get_height_within_polygon(polygon_x_gpu, polygon_y_gpu, height_data_gpu, transform, width, height, bounds)
    
    # Convert CuPy arrays to NumPy arrays for serialization
    heights = heights.get()  # Convert to NumPy array
    highest_points = highest_points.get()  # Convert to NumPy array
    
    updated_features = []

    for i, feature in enumerate(features):
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
        new_properties = dict(feature['properties'])
        new_properties.update({
            'poly_id': polygon_id,
            'Area': area,
            'TreeHeight': height,
            'Centroid': {'x': float(centroid[0]), 'y': float(centroid[1])}  # Ensure JSON compatibility
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
    print(f"Processed {len(updated_features)} features.")
    return updated_features

# Helper function to calculate centroid on the GPU (if possible)
def get_centroid_gpu(polygon_x_gpu, polygon_y_gpu):
    # Example: Calculate the centroid using CuPy (replace with your actual method)
    x_mean = cp.mean(polygon_x_gpu)
    y_mean = cp.mean(polygon_y_gpu)
    return cp.array([x_mean, y_mean])

def update_feature_visualization(updated_features, contained_ids_per_feature, contained_ids_per_feature_flat, id_to_area, polygon_dict):
    """
    Update the visualization property of each feature based on containment and surrounding rules.

    Args:
        updated_features (list): List of GeoJSON features.
        contained_ids_per_feature (dict): Dictionary mapping feature IDs to contained feature IDs.
        contained_ids_per_feature_flat (list): Flattened list of contained feature IDs.
        id_to_area (dict): Dictionary mapping feature IDs to their areas.
        polygon_dict (dict): Dictionary of polygons with their IDs.

    Returns:
        list: Updated features with modified 'visualize' properties.
    """    
    #TODO Make this new using other features!
    def is_contained(outer_polygon, inner_polygon, threshold=0.9):
        """
        Check if 90% or more of the inner_polygon is contained within the outer_polygon.
        
        Args:
            outer_polygon (shapely.geometry.Polygon): The outer polygon.
            inner_polygon (shapely.geometry.Polygon): The inner polygon.
        
        Returns:
            bool: True if 90% or more of the inner_polygon is within the outer_polygon.
        """
        inner_area = inner_polygon.area
        intersection_area = outer_polygon.intersection(inner_polygon).area
        return (intersection_area / inner_area) >= threshold
    for feature in updated_features:
        feature_id = feature['properties']['poly_id']
        feature['properties']['IsSurrounded'] = is_surrounded(contained_ids_per_feature_flat, feature_id)

        if 'visualize' not in feature['properties']:
            feature['properties']['visualize'] = 1

        if feature['properties']['IsSurrounded'] > 2:
            feature['properties']['visualize'] = 0
            for surrounding_feature in updated_features:
                if feature_id in contained_ids_per_feature[surrounding_feature['properties']['poly_id']]:
                    feature['properties']['ContainedCount'] -= 1

    for feature in updated_features:
        feature_id = feature['properties']['poly_id']
        if feature['properties']['visualize'] == 0:
            continue

        if feature['properties']['ContainedCount'] >= 4:
            feature['properties']['visualize'] = 0
        elif feature['properties']['ContainedCount'] > 0:
            contained_ids = contained_ids_per_feature[feature_id]
            outer_polygon = polygon_dict[str(feature_id)]

            for id in contained_ids:
                inner_polygon = polygon_dict[str(id)]
                if id_to_area[str(id)] < id_to_area[str(feature_id)] and is_contained(outer_polygon, inner_polygon):
                    for feature2 in updated_features:
                        if feature2['properties']['poly_id'] == id:
                            feature2['properties']['visualize'] = 1
                            feature['properties']['visualize'] = 0
                else:
                    for feature2 in updated_features:
                        if feature2['properties']['poly_id'] == id:
                            feature2['properties']['visualize'] = 0
                            feature['properties']['visualize'] = 1

    return updated_features

def process_geojson(data, confidence_threshold, containment_threshold, height_data_path):
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
    features = data['features']#
    # 1. Filter features based on confidence score
    filtered_features = [
        feature for feature in features 
        if feature['properties'].get('Confidence_score') is not None and 
        float(feature['properties'].get('Confidence_score', 0)) >= confidence_threshold
    ]

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

    # Prepare confidence scores for selection
    confidence_scores = {feature['properties']['poly_id']: feature['properties']['Confidence_score'] for feature in filtered_features}
    iou_threshold = 0.9
    area_threshold = 0.5
    # Apply filtering to keep only selected polygons
    retained_ids = filter_polygons_by_iou_and_area(polygon_dict, id_to_area, confidence_scores, iou_threshold, area_threshold)
    filtered_features = [feature for feature in filtered_features if feature['properties']['poly_id'] in retained_ids]
    # Continue with remaining processing steps
    with rasterio.open(height_data_path) as src:
        height_data = src.read(1)
        transform = src.transform
        width_tif, height_tif = src.width, src.height
        bounds = src.bounds

    filtered_features = process_features(filtered_features, polygon_dict, id_to_area, containment_threshold, height_data, transform, width_tif, height_tif, bounds)
    '''
    contained_ids_per_feature_flat = [item for sublist in contained_ids_per_feature.values() for item in sublist]
    updated_features = update_feature_visualization(
        updated_features, contained_ids_per_feature, contained_ids_per_feature_flat, id_to_area, polygon_dict
    )
    print(f"Updated {len(updated_features)} features with additional properties.")
    '''
    data['features'] = filtered_features
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

def process_single_file(file_path, processed_file_path, confidence_threshold, containment_threshold, height_data_path):
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
        features = [feature for feature in source]
        schema = source.schema
        crs = source.crs

    data = {
        "type": "FeatureCollection",
        "features": features
    }
    print(f"Processing {len(features)} features from {file_path}.")
    processed_data = process_geojson(data, confidence_threshold, containment_threshold, height_data_path)

    new_schema = schema.copy()
    new_properties_schema = {
        'Confidence_score': 'float',
        'poly_id': 'str',
        'Area': 'float',
        'ContainedCount': 'int',
        'TreeHeight': 'float',
        'Centroid': 'str',
        'ContainedIDs': 'str',
        'IsSurrounded': 'int',
        'visualize': 'int'
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

def process_files_in_directory(directory, height_directory, confidence_threshold, containment_threshold, parallel=True, filename_pattern=None):
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
                    height_concat = ''.join(height_groups[:len(geojson_groups)])  # Concatenate height groups for comparison
                    # Check if height groups start with geojson groups
                    if height_concat == geojson_concat:                 
                        return os.path.join(height_directory, height_file)
        return None

    if not parallel:
        # Sequential processing
        for filename in geojson_files:
            file_path = os.path.join(directory, filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            height_file_path = find_matching_height_file(base_name)

            if height_file_path:
                processed_file_path = os.path.join(directory, f"processed_{filename}")
                process_single_file(file_path, processed_file_path, confidence_threshold, containment_threshold, height_file_path)
            else:
                warnings.warn(f"Height data file not found for: {filename}, searched pattern for base name: {base_name}")
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

                if height_file_path:
                    processed_file_path = os.path.join(directory, f"processed_{filename}")
                    futures.append(executor.submit(process_single_file, file_path, processed_file_path, confidence_threshold, containment_threshold, height_file_path))
                else:
                    warnings.warn(f"Height data file not found for: {filename}, searched pattern for base name: {base_name}")

            # Ensure all futures complete
            for future in futures:
                future.result()

def profile_code():
    """
    Profile the code to analyze performance using cProfile.
    """
    pr = cProfile.Profile()
    
    CONFIDENCE_THRESHOLD = 0.5
    CONTAINMENT_THRESHOLD = 0.7

    geojson_directory = '/home/jonas/TreeDetection/output/geojson_predictions'
    height_directory = '/home/jonas/TreeDetection/data/nDSM_anno'

    pr.enable()
    process_files_in_directory(geojson_directory, height_directory, CONFIDENCE_THRESHOLD, CONTAINMENT_THRESHOLD, parallel=False, filename_pattern=("FDOP20_(\\d+)_(\\d+)_rgbi\\.tif","nDSM_(\\d+)_1km\\.tif"))

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
    height_directory = 'data/nDSM'


    process_files_in_directory(geojson_directory, height_directory, CONFIDENCE_THRESHOLD, CONTAINMENT_THRESHOLD, parallel=False)
