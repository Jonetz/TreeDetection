import os
import numpy as np
import cupy as cp
import yaml
import geopandas as gpd

"""
Simple utility functions for raster and vector data processing.

Differs from the helpers, as these are more generic functions, primarily used for data processing.
"""

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

def calculate_area(polygon):
    """
    Calculate the area of a polygon.

    Args:
        polygon (shapely.geometry.Polygon): Polygon object.

    Returns:
        float: Area of the polygon.
    """
    return polygon.area

def calculate_iou(batch_boxes1, batch_boxes2):
    """
    Calculate the Intersection over Union (IoU) for two sets of bounding boxes.
    
    Args:
        batch_boxes1 (list): List of bounding boxes in the format [x1, y1, x2, y2].
        batch_boxes2 (list): List of bounding boxes in the format [x1, y1, x2, y2].
        
    Returns:
        numpy.ndarray: 2D array of IoU values for all pairs of boxes.
    """
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

def xy_gpu(raster_transform, y_coords, x_coords):
    """
    Transforms coordinates from raster space to the spatial reference system using the affine transformation matrix.

    Args:
        raster_transform (Affine): The affine transformation matrix.
        y_coords (np.ndarray): Array of y-coordinates.
        x_coords (np.ndarray): Array of x-coordinates.

    Returns:
        tuple: Transformed x and y coordinates.
    """
    # Convert coordinates to cupy arrays for GPU processing
    y_coords_gpu = cp.asarray(y_coords)
    x_coords_gpu = cp.asarray(x_coords)

    # Extract affine transformation parameters
    a, b, c, d, e, f, g, h, i = raster_transform

    # Perform the affine transformation (x', y' = Ax + By + C, Dx + Ey + F)
    # Using GPU with cupy for parallel processing
    transformed_x = a * x_coords_gpu + b * y_coords_gpu + c
    transformed_y = d * x_coords_gpu + e * y_coords_gpu + f

    # Convert the results back to numpy arrays (if needed) or return as cupy arrays for further GPU processing
    return cp.asnumpy(transformed_x), cp.asnumpy(transformed_y)

def calc_iou(shape1, shape2):
    """Calculate the IoU of two shapes."""
    iou = shape1.intersection(shape2).area / shape1.union(shape2).area
    return iou
