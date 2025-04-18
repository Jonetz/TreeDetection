import os
import numpy as np
import cupy as cp
import yaml

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

def load_prediction_recovery_data(output_path, tiles_path, model_path, logger):
    """
    Loads the recovery data to check which files have been processed already.
    
    Args:
        output_path (str): Path to the output directory.
        tiles_path (str): Path to the directory containing the tiles.
        model_path (str): Path to the model used for prediction.
        logger (logging.Logger): Logger for logging information.
        
    Returns:
        (list, set): A list of unprocessed file paths and a set of processed files.
    """
    recovery_file = os.path.join(output_path, "prediction_recovery.yaml")
    processed_files = set()
    skipped_files = 0
    file_list = []

    if os.path.exists(recovery_file):
        #try:
        with open(recovery_file, "r") as f:
            recovery_data = yaml.safe_load(f)

        # Check if the model_path matches the one stored in the recovery file
        if recovery_data.get("model_path") != model_path:
            logger.warning(f"Model path does not match the one stored in the recovery file. Skipping recovery.")
            return file_list, processed_files

        # Validate the matching of output folder files and metadata in JSON
        for file_path, keys in recovery_data["files"].items():
            json_path = os.path.join(tiles_path, os.path.basename(file_path).replace('.tif', '.json'))
            if os.path.exists(json_path):
                # Validate number of files in the folder and the JSON keys match
                folder_files = len(os.listdir(os.path.join(output_path, os.path.basename(file_path.replace('.tif', '')))))
                if folder_files == len(keys):  # Compare number of files with the number of keys
                    processed_files.add(file_path)
                else:
                    logger.debug(f"Mismatch between output folder and JSON for {file_path}. Probably not all files are processed either intentionally or not.")
            else:
                logger.debug(f"Missing JSON metadata for {file_path}. Skipping.")

        # Filter out already processed files from the list
        original_len = len(file_list)
        file_list = [f for f in file_list if f not in processed_files]
        skipped_files = original_len - len(file_list)
        if skipped_files > 0:
            logger.info(f"Skipped {skipped_files} files that were already processed.")
            
        #except Exception as e:
        #    logger.warning(f"Could not load prediction recovery file: {e}")
    return file_list, processed_files

def save_prediction_recovery_data(output_path, tiles_path, model_path, processed_files, file_list):
    """
    Saves the recovery state after the prediction process.
    
    Args:
        output_path (str): Path to the output directory.
        tiles_path (str): Path to the directory containing the tiles.
        model_path (str): Path to the model used for prediction.
        processed_files (set): A set of processed files.
        file_list (list): The original list of files to process.
    """
    recovery_file = os.path.join(output_path, "prediction_recovery.yaml")
    try:
        recovery_data = {
            "model_path": model_path,
            "files": {}
        }
        files = list(file_list) + list(processed_files)
        for file_path in files:
            json_filename = os.path.basename(file_path).replace('.tif', '.json')
            json_path = os.path.join(tiles_path, json_filename)

            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    json_data = yaml.safe_load(json_file)
                
                # Only store the keys (filenames) from the JSON data
                recovery_data["files"][file_path] = list(json_data.keys())

        with open(recovery_file, "w") as f:
            yaml.safe_dump(recovery_data, f, sort_keys=False)

    except Exception as e:
        print(f"Failed to save prediction recovery file: {e}")
