from concurrent.futures import ThreadPoolExecutor
import re
from typing import Dict
import os
import warnings
import rtree
import torch
import json
import math

import cupy as cp
import numpy as np

from TreeDetection.config import Config
from TreeDetection.postprocessing import process_containment_features

from SolarDetection.buildings import ShapeCache, find_roof_solar, partition_geometries
from SolarDetection.utilities import compute_rectangularity, gradient_to_azimuth, iou, order_properties, convert_to_python_types

import fiona
from fiona.model import to_dict#

import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol

from shapely.geometry import shape

def filter_duplicates(features, iou_threshold=0.5):
    """Filter out duplicate polygons based on IoU threshold."""
    # Sort features by confidence score (descending)
    features = sorted(features, key=lambda f: float(f['properties']['Confidence_score']), reverse=True)

    # Create an R-tree index
    idx = rtree.index.Index()
    polygons = []
    
    for i, feature in enumerate(features):
        polygon = shape(feature['geometry'])
        polygons.append((feature, polygon))
        idx.insert(i, polygon.bounds)
    
    # Keep track of removed features
    to_remove = set()

    for i, (feature1, poly1) in enumerate(polygons):
        if i in to_remove:
            continue

        # Find possible intersections using R-tree
        for j in idx.intersection(poly1.bounds):
            if i == j or j in to_remove:
                continue
            
            feature2, poly2 = polygons[j]
            try:
                if iou(poly1, poly2) > iou_threshold:
                    to_remove.add(j)  # Remove lower-ranked feature
            except Exception as e:
                warnings.warn(f"Error calculating IoU for features {i} and {j}: {e}")
                continue

    return [feature for i, (feature, _) in enumerate(polygons) if i not in to_remove]

def get_height_and_slope_at_centroids(height_path, centroids, buffer_m=1.0):
    with rasterio.open(height_path) as src:
        band = src.read(1, masked=True)
        transform = src.transform
        nodata = src.nodata
        res_x = transform.a
        res_y = -transform.e  # normally positive (cell size)
        px_buffer = int(np.ceil(buffer_m / max(res_x, res_y)))
        window_size = 2 * px_buffer + 1

        results = []
        for x, y in centroids:
            try:
                row, col = rowcol(transform, x, y)
                row_start = max(0, row - px_buffer)
                row_stop = min(src.height, row + px_buffer + 1)
                col_start = max(0, col - px_buffer)
                col_stop = min(src.width, col + px_buffer + 1)

                window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                window_data = src.read(1, window=window, masked=True)

                if window_data.mask.all():
                    results.append((None, None, None, None))
                    continue

                # Mean height
                mean_height = float(window_data.mean())

                if window_data.shape[0] < 3 or window_data.shape[1] < 3:
                    results.append((mean_height, None, None, None))
                    continue

                center = px_buffer
                dzdx = (
                    window_data[center, min(center+1, window_data.shape[1]-1)] -
                    window_data[center, max(center-1, 0)]
                ) / (2 * res_x)

                dzdy = (
                    window_data[min(center+1, window_data.shape[0]-1), center] -
                    window_data[max(center-1, 0), center]
                ) / (2 * res_y)

                slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
                slope_deg = np.degrees(slope_rad)

                results.append((mean_height, slope_deg, float(dzdx), float(dzdy)))
            except Exception:
                results.append((None, None, None, None))
        return results
"""
def get_height_and_slope_at_centroids_plane_fit(height_path, centroids, buffer_m=1.0):
    with rasterio.open(height_path) as src:
        band = src.read(1, masked=True)
        transform = src.transform
        res_x = transform.a
        res_y = abs(transform.e)
        px_buffer = int(np.ceil(buffer_m / max(res_x, res_y)))

        results = []

        for x, y in centroids:
            try:
                row, col = rowcol(transform, x, y)
                row_start = max(0, row - px_buffer)
                row_stop = min(src.height, row + px_buffer + 1)
                col_start = max(0, col - px_buffer)
                col_stop = min(src.width, col + px_buffer + 1)

                window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                window_data = src.read(1, window=window, masked=True)

                if window_data.mask.all():
                    results.append((None, None, None, None, None))
                    continue

                # Get corresponding X, Y positions
                rows, cols = np.meshgrid(
                    np.arange(row_start, row_stop),
                    np.arange(col_start, col_stop),
                    indexing='ij'
                )
                xs, ys = rasterio.transform.xy(transform, rows, cols)
                xs = np.array(xs).flatten()
                ys = np.array(ys).flatten()
                zs = window_data.filled(np.nan).flatten()

                valid = ~np.isnan(zs)
                if valid.sum() < 3:
                    results.append((None, None, None, None, None))
                    continue

                # Fit plane: z = ax + by + c
                A = np.c_[xs[valid], ys[valid], np.ones(valid.sum())]
                coeffs, _, _, _ = np.linalg.lstsq(A, zs[valid], rcond=None)
                a, b, _ = coeffs  # dz/dx, dz/dy

                slope_rad = np.arctan(np.sqrt(a**2 + b**2))
                slope_deg = np.degrees(slope_rad)

                mean_height = float(np.nanmean(zs[valid]))
                height_variance = float(np.nanvar(zs[valid]))

                results.append((mean_height, slope_deg, float(a), float(b), height_variance))

            except Exception:
                results.append((None, None, None, None, None))

        return results
"""
def get_avg_rgb_at_centroids(rgb_path, centroids, buffer_m=1.0):
    results = []
    with rasterio.open(rgb_path) as src:
        transform = src.transform
        res_x = transform.a
        res_y = abs(transform.e)
        px_buffer = int(np.ceil(buffer_m / max(res_x, res_y)))
        for x, y in centroids:
            try:
                row, col = rowcol(transform, x, y)
                row_start = max(0, row - px_buffer)
                row_stop = min(src.height, row + px_buffer + 1)
                col_start = max(0, col - px_buffer)
                col_stop = min(src.width, col + px_buffer + 1)
                win = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                region = src.read([1, 2, 3], window=win)  # shape: (3, H, W)
                if region.shape[1] == 0 or region.shape[2] == 0:
                    results.append((None, None, None))
                    continue
                mean_rgb = region.reshape(3, -1).mean(axis=1)
                results.append(tuple(mean_rgb.round(2)))
            except Exception:
                results.append((None, None, None))
    return results

def filter_height_slope(height_data_path, rgbi_data_path, filtered_features):
    centroids = [feature['properties']['Centroid'] for feature in filtered_features]  

    # TODO Find out if the plane fit variance is useful for filtering out bad polygons 
    #height_slope_1 = get_height_and_slope_at_centroids_plane_fit(height_data_path, centroids)  # should return (height, slope, dx, dy, height_variance)
    heights_slopes = get_height_and_slope_at_centroids(height_data_path, centroids)  # should return (height, slope, dx, dy)
    avg_rgbs = get_avg_rgb_at_centroids(rgbi_data_path, centroids)

    feature_data = []
    for feature, (height, slope, dx, dy), avg_rgb in zip(filtered_features, heights_slopes, avg_rgbs):
        feature['properties']['AvgRGB'] = (
            '#{:02X}{:02X}{:02X}'.format(int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2]))
            if None not in avg_rgb else None
        )
        feature['properties']['Height'] = float(height) if height is not None else None
        feature['properties']['Slope'] = float(slope) if slope is not None else None

        polygon = shape(feature['geometry'])

        # Get slope orientation based on gradient
        if dx is not None and dy is not None:
            tilt_azimuth = (gradient_to_azimuth(dx, dy) + 360) % 360  # We want to have the direction towards the sun.
            feature['properties']['Orientation_deg'] = round(tilt_azimuth, 1)
            directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            idx = int(((tilt_azimuth) % 360) // 45)
            feature['properties']['Orientation_label'] = directions[idx]
        else:
            feature['properties']['Orientation_deg'] = None
            feature['properties']['Orientation_label'] = None

        feature_data.append(feature)

        if height is not None and slope is not None:
            projected_area = polygon.area
            slope_rad = math.radians(slope)
            true_area = projected_area / math.cos(slope_rad)

            feature['properties']['True_Area'] = true_area
    return feature_data

def visualize_features_with_matplotlib(filtered_features, rgbi_data_path: str):
    import numpy as np
    import matplotlib.pyplot as plt
    import rasterio
    from shapely.geometry import shape
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.patches import FancyArrow

    with rasterio.open(rgbi_data_path) as src:
        rgb_image = src.read([1, 2, 3])
        transform = src.transform

    # Move to HWC for plotting
    img = np.moveaxis(rgb_image, 0, -1)

    for feature in filtered_features:
        polygon = shape(feature['geometry'])
        coords = np.array(polygon.exterior.coords)

        # Transform polygon world coords to image pixel coords
        pixel_coords = np.array([~transform * (x, y) for x, y in coords])
        pixel_coords = np.array(pixel_coords)

        # Get tight bounds for plotting
        min_x, min_y = np.floor(pixel_coords.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(pixel_coords.max(axis=0)).astype(int)

        # Clip to image bounds
        height, width = img.shape[:2]
        min_x = max(min_x - 20, 0)
        min_y = max(min_y - 20, 0)
        max_x = min(max_x + 20, width)
        max_y = min(max_y + 20, height)

        sub_img = img[min_y:max_y, min_x:max_x]

        fig, ax = plt.subplots()
        ax.imshow(sub_img)

        # Offset pixel coords to sub-image
        offset_coords = pixel_coords - np.array([min_x, min_y])
        patch = MplPolygon(offset_coords, closed=True, edgecolor='lime', facecolor='none', linewidth=2)
        ax.add_patch(patch)

        # Centroid
        centroid = polygon.centroid
        cx, cy = ~transform * (centroid.x, centroid.y)
        cx -= min_x
        cy -= min_y
        ax.plot(cx, cy, 'bo', markersize=5, label='Centroid')

        # Principal direction arrow
        azimuth = feature['properties'].get('Orientation_deg', None)
        if azimuth is not None:
            angle_rad = np.radians(azimuth)
            dx, dy = 30 * np.sin(angle_rad), -30 * np.cos(angle_rad)  # minus because y is down in image space
            ax.add_patch(FancyArrow(cx, cy, dx, dy, width=2, color='red'))

        ax.set_title(f"Feature ID: {feature['properties'].get('id', 'n/a')}, Orientation: {azimuth:.1f}°" if azimuth is not None else "No orientation")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

def process_geojson(data: Dict, confidence_threshold: float, area_threshold: float, height_data_path: str, rgbi_data_path: str) -> Dict:
    """
    Process a GeoJSON file and return the processed data.
    
    Args:
        data (Dict): GeoJSON data to process.
        confidence_threshold (float): Minimum confidence score required to include a feature.
        area_threshold (float): Minimum area required to include a feature.
        height_data_path (str): Path to the raster file containing height data.
        rgbi_data_path (str): Path to the raster file containing RGBI data.
    
    Returns:
        Dict: Processed GeoJSON data
    """
    config = Config()    
    features = data['features']
    
    if len(features) == 0:
        print("No features to process.")
        return data
    
    # 1. Filter using confidence threshold and area_threshold threshold
    for feature in features:
        feature['properties']['Area'] = shape(feature['geometry']).area

    filtered_features = [
        feature for feature in features
        if feature['properties'].get('Confidence_score') is not None and
           float(feature['properties'].get('Confidence_score', 0)) >= confidence_threshold
    ]
    filtered_features = [
        feature for feature in filtered_features
        if feature['properties']['Area'] >= area_threshold
    ]   

    # 2. Calculate the centroid of the polygon    
    for feature in filtered_features:
        polygon = shape(feature['geometry'])
        centroid = polygon.centroid
        feature['properties']['Centroid'] = (centroid.x, centroid.y)
    
    minx = min([shape(feature['geometry']).bounds[0] for feature in filtered_features])
    miny = min([shape(feature['geometry']).bounds[1] for feature in filtered_features])
    maxx = max([shape(feature['geometry']).bounds[2] for feature in filtered_features])
    maxy = max([shape(feature['geometry']).bounds[3] for feature in filtered_features])
    bbox = (minx, miny, maxx, maxy)
    
    # 3. Advanced filtering of features
    # Filter out duplicates based on IoU
    filtered_features = filter_duplicates(filtered_features, iou_threshold=0.5)

    # Filter out features based on Containment
    all_ids = []
    all_bounds = []
    for i, feature in enumerate(filtered_features):
        feature['properties']['poly_id'] = str(i)
        all_ids.append(str(i))
        all_bounds.append(shape(feature['geometry']).bounds)
    all_bounds = cp.array(all_bounds, dtype=cp.float32)    
    filtered_features = process_containment_features(filtered_features, all_ids, all_bounds, config.containment_threshold)

    # 4. Check if the centroid is in a building shape if possible
    if config.building_shapes is not None:
        filtered_features = find_roof_solar(filtered_features, config.building_shapes, bbox)    
    else:
        for feature in filtered_features:
            feature['properties']['In_building'] = None

    # Filter out features based on Rectangularity ?  
    for feature in filtered_features:
        polygon = shape(feature['geometry'])
        rectangularity = compute_rectangularity(polygon)
        feature['properties']['Rectangularity'] = round(rectangularity, 3)
        if feature['properties']['In_building'] is None or feature['properties']['In_building'] == 1:
            continue
        if feature['properties']['Area'] > 500:
            # If the area is large we assume its a whole field of solar panels
            continue
        if rectangularity < config.rectangularity_threshold:
            filtered_features.remove(feature)
            continue

    # 5. Filter out features based on height and slope (if possible)
    filtered_features = filter_height_slope(height_data_path, rgbi_data_path, filtered_features) #TODO: Add height and slope thresholds
    
    for feature in filtered_features:
        if feature['properties']['Height'] and feature['properties']['Height'] < config.height_threshold:
            filtered_features.remove(feature)
            continue
        if feature['properties']['num_contained'] > 2:
            filtered_features.remove(feature)
            continue

    # Return the processed data, for this return the filtered features
    #visualize_features_with_matplotlib(filtered_features, rgbi_data_path)

    updated_features = []

    # We need to save the indices of selected features, to get the right ndvi indices / heights later
    for feature in filtered_features:
         # Create a new properties dictionary to avoid direct mutation
        centroid = feature['properties']['Centroid']
        new_properties = dict(feature['properties'])
        new_properties.update({
            'Confidence_score': feature['properties']['Confidence_score'],
            'poly_id': feature['properties']['poly_id'],
            'Centroid': {'x': float(centroid[0]), 'y': float(centroid[1])},  # Ensure JSON compatibility
            'Area': feature['properties']['Area'],
            'True_Area': feature['properties'].get('True_Area', None),
            'In_Building': feature['properties']['In_building'],
            'height': feature['properties']['Height'],
            'Slope': feature['properties']['Slope'],
            'AvgRGB': str(feature['properties']['AvgRGB']),
            'Orientation_deg': feature['properties'].get('Orientation_deg', None),
            'Orientation_label': feature['properties'].get('Orientation_label', None),
            'Rectangularity': feature['properties'].get('Rectangularity', None),
            'containment_ratio': feature['properties'].get('containment_ratio', None),
            'num_contained': feature['properties'].get('num_contained', 0),
            'is_contained': feature['properties'].get('is_contained', False)
        })
        new_feature = {
            'type': 'Feature',
            'properties': new_properties,
            'geometry': {
                'type': feature['geometry']['type'],
                'coordinates': feature['geometry']['coordinates']
            }
        }
        updated_features.append(new_feature)
        
    data['features'] = updated_features
    return data

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
    processed_data = process_geojson(data, config.confidence_threshold, config.area_threshold, height_data_path, rgbi_data_path)
    
    new_schema = schema.copy()
    new_properties_schema = {
        'Confidence_score': 'float',
        'poly_id': 'str',
        'Area': 'float',
        'True_Area': 'float',
        'Centroid': 'str',
        'In_Building': 'int',
        'height': 'float',
        'Slope': 'float',
        'AvgRGB': 'str',
        'Orientation_deg': 'float',
        'Orientation_label': 'str',
        'Rectangularity': 'float',
        'containment_ratio': 'float',
        'num_contained': 'int',
        'is_contained': 'str',
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
    Same as in TreeDetection
    """
    geojson_files = [f for f in os.listdir(directory) if f.endswith('.gpkg')]
    geojson_files = [file for file in geojson_files if not file.startswith("processed_")]

    if filename_pattern is None:
        height_data_pattern = "(\\d+)\\.tif"
        image_pattern = "(\\d+)\\.tif"
    else:
        image_pattern, height_data_pattern = filename_pattern

    if height_data_pattern is None:
        height_data_pattern = "(\\d+)\\.tif"
    if image_pattern is None:
        image_pattern = "(\\d+)\\.tif"

    image_merged_pattern = "FDOP20_(\\d+)_(\\d+)_(\\d+)_(\\d+)_rgbi\\.tif"
    height_merged_pattern = "nDSM_(\\d+)(\\d+)_1km\\.tif"
    image_merged_pattern = re.compile(image_merged_pattern)
    height_merged_pattern = re.compile(height_merged_pattern)

    image_pattern = re.compile(image_pattern)
    height_data_pattern = re.compile(height_data_pattern)

    def find_matching_file(base_name, geojson_pattern, search_pattern, directory):
        """Find a matching height data file based on regex groups from the base name."""
        geojson_match = geojson_pattern.match(base_name + ".tif")
        if geojson_match:
            geojson_groups = geojson_match.groups()  # Capture groups for matching
            geojson_concat = ''.join(geojson_groups)
            for root, _, files in os.walk(directory):
                for file in files:
                    search_match = search_pattern.match(file)
                    if search_match:
                        search_groups = search_match.groups()
                        search_concat = ''.join(search_groups[:len(geojson_groups)])  # Concatenate height groups for comparison
                        # Check if height groups start with geojson groups
                        if search_concat == geojson_concat:
                            return os.path.join(root, file)
        return None
    
    config = Config()
    index_dir = os.path.join(config.output_directory, 'building_index')
    index_path = os.path.join(index_dir, "tile_index.gpkg")  
    if not os.path.exists(index_path):  
        os.makedirs(index_dir, exist_ok=True)
        print("Preprocessed building chunks not found. Running preprocessing, this may take a while ...")
        try:
            partition_geometries(config.building_shapes, index_dir, max_per_tile=10000)
        except Exception as e:
            print(f" Failed to partition geometries: {e}")
            return            
    ShapeCache.initialize(index_path)


    if not parallel:
        # Sequential processing
        for filename in geojson_files:
            file_path = os.path.join(directory, filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            height_file_path = find_matching_file(base_name, image_pattern, height_data_pattern, height_directory)
            image_file_path = find_matching_file(base_name, image_pattern, image_pattern, image_directory)

            if height_file_path is None or image_file_path is None:
                height_file_path = find_matching_file(base_name, image_merged_pattern, height_merged_pattern,
                                                      height_directory)
                image_file_path = find_matching_file(base_name, image_merged_pattern, image_merged_pattern,
                                                     image_directory)

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
                file_path = os.path.join(directory, filename)
                base_name = os.path.splitext(os.path.basename(filename))[0]
                height_file_path = find_matching_file(base_name, image_pattern, height_data_pattern, height_directory)
                image_file_path = find_matching_file(base_name, image_pattern, image_pattern, image_directory)

                if height_file_path is None or image_file_path is None:
                    height_file_path = find_matching_file(base_name, image_merged_pattern, height_merged_pattern,
                                                          height_directory)
                    image_file_path = find_matching_file(base_name, image_merged_pattern, image_merged_pattern,
                                                         image_directory)

                if height_file_path and image_file_path:
                    processed_file_path = os.path.join(directory, f"processed_{filename}")
                    futures.append(executor.submit(process_single_file, file_path, processed_file_path, height_file_path, image_file_path))
                else:
                    warnings.warn(
                        f"Height data file not found for: {filename}, searched pattern for base name: {base_name}")

            # Ensure all futures complete
            for future in futures:
                future.result()
    ShapeCache.get_instance().clear()
    torch.cuda.empty_cache()