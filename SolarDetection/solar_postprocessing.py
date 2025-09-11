from concurrent.futures import ThreadPoolExecutor
import re
from typing import Dict
import os
import warnings
import rtree
import torch
import json
import math
import numpy as np

from SolarDetection.config import Config
from SolarDetection.buildings import ShapeCache, find_roof_solar, partition_geometries
from SolarDetection.utilities import compute_rectangularity, gradient_to_azimuth, iou, order_properties, convert_to_python_types
from SolarDetection.containment_processing import filter_containment

import fiona
from fiona.model import to_dict

import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol

from shapely.ops import unary_union
from shapely.geometry import shape, mapping
from shapely import MultiPolygon, Polygon

def filter_duplicates(features, iou_threshold=0.5, merge_threshold=0.9):
    """Filter and optionally merge duplicate polygons based on IoU and confidence."""

    def get_rectangularity(feature):
        return float(feature['properties'].get('Rectangularity', 0))

    def get_confidence(feature):
        return float(feature['properties'].get('Confidence_score', 0))

    # Step 1: Sort features
    rect_ge_075 = [f for f in features if get_rectangularity(f) >= 0.75]
    rect_lt_075 = [f for f in features if get_rectangularity(f) < 0.75]

    rect_ge_075_sorted = sorted(rect_ge_075, key=get_rectangularity)
    rect_lt_075_sorted = sorted(rect_lt_075, key=get_confidence, reverse=True)

    sorted_features = rect_ge_075_sorted + rect_lt_075_sorted

    # Step 2: Build spatial index
    idx = rtree.index.Index()
    polygons = []

    for i, feature in enumerate(sorted_features):
        polygon = shape(feature['geometry'])
        polygons.append((feature, polygon))
        idx.insert(i, polygon.bounds)

    to_remove = set()
    merged_features = {}

    for i, (feature1, poly1) in enumerate(polygons):
        if i in to_remove:
            continue

        for j in idx.intersection(poly1.bounds):
            if i == j or j in to_remove:
                continue

            feature2, poly2 = polygons[j]

            try:
                if iou(poly1, poly2) > iou_threshold:
                    conf1 = get_confidence(feature1)
                    conf2 = get_confidence(feature2)

                    if conf1 > merge_threshold and conf2 > merge_threshold:
                        # Merge geometries
                        merged_geom = unary_union([poly1, poly2])

                        # Ensure result is a Polygon (take largest part if MultiPolygon)
                        if merged_geom.geom_type == 'MultiPolygon':
                            merged_geom = max(merged_geom.geoms, key=lambda g: g.area)
                        elif merged_geom.geom_type != 'Polygon':
                            warnings.warn(f"Merged geometry is of unexpected type: {merged_geom.geom_type}")

                        # Merge properties (average values; more can be added as needed)
                        merged_props = {
                            'Confidence_score': (conf1 + conf2) / 2,
                            'Rectangularity': (get_rectangularity(feature1) + get_rectangularity(feature2)) / 2,
                            'Merged': True
                        }

                        # Copy all other properties from feature1 (optionally: merge further)
                        for k, v in feature1['properties'].items():
                            if k not in merged_props:
                                merged_props[k] = v

                        # Replace feature1 with merged one
                        merged_feature = {
                            'type': 'Feature',
                            'geometry': mapping(merged_geom),
                            'properties': merged_props
                        }

                        polygons[i] = (merged_feature, merged_geom)
                        merged_features[i] = merged_feature
                        to_remove.add(j)

                    else:
                        # Remove the lower-ranked one (j comes later in sort order)
                        to_remove.add(j)
                

            except Exception as e:
                warnings.warn(f"Error calculating IoU for features {i} and {j}: {e}")
                continue

    # Step 3: Return kept and merged features
    return [merged_features.get(i, feature) for i, (feature, _) in enumerate(polygons) if i not in to_remove]

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
        feature['properties']['Height'] = float(height) if height is not None else -1
        feature['properties']['Slope'] = float(slope) if slope is not None else -1

        polygon = shape(feature['geometry'])

        # Get slope orientation based on gradient
        if dx is not None and dy is not None:
            tilt_azimuth = (gradient_to_azimuth(dx, dy) + 360) % 360  # We want to have the direction towards the sun.
            if not math.isnan(tilt_azimuth):
                feature['properties']['Orientation_deg'] = round(tilt_azimuth, 1)
                directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                idx = int(((tilt_azimuth) % 360) // 45)
                feature['properties']['Orientation_label'] = directions[idx]
            else:
                feature['properties']['Orientation_deg'] = None
                feature['properties']['Orientation_label'] = None
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

def process_geojson(data: Dict, height_data_path: str, rgbi_data_path: str) -> Dict:
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

    features = data.get('features', [])
    for i, feature in enumerate(features):
        geom = shape(feature['geometry'])  # Convert GeoJSON dict → Shapely geometry
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
            if isinstance(geom, MultiPolygon):
                largest_poly = max(geom.geoms, key=lambda p: p.area)
                geom = largest_poly
            if not isinstance(geom, Polygon):
                continue        
            if not geom.is_valid:
                continue
        except Exception as e:
            print(f"Failed to fix geometry: {e}")
            continue

        feature['properties']['poly_id'] = feature['properties'].get('poly_id', i)
        feature['geometry'] = mapping(geom)  # Convert back → GeoJSON dict

    if len(features) == 0:
        return data
    
    # 1. Filter using confidence threshold and area_threshold threshold
    for feature in features:
        feature['properties']['Area'] = shape(feature['geometry']).area
    if hasattr(config, 'confidence_threshold'):
        filtered_features = [
            feature for feature in features
            if feature['properties'].get('Confidence_score') is not None and
            float(feature['properties'].get('Confidence_score', 0)) >= config.confidence_threshold
        ]
    if hasattr(config, 'area_threshold'):
        filtered_features = [
            feature for feature in filtered_features
            if feature['properties']['Area'] >= config.area_threshold
        ]   
    # 2. Calculate the centroid & rectangularity of the polygon    
    new_features = []
    for feature in filtered_features:
        polygon = shape(feature['geometry'])
        if polygon.is_empty:
            continue
        centroid = polygon.centroid
        rectangularity = compute_rectangularity(polygon)
        complexity = 1- (polygon.area/polygon.convex_hull.area) if polygon.area > 0 or (polygon.area != polygon.convex_hull.area) else 0        
        if complexity >= 0.3:
            continue
        feature['properties']['Centroid'] = (centroid.x, centroid.y)
        feature['properties']['Rectangularity'] = round(rectangularity, 3)
        feature['properties']['Shape_complexity'] =complexity
        new_features.append(feature)
    filtered_features = new_features
    
    if len(filtered_features) == 0:
        return data
    minx = min([shape(feature['geometry']).bounds[0] for feature in filtered_features])
    miny = min([shape(feature['geometry']).bounds[1] for feature in filtered_features])
    maxx = max([shape(feature['geometry']).bounds[2] for feature in filtered_features])
    maxy = max([shape(feature['geometry']).bounds[3] for feature in filtered_features])
    bbox = (minx, miny, maxx, maxy)

    # 3. Advanced filtering of features
    # Filter out duplicates based on IoU & Rectangularity
    filtered_features = filter_duplicates(filtered_features, iou_threshold=0.6)

    new_features = []
    for feature in filtered_features:
        if 'merged' in feature['properties'] and feature['properties']['merged'] is True:
            # Make Centroid rectangularity and shape complexity and area again
            polygon = shape(feature['geometry'])
            centroid = polygon.centroid
            feature['properties']['Centroid'] = (centroid.x, centroid.y)
            feature['properties']['Rectangularity'] = round(compute_rectangularity(polygon), 3)
            feature['properties']['Shape_complexity'] = 1 - (polygon.area / polygon.convex_hull.area) if polygon.area > 0 or (polygon.area != polygon.convex_hull.area) else 0
            feature['properties']['Area'] = polygon.area      
        new_features.append(feature)
    filtered_features = new_features

    # 4. Check if the centroid is in a building shape if possible
    if config.building_shapes is not None:
        filtered_features = find_roof_solar(filtered_features, config.building_shapes, bbox)    
    else:
        for feature in filtered_features:
            feature['properties']['In_building'] = None
    # Filter out features based on Rectangularity  
    for feature in filtered_features:
        polygon = shape(feature['geometry'])
        if feature['properties']['In_building'] is None or feature['properties']['In_building'] == 1:
            continue
        if feature['properties']['Area'] > 500:
            # If the area is large we assume its a whole field of solar panels
            continue
        if feature['properties']['Rectangularity'] < config.rectangularity_threshold:
            filtered_features.remove(feature)
            continue

    # 5. Filter out features based on height and slope (if possible)
    if not height_data_path or not rgbi_data_path:
        print("Height data or RGBI data path not provided, skipping height and slope filtering.")
    else:
        filtered_features = filter_height_slope(height_data_path, rgbi_data_path, filtered_features) #TODO: Add height and slope thresholds
        if hasattr(config, 'height_threshold'):
            filtered_features = [feature for feature in filtered_features if feature['properties'].get('Height', -1) > config.height_threshold]
        if hasattr(config, 'outside_area_threshold'):
            filtered_features = [feature for feature in filtered_features if feature['properties'].get('In_building', True) \
                                or feature['properties'].get('True_Area', -1) > config.outside_area_threshold]

    # Filter out features based on Containment
    filtered_features = filter_containment(filtered_features, config.containment_threshold)

    # We need to save the indices of selected features, to get the right ndvi indices / heights later
    updated_features = []
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
            'is_contained': feature['properties'].get('is_contained', False),
            'Shape_complexity': feature['properties'].get('Shape_complexity', 0),
            'Containment_coverage': feature['properties'].get('containment_coverage', 0.0)     
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
    processed_data = process_geojson(data, height_data_path, rgbi_data_path)
    
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
        'Shape_complexity': 'float',
        'Containment_coverage': 'float'
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
    config.logger.info(f'Processed {os.path.basename(file_path)} from {len(features)} to {len(filtered_features)}')
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
