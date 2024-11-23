import json
import os
import sys
import time

import rasterio
import fiona
import numpy as np
import re
import warnings
import numba
import shapely
from rasterio._base import Affine
from rasterio.coords import BoundingBox

from shapely.geometry import shape, Point
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from rtree import index

import cProfile
import pstats
import io

from helpers import ndvi_array_from_rgbi


#@numba.jit
def raster_to_geo(transform, row, col):
    """
    Convert raster coordinates to geographical coordinates.

    Args:
        transform (Affine): Transformation matrix for raster to geographical coordinates.
        row (int): Raster row index.
        col (int): Raster column index.

    Returns:
        tuple: (x, y) geographical coordinates corresponding to the raster indices.
    """
    x, y = transform * (col, row)
    return x, y

#@numba.jit
def geo_to_raster(transform, x, y):
    """
    Convert geographical coordinates to raster coordinates.

    Args:
        transform (Affine): Transformation matrix for geographical to raster coordinates.
        x (float): X geographical coordinate.
        y (float): Y geographical coordinate.

    Returns:
        tuple: (row, col) Raster coordinates corresponding to the geographical coordinates.
    """
    row, col = ~transform * (x, y)
    return int(row), int(col)

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

def get_height_within_polygon(polygon, height_data, transform, width, height, bounds):
    """
    Find the maximum height within a polygon from raster height data.

    Args:
        polygon (shapely.geometry.Polygon): Polygon defining the area of interest.
        height_data (numpy.ndarray): 2D array of height data from the raster.
        transform (Affine): Transformation matrix for raster coordinates.
        width (int): Width of the raster in pixels.
        height (int): Height of the raster in pixels.
        bounds (BoundingBox): Bounds of the raster.

    Returns:
        tuple: (max_height, max_coordinates)
            - max_height (float): Maximum height within the polygon.
            - max_coordinates (tuple): (x, y) geographical coordinates of the highest point.
            - (-1, None) if no valid data points are found within the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])

    if maxx < bounds.left or minx > bounds.right or maxy < bounds.bottom or miny > bounds.top:
        print(f"Polygon bounds {minx, miny, maxx, maxy} are out of raster bounds.")
        return -1, None

    min_col, min_row = geo_to_raster(transform, minx, miny)
    max_col, max_row = geo_to_raster(transform, maxx, maxy)
    min_row, max_row = sorted([min_row, max_row])
    min_col, max_col = sorted([min_col, max_col])
    min_row, min_col = max(0, min_row), max(0, min_col)
    max_row, max_col = min(height - 1, max_row), min(width - 1, max_col)

    subset = height_data[min_row:max_row + 1, min_col:max_col + 1]
    heights = []
    coordinates = []

    for r in range(subset.shape[0]):
        for c in range(subset.shape[1]):
            x, y = raster_to_geo(transform, r + min_row, c + min_col)
            if polygon.contains(Point(x, y)):
                heights.append(subset[r, c])
                coordinates.append((x, y))

    if not heights:
        centroid = get_centroid(polygon)
        centroid = Point(centroid[0]-bounds.left, centroid[1]-bounds.bottom)

        min_dist =  np.finfo(np.float64).max
        min_row_dist = -1
        min_col_dist = -1
        coord_x = -1
        coord_y = -1
        for r in range(3):
            for c in range(3):
                x, y = raster_to_geo(transform, min_row - 1 + r, min_col - 1 + c)
                x, y = geo_to_raster(transform, x, y)

                curr_dist = euclidean_distance(centroid, Point(x, 1000 - y))
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_row_dist = r
                    min_col_dist = c
                    coord_x = x
                    coord_y = y

        # TODO runden?
        subset = height_data[min_row - 1:min_row + 2, min_col - 1:min_col + 2]
        heights.append(subset[min_row_dist, min_col_dist])
        coordinates.append((coord_x, coord_y))


    max_height = max(heights)
    max_index = heights.index(max_height)
    max_coordinates = coordinates[max_index]

    return max_height, max_coordinates


def get_ndvi_within_polygon(polygon: shapely.geometry.Polygon, ndvi_data: np.ndarray, transform: Affine, width: int, height: int, bounds: BoundingBox):
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
    minx, miny, maxx, maxy = polygon.bounds

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
    ndvi_values = []

    for r in range(subset.shape[0]):
        for c in range(subset.shape[1]):
            x, y = raster_to_geo(transform, r + min_row, c + min_col)
            if polygon.contains(Point(x, y)):
                ndvi_value = ndvi_data[r + min_row, c + min_col]
                ndvi_values.append(ndvi_value)

    if len(ndvi_values) == 0:
        print(f"No NDVI values found within the polygon. Something went wrong.")

    return np.min(ndvi_values), np.max(ndvi_values), np.mean(ndvi_values)


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



def process_feature(feature, polygon_dict, id_to_area, containment_threshold, ndvi_data, ndvi_transform, ndvi_bounds, height_data, height_transform, height_data_width, height_data_height, height_data_bounds, spatial_idx):
    """
    Process a single feature to update its properties with additional information.

    Args:
        feature (dict): GeoJSON feature to process.
        polygon_dict (dict): Dictionary of polygons with their IDs.
        id_to_area (dict): Dictionary mapping polygon IDs to their areas.
        containment_threshold (float): Threshold for polygon containment.
        height_data (numpy.ndarray): 2D array of height data from the raster.
        height_transform (Affine): Transformation matrix for raster coordinates.
        height_data_width (int): Width of the raster in pixels.
        height_data_height (int): Height of the raster in pixels.
        height_data_bounds (BoundingBox): Bounds of the raster.
        spatial_idx (index.Index): Rtree spatial index of polygons.

    Returns:
        tuple: (new_feature, contained_ids)
            - new_feature (dict): Updated feature with additional properties.
            - contained_ids (list): List of IDs of polygons contained within the processed feature.
    """
    polygon = shape(feature['geometry'])
    polygon_id = feature['properties']['poly_id']

    contained_ids = is_contained(polygon, polygon_dict, id_to_area, polygon_id, containment_threshold, spatial_idx)
    contained_ids.remove(polygon_id) if polygon_id in contained_ids else None
    contained_count = len(contained_ids)

    try:  
        rounded_coords = round_coordinates(feature['geometry']['coordinates'])
    except:
        print(f"Error rounding coordinates for polygon {polygon_id}")

    height_data_height, highest_point = get_height_within_polygon(polygon, height_data, height_transform, height_data_width, height_data_height, height_data_bounds)
    if highest_point is None:
        height_data_height = -1

    min_ndvi, max_ndvi, mean_ndvi = get_ndvi_within_polygon(polygon,
                                                            ndvi_data,
                                                            ndvi_transform,
                                                            ndvi_data.shape[0],
                                                            ndvi_data.shape[1],
                                                            ndvi_bounds)

    centroid = get_centroid(polygon)

    new_properties = dict(feature['properties'])
    new_properties['poly_id'] = polygon_id
    new_properties['Area'] = id_to_area.get(str(polygon_id))
    new_properties['ContainedCount'] = contained_count
    new_properties['TreeHeight'] = float(height_data_height) if height_data_height is not None else None
    new_properties['Centroid'] = {'x': float(centroid[0]), 'y': float(centroid[1])}
    new_properties['MeanNDVI'] = mean_ndvi
    new_properties['MaxNDVI'] = max_ndvi
    new_properties['MinNDVI'] = min_ndvi

    new_feature = {
        'type': 'Feature',
        'properties': new_properties,
        'geometry': {
            'type': feature['geometry']['type'],
            'coordinates': rounded_coords
        }
    }

    return new_feature, contained_ids

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

def process_geojson(data, confidence_threshold, containment_threshold, height_data_path, rgbi_data_path):
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
    filtered_features = [
    feature for feature in features 
    if feature['properties'].get('Confidence_score') is not None and 
    float(feature['properties'].get('Confidence_score', 0)) >= confidence_threshold
    ]

    id_to_area = {}
    i = 0
    for feature in filtered_features:
        polygon = shape(feature['geometry']).simplify(0.5)
        polygon_id = str(i)
        feature['properties']['poly_id'] = polygon_id
        area = calculate_area(polygon)
        id_to_area[polygon_id] = area
        i += 1

    polygon_dict = {feature['properties']['poly_id']: shape(feature['geometry']) for feature in filtered_features}

    if not polygon_dict:
        return {'type': 'FeatureCollection', 'features': []}

    spatial_idx = create_spatial_index(polygon_dict)

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

    # if height_width_tif != nir_width_tif or height_height_tif != nir_height_tif or height_bounds != nir_bounds:
    #     raise ValueError(f"Height and NIR data have different dimensions. Height Width: {height_width_tif}, Height Height: {height_height_tif}, NIR Height: {nir_height_tif}, NIR Width: {nir_width_tif}")
    #     # TODO also check for transform?

    updated_features = []
    contained_ids_per_feature = {}

    for i, feature in enumerate(filtered_features):
        feature, contained_ids = process_feature(feature, polygon_dict, id_to_area, containment_threshold, ndvi_data, ndvi_transform, ndvi_bounds, height_data, height_transform, height_width_tif, height_height_tif, height_bounds, spatial_idx)
        contained_ids_per_feature[feature['properties']['poly_id']] = contained_ids
        feature['properties']['ContainedIDs'] = str(contained_ids)
        updated_features.append(feature)

    contained_ids_per_feature_flat = [item for sublist in contained_ids_per_feature.values() for item in sublist]
    updated_features = update_feature_visualization(updated_features, contained_ids_per_feature, contained_ids_per_feature_flat, id_to_area, polygon_dict)
    

    data['features'] = updated_features
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
        features = [feature for feature in source]
        schema = source.schema
        crs = source.crs

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
        'ContainedIDs': 'str',
        'IsSurrounded': 'int',
        'visualize': 'int',
        'MeanNDVI': 'float',
        'MaxNDVI': 'float',
        'MinNDVI': 'float'
    }
    new_schema['properties'] = new_properties_schema

    # Filter features based on the provided conditions
    filtered_features = []
    for feature in processed_data["features"]:
        properties = feature['properties']
        if (
            float(properties.get('Confidence_score', 0)) > 0.35 and
            3 < float(properties.get('TreeHeight', 0)) < 30 and
            bool(properties.get('visualize', 0)) == 1
        ):
            if 'Centroid' in feature['properties']:
                feature['properties']['Centroid'] = json.dumps(feature['properties']['Centroid'])
            feature = order_properties(feature, new_properties_schema)
            filtered_features.append(feature)

    # Write the filtered features to the new GeoJSON file
    with fiona.open(processed_file_path, 'w', driver='GeoJSON', schema=new_schema, crs=crs) as dest:
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

    geojson_files = [f for f in os.listdir(directory) if f.endswith('.geojson')]

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
            else:
                warnings.warn(f"Height data file not found for: {filename}, searched pattern for base name: {base_name}")
    else:
        # Parallel processing
        with ThreadPoolExecutor() as executor:
            futures = []
            for filename in geojson_files:
                file_path = os.path.join(directory, filename)
                base_name = os.path.splitext(os.path.basename(filename))[0]
                height_file_path = find_matching_height_file(base_name)
                image_file_path = find_matching_image_file(base_name)

                if height_file_path and image_file_path:
                    processed_file_path = os.path.join(directory, f"processed_{filename}")
                    futures.append(executor.submit(process_single_file, file_path, processed_file_path, confidence_threshold, containment_threshold, height_file_path, image_file_path))
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
    pr.enable()
    
    CONFIDENCE_THRESHOLD = 0.7
    CONTAINMENT_THRESHOLD = 0.9

    geojson_directory = 'C:\\Users\\jonas\\Documents\\GitHub\\tree_detection\\postprocessing\\predictions\\raw'
    height_directory = 'C:\\Users\\jonas\\Documents\\GitHub\\tree_detection\\postprocessing\\nDOM'
    image_directory = 'data/rgb'

    process_files_in_directory(geojson_directory, height_directory, image_directory, CONFIDENCE_THRESHOLD, CONTAINMENT_THRESHOLD, parallel=False)

    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
    ps.print_stats()
    print(s.getvalue())

if __name__ == "__main__":
    CONFIDENCE_THRESHOLD = 0.3
    CONTAINMENT_THRESHOLD = 0.9

    geojson_directory = 'output/geojson_predictions'
    height_directory = 'data/nDSM'
    image_directory = 'data/rgb'


    process_files_in_directory(geojson_directory, height_directory, image_directory, CONFIDENCE_THRESHOLD, CONTAINMENT_THRESHOLD, parallel=False)
