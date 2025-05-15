import numpy as np
import rasterio
import os
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon, Point
from scipy.spatial import Voronoi, ConvexHull
from scipy.ndimage import gaussian_filter
from skimage import measure
from rasterio.features import rasterize
import matplotlib.pyplot as plt


def read_height_map(file_path):
    """
    Reads a height map from a TIFF file.

    Parameters:
    - file_path (str): Path to the TIFF file.

    Returns:
    - height_map (np.ndarray): The height map array.
    - transform (Affine): Affine transform for the height map.
    - crs (CRS): Coordinate reference system of the height map.
    """
    with rasterio.open(file_path) as src:
        height_map = src.read(1)
        transform = src.transform
        crs = src.crs
    return height_map, transform, crs

def apply_gaussian_smoothing(height_map, sigma=1):
    """
    Applies Gaussian smoothing to the height map.

    Parameters:
    - height_map (np.ndarray): The height map array.
    - sigma (float): Standard deviation for Gaussian kernel.

    Returns:
    - smoothed_map (np.ndarray): The smoothed height map.
    """
    return gaussian_filter(height_map, sigma=sigma)

def mask_areas_below_threshold(height_map, threshold):
    """
    Masks areas in the height map below a specified threshold.

    Parameters:
    - height_map (np.ndarray): The height map array.
    - threshold (float): Height threshold value.

    Returns:
    - masked_map (np.ndarray): The masked height map with NaN for areas below the threshold.
    """
    valid_areas = height_map > threshold
    return np.where(valid_areas, height_map, np.nan)

def find_tree_crowns(height_map, neighborhood_size=5, min_height=2):
    """
    Finds tree crowns as local maxima in the height map.

    Parameters:
    - height_map (np.ndarray): The height map array.
    - neighborhood_size (int): Size of the neighborhood for maximum filter.
    - min_height (float): Minimum height to consider a crown.

    Returns:
    - crown_indices (np.ndarray): Indices of the local maxima.
    """
    from scipy.ndimage import maximum_filter
    local_maxima = (height_map == maximum_filter(height_map, size=neighborhood_size))
    crown_indices = np.argwhere(local_maxima & (height_map > min_height))
    return crown_indices

def generate_voronoi_diagram(crown_indices, transform):
    """
    Generates a Voronoi diagram based on crown indices.

    Parameters:
    - crown_indices (np.ndarray): Indices of tree crowns.
    - transform (Affine): Affine transform for the height map.

    Returns:
    - vor (Voronoi): The Voronoi diagram.
    """
    crown_points = [Point(transform * (x, y)) for y, x in crown_indices]
    coords = np.array([(point.x, point.y) for point in crown_points])
    vor = Voronoi(coords)
    return vor

def voronoi_polygons(vor, bounding_polygon):
    """
    Converts Voronoi diagram regions to polygons and clips with bounding polygon.

    Parameters:
    - vor (Voronoi): The Voronoi diagram.
    - bounding_polygon (Polygon): Bounding polygon to clip Voronoi regions.

    Returns:
    - polygons (list of Polygon): List of clipped Voronoi polygons.
    """
    result = []
    for region in vor.regions:
        if not -1 in region and region:
            polygon = Polygon([vor.vertices[i] for i in region])
            if polygon.is_valid:
                polygon = polygon.intersection(bounding_polygon)
                result.append(polygon)
    return result

def filter_cells_by_height(height_map, transform, polygons, height_threshold):
    valid_polygons = []
    
    height_map_shape = height_map.shape    

    # Create mask where height is below the threshold
    height_below_threshold_mask = height_map <= height_threshold

    for i, poly in enumerate(polygons, start=1):
        print(f"Processing polygon {i}/{len(polygons)}", end='\r')

        if poly.is_empty:
            continue
        
        # Rasterize the polygon into a boolean mask
        polygon_mask = rasterize(
            [(poly, 1)],
            out_shape=height_map_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)

        # Check if the entire polygon is above the threshold
        if np.all(height_map[polygon_mask] > height_threshold):
            valid_polygons.append(poly)
            continue

        # Extract the masked height values
        masked_height_map = np.where(polygon_mask, height_map, np.nan)
        
        # Create a binary mask where height is above the threshold
        binary_mask = masked_height_map > 2.5

        # Label connected components in the binary mask
        labels = measure.label(binary_mask, connectivity=1)
        
        # Collect points for convex hull calculation
        all_points = []
        all_points, heights = collect_points_for_convex_hull(labels, masked_height_map, transform)
        
        if all_points:
            polygon = generate_convex_hull(all_points)
            
            # Calculate the area of the polygon
            polygon_area = np.sum(polygon_mask)
            # Calculate the area above the threshold
            above_threshold_area = np.sum(binary_mask[polygon_mask])
            
            # Check if at least 80% of the polygon is above the height threshold
            if above_threshold_area / polygon_area >= 0.2:
                valid_polygons.append(polygon)

    return valid_polygons

def collect_points_for_convex_hull(labels, masked_height_map, transform):
    """
    Collects points and heights for convex hull calculation from labeled mask.

    Parameters:
    - labels (np.ndarray): Labeled mask of connected components.
    - masked_height_map (np.ndarray): Height map with applied mask.
    - transform (Affine): Affine transform for the height map.

    Returns:
    - all_points (list of tuples): List of points for convex hull.
    - heights (list of float): List of heights corresponding to the points.
    """
    all_points = []
    heights = []

    for label in np.unique(labels):
        if label == 0:
            continue
        
        mask = labels == label
        coords = np.column_stack(np.nonzero(mask))
        
        if len(coords) > 0:
            polygon_coords = [transform * (x, y) for y, x in coords]
            polygon_heights = masked_height_map[mask]
            all_points.extend(polygon_coords)
            heights.extend(polygon_heights)

    return all_points, heights

def generate_convex_hull(all_points):
    """
    Generates a convex hull polygon from a set of points.

    Parameters:
    - all_points (list of tuples): List of points to form the convex hull.

    Returns:
    - hull_polygon (Polygon): Convex hull polygon.
    """
    if len(all_points) >= 3:
        hull = ConvexHull(np.array(all_points))
        hull_points = [tuple(np.array(all_points)[vertex]) for vertex in hull.vertices]
        if hull_points[0] != hull_points[-1]:
            hull_points.append(hull_points[0])
        hull_polygon = Polygon(hull_points)
        if not hull_polygon.is_valid:
            hull_polygon = hull_polygon.buffer(0)
        return hull_polygon
    return None

def process_clipped_polygon(clipped_polygon):
    """
    Processes and returns valid polygons from a clipped polygon.

    Parameters:
    - clipped_polygon (Polygon or MultiPolygon): Clipped polygon to be processed.

    Returns:
    - polygons (list of Polygon): List of valid polygons.
    """
    polygons = []
    if isinstance(clipped_polygon, Polygon):
        polygons.append(clipped_polygon)
    elif isinstance(clipped_polygon, MultiPolygon):
        polygons.extend(clipped_polygon.geoms)
    return polygons

def create_bounding_polygon(height_map_shape, transform):
    """
    Creates a bounding polygon for the height map.

    Parameters:
    - height_map_shape (tuple): Shape of the height map array.
    - transform (Affine): Affine transform for the height map.

    Returns:
    - bounding_polygon (Polygon): Bounding polygon for the height map.
    """
    bounding_polygon = Polygon([
        (transform.c, transform.f),
        (transform.c + transform.a * height_map_shape[1], transform.f),
        (transform.c + transform.a * height_map_shape[1], transform.f + transform.e * height_map_shape[0]),
        (transform.c, transform.f + transform.e * height_map_shape[0])
    ])
    return bounding_polygon

def save_results(voronoi_polygons, crs, output_gpkg_path):
    """
    Saves Voronoi and refined polygons to a GeoPackage file.

    Parameters:
    - voronoi_polygons (list of Polygon): List of Voronoi polygons.
    - crs (CRS): Coordinate reference system.
    - output_gpkg_path (str): Path to the output GeoPackage file.
    """
    voronoi_gdf = gpd.GeoDataFrame({'geometry': voronoi_polygons}, crs=crs)    
    voronoi_gdf.to_file(output_gpkg_path, layer='voronoi_polygons', driver='GPKG')

def plot_results(valid_polygons):
    """
    Plots valid and cut-off polygons.

    Parameters:
    - valid_polygons (list of Polygon): List of polygons above the threshold.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot valid polygons
    for poly in valid_polygons:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.plot(x, y, 'g-', label='Above Threshold')
        elif poly.geom_type == 'MultiPolygon':
            for single_poly in poly.geoms:
                x, y = single_poly.exterior.xy
                ax.plot(x, y, 'g-', label='Above Threshold')

    # Remove duplicates in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_title('Polygons Above Threshold vs. Cut Off Polygons')
    plt.show()

def process_files(file_paths, neighborhood_size=7, min_height=3, height_threshold=2.5, gaussian_sigma=0.5):
    """
    Processes multiple height map files and generates Voronoi and refined polygons.

    Parameters:
    - file_paths (list of str): List of file paths to height map TIFF files.
    - neighborhood_size (int): Size of the neighborhood for maximum filter.
    - min_height (float): Minimum height for tree crown detection.
    - height_threshold (float): Height threshold for filtering polygons.
    - gaussian_sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
    - all_voronoi_polys (list of Polygon): List of all Voronoi polygons.
    - crs (CRS): Coordinate reference system used.
    """
    all_voronoi_polys = []

    for file_path in file_paths:
        print(f"Processing {file_path}")
        basename = os.path.basename(file_path)
        name = os.path.splitext(basename)[0]

        height_map, transform, crs = read_height_map(file_path)
        smoothed_height_map = apply_gaussian_smoothing(height_map, sigma=gaussian_sigma)
        masked_height_map = mask_areas_below_threshold(smoothed_height_map, height_threshold)

        crown_indices = find_tree_crowns(masked_height_map, neighborhood_size, min_height)
        vor = generate_voronoi_diagram(crown_indices, transform)

        bounding_polygon = create_bounding_polygon(height_map.shape, transform)

        voronoi_polys = voronoi_polygons(vor, bounding_polygon)
        valid_voronoi_polys = filter_cells_by_height(height_map, transform, voronoi_polys, height_threshold)
        save_results(valid_voronoi_polys, crs, os.path.join(output_gpkg_path, f"{name}.gpkg"))

        all_voronoi_polys.extend(valid_voronoi_polys)

    return all_voronoi_polys, crs

# Example usage
file_paths = [r"325695285.tif"]

output_gpkg_path = r"voronoi"
voronoi_polygons_data, crs = process_files(file_paths)
save_results(voronoi_polygons_data, crs, os.path.join(output_gpkg_path, "all.gpkg"))
# plot_results(voronoi_polygons_data)
