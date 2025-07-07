
from shapely import MultiPolygon, Point, Polygon
import numpy as np

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

def is_valid_geometry(geometry):
    """Check if geometry is valid (not empty and properly constructed)."""
    if geometry.is_empty:
        return False
    if isinstance(geometry, Polygon) or isinstance(geometry, MultiPolygon) or isinstance(geometry, Point):
        return geometry.is_valid
    return False

def get_centroid(geometry):
    """Get the centroid of a geometry, handling both Polygon and MultiPolygon."""
    if isinstance(geometry, MultiPolygon):
        return geometry.representative_point()  # This will give a point within the MultiPolygon
    return geometry.centroid

def angle_to_direction(angle_deg: float) -> str:
    """
    Convert an angle in degrees to a cardinal direction label.
    """
    directions = [
        "north", "northeast", "east", "southeast",
        "south", "southwest", "west", "northwest"
    ]
    idx = int((angle_deg + 22.5) % 360 // 45)
    
def gradient_to_azimuth(dx, dy):
    """
    Converts dx, dy raster gradient to azimuth in degrees.
    dx = ∂height/∂x
    dy = ∂height/∂y
    """
    angle_rad = np.arctan2(dx, -dy)  # negative dy because north is upward
    angle_deg = (np.degrees(angle_rad)) % 360
    return angle_deg
    
def iou(poly1, poly2):
    """Compute Intersection over Union (IoU) between two polygons."""
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return intersection / union if union > 0 else 0


def compute_rectangularity(polygon) -> float:
    if not polygon.is_valid or polygon.is_empty:
        return 0.0

    area_polygon = polygon.area
    min_rect = polygon.minimum_rotated_rectangle
    area_rect = min_rect.area

    if area_rect == 0:
        return 0.0

    rectangularity = area_polygon / area_rect
    return rectangularity