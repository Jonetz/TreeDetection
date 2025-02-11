import os
from glob import glob
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
from shapely.geometry import shape, box
from rasterio.transform import rowcol

from samgeo.common import raster_to_geojson, transform_coords
from detectree2.preprocessing.tiling import tile_data_train
from detectree2.models.outputs import stitch_crowns

# Set up logging
import logging



def save_bounding_boxes_as_geojson(bounding_boxes, crs, output_path):
    """
    Save bounding boxes to a GeoJSON file.

    Args:
        bounding_boxes (list): List of bounding boxes [min x, min y, max x, max y].
        crs (str): Coordinate reference system for the GeoJSON file.
        output_path (str): Path where the GeoJSON file will be saved.
    """
    features = []
    for bbox in bounding_boxes:
        min_x, min_y, max_x, max_y = bbox
        geom = box(min_x, min_y, max_x, max_y)
        features.append({
            "type": "Feature",
            "geometry": json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry'],
            "properties": {}
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {
                "name": crs
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
        

def clean_crowns(crowns, bounding_boxes, box_threshold=0.15, iou_threshold=0.3):
    """
    Clean the crowns by removing any overlapping crowns and crowns that differ largely from the given bounding box.

    Args:
        crowns (gpd.GeoDataFrame): Crowns to be cleaned.
        bounding_boxes (gpd.GeoDataFrame): Bounding boxes of the crowns.
        box_threshold (float, optional): When clipped, the IoU threshold that determines whether the clipped crown is within a certain range around the bounding box. Defaults to 0.15.
        iou_threshold (float, optional): IoU threshold that determines whether crowns are overlapping.

    Returns:
        gpd.GeoDataFrame: Cleaned crowns.
    """
    def calculate_iou(shape1, shape2):
        """Calculate the IoU of two shapes."""
        iou = shape1.intersection(shape2).area / shape1.union(shape2).area
        return iou
    
    # Filter any rows with empty or invalid geometry
    crowns = crowns[~crowns.is_empty & crowns.is_valid]
    
    crowns.reset_index(drop=True, inplace=True)

    # Every crown was prompted by a bounding box as annotation now use this information to see whether the crown is valid or not:
    fitted_crowns = []  # Create an empty GeoDataFrame to store the cleaned crowns

    for _, box in bounding_boxes.iterrows():  # Loop through the crowns
        if box.geometry is None:
            continue
        
        intersecting_rows = crowns[crowns.intersects(box.geometry)]
        
        best_fit = None
        best_iou = 0

        for _, row in intersecting_rows.iterrows():
            iou = calculate_iou(box.geometry, row.geometry)
            if iou > best_iou:
                best_fit = row.geometry
                best_iou = iou
        
        if best_fit is not None and best_iou > iou_threshold:
            # If IoU is high enough, add the best fitting crown
            fitted_crowns.append(gpd.GeoDataFrame({'geometry': [best_fit]}, geometry='geometry'))
        elif best_fit is not None:
            # If not, try to clip and add the clipped crown if it fits well enough
            clipped_crown = best_fit.intersection(box.geometry)
            if not clipped_crown.is_empty:
                iou = calculate_iou(clipped_crown, box.geometry)
                if iou > box_threshold:
                    fitted_crowns.append(gpd.GeoDataFrame({'geometry': [clipped_crown]}, geometry='geometry'))
        else:
            # If no crown fits, add the bounding box
            fitted_crowns.append(gpd.GeoDataFrame({'geometry': [box.geometry]}, geometry='geometry'))



    crowns_out = pd.concat(fitted_crowns, ignore_index=True)
    return crowns_out
    
def clip(tile_data, coords, coord_crs='EPSG:4326', **kwargs):
    """
    Clip a list of bounding boxes to the extent of a raster file.

    Args:
        tile_data (rasterio.io.DatasetReader): The raster file reader with the bounds.
        coords (list): A list of coordinates in the format of [[minx, miny, maxx, maxy], [minx, miny, maxx, maxy], ...].
        coord_crs (str, optional): The coordinate CRS of the input coordinates. Defaults to "EPSG:4326".

    Returns:
        list: A list of pixel coordinates in the format of [[minx, miny, maxx, maxy], ...] from bottom-left to top-right.
    """
    bounds = tile_data.bounds

    clipped_coords = []
    for coord in coords:
        minx, miny, maxx, maxy = coord

        if coord_crs != tile_data.crs:
            minx, miny = transform_coords(minx, miny, coord_crs, tile_data.crs, **kwargs)
            maxx, maxy = transform_coords(maxx, maxy, coord_crs, tile_data.crs, **kwargs)


        # Ensure the bounding box coordinates are correctly ordered
        minx, maxx = sorted([minx, maxx])
        miny, maxy = sorted([miny, maxy])

        minx = max(minx, bounds.left + 2)
        miny = max(miny, bounds.bottom + 2)
        maxx = min(maxx, bounds.right - 2)
        maxy = min(maxy, bounds.top - 2)
        
        if minx < maxx and miny < maxy:
            if coord_crs != tile_data.crs:
                minx, miny = transform_coords(minx, miny, tile_data.crs, coord_crs, **kwargs)
                maxx, maxy = transform_coords(maxx, maxy, tile_data.crs, coord_crs, **kwargs)
            clipped_coords.append([minx, miny, maxx, maxy])

    return clipped_coords
    
def plot_bounding_boxes(tile_file, bounding_boxes, output_path, coord_crs='EPSG:4326'):
    """
    Plot the bounding boxes on the image for debugging purposes and save the plot as an image file.
    Caution: There is a problem with the coordinate system. The bounding boxes are not plotted correctly (due to problems with transfer from crs to pixel coordinates)
    
    Args:
        tile_file (str): Path to the tile file.
        bounding_boxes (list): List of bounding boxes.
        output_path (str): Path to save the plot image.
    """
    print(f"Plotting Bounding boxes {bounding_boxes}")
    new_coords = []
    with rio.open(tile_file) as src:
        # Get the CRS of the raster
        raster_crs = src.crs

        # Translate bounding boxes to image coordinates
        width = src.width
        height = src.height
        for coord in bounding_boxes:
            minx, miny, maxx, maxy = coord
            if coord_crs != raster_crs:
                minx, miny = transform_coords(minx, miny, coord_crs, raster_crs)
                maxx, maxy = transform_coords(maxx, maxy, coord_crs, raster_crs)

            rows1, cols1 = rowcol(src.transform, minx, miny)
            rows2, cols2 = rowcol(src.transform, maxx, maxy)

            new_coords.append([cols1, rows1, cols2, rows2])

        result = []
        for coord in new_coords:
            minx, miny, maxx, maxy = coord

            minx = max(0, minx)
            miny = max(0, miny)
            maxx = min(width, maxx)
            maxy = min(height, maxy)
            result.append([minx, miny, maxx, maxy])

        # Plot
        image = src.read([1, 2, 3])
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(np.transpose(image, (1, 2, 0)))

        for bbox in result:        
            minx, miny, maxx, maxy = bbox
            width =  minx - maxx
            height = miny- maxy
            print(minx,miny,width,height)
            rect = plt.Rectangle((minx, miny), width, height, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

        plt.savefig(output_path)
        plt.close(fig)
        
def prepare_tile_boxes(geojson_file, tile_data):
    """
    Prepare the bounding boxes of the tile.
    Reads the geojson file and extracts the bounding boxes of the tile, it also clips the boxes to the tile data

    Args:
        geojson_file (str): Path to the geojson file.
        tile_data (str): Path to the tile file.

    Returns:
        list: List of bounding boxes.
    """

    # Read the geojson file
    gdf = gpd.read_file(geojson_file)
    old_size = len(gdf)
    gdf = gdf.to_crs('EPSG:32432')  # making sure CRS match
    if len(gdf) != old_size:
        print('Warning: Some bounding boxes were lost in the conversion to EPSG:32432')
        logger.warn('Warning: The crs of the geojson file was not EPSG:32432')

    # Extract the bounding box of the tile
    bounding_boxes = []
    for _, row in gdf.iterrows():
        bbox = row.geometry.bounds
        bounding_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])

    if len(bounding_boxes) != len(gdf):
        print('Warning: Some bounding boxes were lost while making Bounds')
    len_before_clipping = len(bounding_boxes)
    bounding_boxes = clip(tile_data, bounding_boxes, coord_crs='EPSG:32432')

    # Check if bounding boxes were lost during clipping
    if not bounding_boxes:
        return None
    elif len(bounding_boxes) != len(gdf):
        print('Warning: Some bounding boxes were lost in preparation')
        
    print(f"Bounding boxes reduce from Input, CRS to Clipping {len(gdf)} > {len_before_clipping} > {len(bounding_boxes)}")
    return bounding_boxes

def postprocess_crowns(out_path, filename, tile_width, buffer, bb_gdf, box_threshold=0.5, iou_threshold=0.3):
    """
    Postprocess the crowns by stitching them together and cleaning them.

    Args:
        out_path (str): Path to the output directory.
        filename (str): Name of the file.
        tile_width (int): Width of the tiles.
        buffer (int): Buffer size.
        bb_gdf (gpd.GeoDataFrame, optional): Bounding boxes of the crowns. Defaults to None.
        box_threshold (float, optional): Threshold that determines whether crowns are within a certain range around the bounding box. Defaults to 0.5.
        iou_threshold (float, optional): IoU threshold that determines whether crowns are overlapping. Defaults to 0.3.
    """

    # Stitch the crowns and clean them
    logger.info('Stiching the Crowns together')
    crowns = stitch_crowns(out_path, 1)

    # Save the cleaned crowns to a GeoPackage file
    crowns_path = os.path.join(site_path, 'crowns' )
    crowns_image_path = os.path.join(crowns_path, filename )
    if not os.path.exists(crowns_image_path):
        os.makedirs(crowns_image_path)
    crowns.to_file(os.path.join(crowns_image_path, f"{filename}_T{tile_width}_B{buffer}_refined.gpkg"))

    # Clean the crowns	
    logger.info(f'Cleaning {crowns.size} crowns')
    try:
        cleaned_crowns = clean_crowns(crowns, bb_gdf, box_threshold=box_threshold, iou_threshold=iou_threshold)
        # Save the cleaned crowns to a GeoPackage file
        clean_crowns_path = os.path.join(site_path, 'clean_crowns' )
        clean_crowns_image_path = clean_crowns_path
        #clean_crowns_image_path = os.path.join(clean_crowns_path, filename )
        if not os.path.exists(clean_crowns_image_path):
            os.makedirs(clean_crowns_image_path)
        cleaned_crowns.to_file(os.path.join(clean_crowns_image_path , f"{filename}.gpkg"), driver='GPKG', crs="EPSG:25832")
    except Exception as e:
        print(f'Error: {e}')

def process_image(image, params , sam):
    """
    Process an image by segmenting the crowns and cleaning them.

    Args:
        image (str): Path to the image file.
        json_path (str): Path to the json files.
        tile_path (str): Path where the tile files will be stored.
        mask_path (str): Path where the mask files will be stored. Here the final masks will be stored.
        site_path (str): Path to the site folder.
    """
    

    # Get the parameters
    json_path, tile_path, mask_path, site_path = params['json_path'], params['tile_path'], params['mask_path'], params['site_path']
    buffer, tile_width, tile_height = params['buffer'], params['tile_width'], params['tile_height']
    box_threshold, iou_threshold = params['box_threshold'], params['iou_threshold']

    # Get filename
    filename_withending = os.path.basename(image)
    filename = os.path.splitext(filename_withending)[0]

    # Read in the tiff file
    data = rio.open(image)

    # Read in bounding boxes
    bb_gdf = gpd.read_file(os.path.join(json_path, filename + '.geojson'))

    # Use detetree2 to create tiles
    appends = str(tile_width) + "_" + str(buffer) # this helps keep file structure organised
    tile_folder =  os.path.join(tile_path, "tiles_" + appends)
    tile_location = tile_folder + "/" + filename + "/"
    out_path = os.path.join(mask_path, "tiles_" + appends, filename)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Create tiles
    tile_data_train(data, tile_location, buffer, tile_width, tile_height, bb_gdf, threshold=0.0)

    # Read in the tiles and geojson data
    tile_files = glob(os.path.join(tile_location, '*.tif'))
    geojson_files = []

    for file in tile_files:        
        filename_withending = os.path.basename(file)
        filename_geosjson = os.path.splitext(filename_withending)[0] + "_geo.geojson"
        if not os.path.exists(os.path.join(tile_location, filename_geosjson)):
            print('File {} has no corresponding geojson skipping it!'.format(filename_withending))
            tile_files.remove(file)
        else:
            geojson_files.append(os.path.join(tile_location, filename_geosjson))

    logger.info('Beginning segmentation of file {}'.format(image))
    logger.info('Number of tiles: {}'.format(len(tile_files)))
    logger.info('Saving to location: {}'.format(out_path))
        
        
    # Iterate over the tile and geojson files
    files_processed = -1
    for tile_file, geojson_file in zip(tile_files, geojson_files):
        files_processed += 1
        print('Progress: {}/{} Tiles processed'.format(files_processed, len(tile_files)), end='\r')
            
        # Read the tile file
        tile_data = rio.open(tile_file)

        # Prepare the bounding boxes
        bounding_boxes = prepare_tile_boxes(geojson_file, tile_data)
        
        if bounding_boxes is None:
            continue            

        sam.set_image(tile_file)
        mask_file = os.path.join(out_path, os.path.basename(tile_file).split('.')[0] + ".tif")
        try:
            sam.predict(boxes=bounding_boxes, point_crs='EPSG:32432', output=mask_file)
                
            logger.info(f'Works in file: {tile_file}')
            logger.debug(f'Works in geojson: {geojson_file}')
            logger.debug(f'Works with boxes: {bounding_boxes} (Length: {len(bounding_boxes)})')
        except Exception as e:

            logger.warning(f'Error: {e}')
            logger.debug(f'Error in file: {tile_file}')
            logger.debug(f'Error in geojson: {geojson_file}')
            logger.debug(f'Error with boxes: {bounding_boxes} (Length: {len(bounding_boxes)})')
            continue

        # Save the segmentation as vector data
        mask_geojson = os.path.join(out_path, os.path.basename(tile_file).split('.')[0] + ".geojson")
        raster_to_geojson(mask_file, mask_geojson)
    print('Progress: {}/{} Tiles processed'.format(files_processed+1, len(tile_files)))
    postprocess_crowns(out_path, filename, tile_width, buffer, bb_gdf, box_threshold=box_threshold, iou_threshold=iou_threshold)

def process_images(params, model = "Sam"):
    
    if model=="SamHQ":
        from samgeo.hq_sam import SamGeo# Create a SAM object
        sam = SamGeo(
                model_type='vit_h',
                sam_kwargs=None,
                automatic = False
            )
    elif model=="Sam":
        from samgeo  import SamGeo
        sam = SamGeo(
                model_type='vit_h',
                sam_kwargs=None,
                automatic = False
            )
    else:
        sam = model
    print(os.path.join(params['image_path'], '*.tif'))
    for image in glob(os.path.join(params['image_path'], '*.tif')):
        print(f'Segmenting image: {image}')
        process_image(image, params, sam=sam)
                

if __name__ == '__main__':

    # Set Parameters
    site_path = 'segmentation/BW'
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(site_path, 'logs.log'), format='%(asctime)s %(message)s', level=logging.DEBUG)

    params = {  
        'site_path': site_path,
        'image_path': os.path.join(site_path, 'rgb'),
        'mask_path': os.path.join(site_path, 'masks'),
        'json_path': os.path.join(site_path, 'json'),
        'tile_path': os.path.join(site_path, 'tiles'),
        'buffer': 20, 
        'tile_width': 50,
        'tile_height': 50,
        'box_threshold': 0.6,
        'iou_threshold': 0.4
    }
    process_images(params, model="Sam")