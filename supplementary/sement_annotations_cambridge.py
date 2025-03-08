import os
from glob import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt

from samgeo.common import raster_to_geojson, transform_coords
from detectree2.preprocessing.tiling import tile_data_train
from detectree2.models.outputs import stitch_crowns

# Set up logging
import logging


site_path = 'segmentation/cambridge_eval/'

'''Preprocessing'''
logger = logging.getLogger(__name__)

def clean_crowns(crowns, bounding_boxes, site_path='', box_threshold=0.15, iou_threshold=0.3):
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

        gdf = gpd.GeoDataFrame({'geometry': [box.geometry]}, geometry='geometry')

        for _, row in intersecting_rows.iterrows():
            iou = calculate_iou(box.geometry, row.geometry)
            if iou > best_iou:
                best_fit = row.geometry
                best_iou = iou
        
        if best_fit is not None and best_iou > iou_threshold:
            # If IoU is high enough, add the best fitting crown
            gdf = gpd.GeoDataFrame({'geometry': [best_fit]}, geometry='geometry')
        elif best_fit is not None:
            # If not, try to clip and add the clipped crown if it fits well enough
            clipped_crown = best_fit.intersection(box.geometry)
            if not clipped_crown.is_empty:
                iou = calculate_iou(clipped_crown, box.geometry)
                if iou > box_threshold:
                    gdf = gpd.GeoDataFrame({'geometry': [clipped_crown]}, geometry='geometry')
        # Add this block to handle cases with no intersecting crowns
        if best_fit is None:
            # If no crown fits, add the bounding box
            gdf = gpd.GeoDataFrame({'geometry': [box.geometry]}, geometry='geometry')
        fitted_crowns.append(gdf)
    #Save the bounding boxes
    bounding_boxes.to_file(os.path.join(site_path, 'bounding_boxes.gpkg'), driver='GPKG')


    crowns_out = pd.concat(fitted_crowns)
    return crowns_out

def clip(tile_data: str, coords: list, coord_crs: str = "epsg:4326", **kwargs) -> list:
    """Clip a list of bounding boxes to the extent of a raster file.

    Args:
        tile_data (str): The Data Set reader with the bounds.
        coords (list): A list of coordinates in the format of [[minx, miny, maxx, maxy], [minx, miny, maxx, maxy], ...].
        coord_crs (str, optional): The coordinate CRS of the input coordinates. Defaults to "epsg:4326".

    Returns:
        list: A list of pixel coordinates in the format of [[minx, maxy, maxx, miny], ...] from top left to bottom right.
    """
    # Get the bounds of the raster file
    bounds = tile_data.bounds

    # Clip the coordinates to the bounds of the raster file
    clipped_coords = []
    for coord in coords:
        minx, miny, maxx, maxy = coord

        
        bounds_minx, bounds_miny = bounds.left, bounds.bottom
        bounds_maxx, bounds_maxy = bounds.right, bounds.top
        try:
            minx = max(int(minx), bounds_minx + 2)
            miny = max(int(miny), bounds_miny + 2)
            maxx = min(int(maxx), bounds_maxx - 2)
            maxy = min(int(maxy), bounds_maxy - 2)
        except OverflowError:
            print("Overflow Error Skipping this bound!")
            print(" Tile Crs, Coord_crs", tile_data.crs, coord_crs)
            print("Bounds:", bounds_minx, bounds_miny, bounds_maxx, bounds_maxy)
            print("Coords:", minx, miny, maxx, maxy )
            continue

        if  tile_data.crs != coord_crs:
            minx, miny = transform_coords(minx, miny, tile_data.crs, coord_crs, **kwargs)
            maxx, maxy = transform_coords(maxx, maxy, tile_data.crs, coord_crs, **kwargs)

        if minx < maxx and miny < maxy:
            clipped_coords.append([minx, miny, maxx, maxy])

    return clipped_coords

def plot_bounding_boxes(tile_file, bounding_boxes, output_path, coord_crs='epsg:4326'):
    """
    Plot the bounding boxes on the image for debugging purposes and save the plot as an image file.
    Caution: There is a problem with the coordinate system. The bounding boxes are not plotted correctly. (due to problems with transfer from crs to pixel coordinates)
    
    Args:
        tile_file (str): Path to the tile file.
        bounding_boxes (list): List of bounding boxes.
        output_path (str): Path to save the plot image.
    """
    new_coords = []
    with rio.open(tile_file) as src:
        # Translate to image coordinates
        width = src.width
        height = src.height
        for coord in bounding_boxes:
            minx, miny, maxx, maxy = coord
            if coord_crs != src.crs:
                minx, miny = transform_coords(minx, miny, coord_crs, src.crs)
                maxx, maxy = transform_coords(maxx, maxy, coord_crs, src.crs)
                

                rows1, cols1 = rio.transform.rowcol(
                    src.transform, minx, min
                )
                rows2, cols2 = rio.transform.rowcol(
                    src.transform, maxx, maxy
                )

                new_coords.append([cols1, rows1, cols2, rows2])

            else:
                new_coords.append([minx, miny, maxx, maxy])
        result = []
        for coord in new_coords:
            minx, miny, maxx, maxy = coord

            minx = max(0, minx)
            miny = max(0, miny)
            maxx = min(width, maxx)
            maxy = min(height, maxy)
            # Note that map bbox coords is [minx, miny, maxx, maxy] from bottomleft to topright
            # While rasterio bbox coords is [minx, max, maxx, min] from topleft to bottomright
            result.append([minx, miny, maxy, maxx])

        # Plot
        image = src.read([1, 2, 3])
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(np.transpose(image, (1, 2, 0)))

        for bbox in new_coords:
            minx, miny, maxx, maxy = bbox
            width = maxx - minx
            height = maxy - miny
            rect = plt.Rectangle((minx, miny), width, height, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

        plt.savefig(output_path)
        plt.close(fig)

def postprocess_crowns(out_path, site_path, filename, tile_width, buffer, bb_gdf, box_threshold=0.5, iou_threshold=0.3):
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

    # Check if crowns were predicted, if there are no tiles, the crowns will not be stitched
    if not os.path.exists(out_path) or len(glob(os.path.join(out_path, '*.tif'))) == 0:
        return
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
        cleaned_crowns = clean_crowns(crowns, bb_gdf, site_path=site_path, box_threshold=box_threshold, iou_threshold=iou_threshold)
        # Save the cleaned crowns to a GeoPackage file
        clean_crowns_path = os.path.join(site_path, 'clean_crowns' )
        clean_crowns_image_path = clean_crowns_path
        #clean_crowns_image_path = os.path.join(clean_crowns_path, filename )
        if not os.path.exists(clean_crowns_image_path):
            os.makedirs(clean_crowns_image_path)
        cleaned_crowns.to_file(os.path.join(clean_crowns_image_path , f"{filename}_T{tile_width}_B{buffer}_refined_cleaned.json"),  driver='GeoJSON')
    except Exception as e:
        print(f'Error: {e}')

def prepare_tile_boxes(geojson_file, tile_data, crs='epsg:4326'):
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
    gdf = gdf.to_crs(crs) # making sure CRS match
    for _, row in gdf.iterrows():
        if float("inf") in row.geometry.bounds:
            logger.error("Bounding boxes contain infity values, please call with another crs!")

    # Extract the bounding box of the tile
    bounding_boxes = []
    for _, row in gdf.iterrows():
        bbox = row.geometry.bounds
        bounding_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])

    bounding_boxes = clip(tile_data, bounding_boxes, coord_crs=crs)

    # This is a workaround to not change the library
    if len(bounding_boxes) == 0:
        return None

    return bounding_boxes

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
    bb_gdf = gpd.read_file(os.path.join(json_path, filename + '.json'))

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
            print('File {} has no corresponding geosjon skipping it!'.format(filename_withending))
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
        bounding_boxes = prepare_tile_boxes(geojson_file, tile_data, crs="epsg:4326") # 
        if bounding_boxes is None:
            print('No bounding boxes found in file: {}. Skipping...'.format(geojson_file))
            continue            
        sam.set_image(tile_file)
        mask_file = os.path.join(out_path, os.path.basename(tile_file).split('.')[0] + ".tif")
        try:
            #sam.predict(boxes=bounding_boxes, point_crs=tile_data.crs, output=mask_file)
            sam.predict(boxes=bounding_boxes, point_crs="EPSG:4326", output=mask_file)
            #sam.predict(boxes=bounding_boxes, output=mask_file)

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
        try:
            raster_to_geojson(mask_file, mask_geojson)
        except Exception as e:
            logger.warning(f'Error: {e}')
            logger.debug(f'Error in file: {tile_file}')
            logger.debug(f'Error in geojson: {geojson_file}')
            logger.debug(f'Error with boxes: {bounding_boxes} (Length: {len(bounding_boxes)})')
            continue
    print('Progress: {}/{} Tiles processed'.format(files_processed+1, len(tile_files)))

    postprocess_crowns(out_path, site_path, filename, tile_width, buffer, bb_gdf, box_threshold=box_threshold, iou_threshold=iou_threshold)
    print('Finished processing image: {}'.format(image))

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
    for image in glob(os.path.join(params['image_path'], '*.tif')):
        print(f'Segmenting image: {image}')
        process_image(image, params, sam=sam)
                

if __name__ == '__main__':

    # Set Parameters
    site_path = 'segmentation/BW'
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(site_path, 'logs.log'), format='%(asctime)s %(message)s', encoding='utf-8', level=logging.DEBUG)

    params = {  
        'site_path': site_path,
        'image_path': os.path.join(site_path, 'rgb'),
        'mask_path': os.path.join(site_path, 'masks'),
        'json_path': os.path.join(site_path, 'json'),
        'tile_path': os.path.join(site_path, 'tiles'),
        'buffer': 10, 
        'tile_width': 100,
        'tile_height': 100,
        'box_threshold': 0.5,
        'iou_threshold': 0.4
    }
    process_images(params)