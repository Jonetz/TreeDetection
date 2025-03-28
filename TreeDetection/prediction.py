import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import json
from affine import Affine
import aiofiles
import cv2
import torch
import rasterio
import numpy as np
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from detectron2.engine import DefaultPredictor

from TreeDetection.helpers import xy_gpu

class Predictor(DefaultPredictor):
    """
    Predictor class for Detectron2 models with tiling support.
    
    Besides being a model wrapper, this class also supports large scale and asynchronous batching and contour extraction (returning vertices of a polygon, rather than single values).
    """
    def __init__(self, cfg, device_type="cpu", max_batch_size=5, output_dir='./output', exclude_vars=None):
        """
        Initialize the Predictor class.
        
        Args:
            cfg: Detectron Configuration for the model.
            device_type: Device to run the model on (e.g., 'cpu', 'cuda').
            max_batch_size: Maximum number of tiles to process in a batch.
            output_dir: Directory to save predictions.
            exclude_vars: List of variables to exclude from tiles, should be given in the metadatafile, if true, the tile will not be predicted (default: None).
        """
        super().__init__(cfg)
        self.device = device_type
        self.max_batch_size = max_batch_size
        self.output_dir = output_dir
        self.exclude_vars = exclude_vars or []  # Default to an empty list if None
        os.makedirs(self.output_dir, exist_ok=True)
                                        
    async def _save_json_async(self, data, output_file):
        """Helper function to save JSON data asynchronously."""
        async with aiofiles.open(output_file, mode="w") as f:
            await f.write(json.dumps(data))
        
    def __call__(self, tifpath, tilepath):
        """
        Wrapper around the inference class, will predict anything from the tifpath with a tiling as given in tilepath
        
        Args:
            tifpath: a path that should contain every tif that was used in the tiling function.
            tilepath: a path that contains valid metadata to the tiling.
        """
        pred_subdir = os.path.join(self.output_dir, os.path.basename(tifpath).replace('.tif', ''))
        os.makedirs(pred_subdir, exist_ok=True)
        tiles = asyncio.run(self._load_tiles_async(tilepath))

        predictions = []
        batch = []

        with rasterio.open(tifpath) as img:
            for i, tile in enumerate(tiles):
                tile_tensor, tile_info = self._process_tile(tile, img)

                if tile_tensor is not None:
                    batch.append({"image": tile_tensor, **tile_info})

                # Process and save predictions once a batch is full
                if len(batch) >= self.max_batch_size:
                    predictions.extend(asyncio.run(self._process_and_save_batch_async(batch, pred_subdir, tifpath)))
                    batch = []

            # Process remaining tiles
            if batch:
                predictions.extend(asyncio.run(self._process_and_save_batch_async(batch, pred_subdir, tifpath)))

        return predictions
    
    def _filter_excluded_vars(self, tiles):
        """Filter out excluded variables from tile metadata based on self.exclude_vars flags."""
        filtered_tiles = []
        for tile in tiles:
            # Check if any flag in self.exclude_vars is set to True
            exclude_tile = False     
            for flag in self.exclude_vars:
                if tile[flag]:
                    exclude_tile = True
            if exclude_tile:
                continue  # Skip this tile if any exclusion flag is True
            filtered_tile = {k: v for k, v in tile.items() if k not in self.exclude_vars}
            filtered_tiles.append(filtered_tile)
        
        return filtered_tiles
        
    async def _load_tiles_async(self, tilepath):
        """Load tile metadata from JSON files asynchronously, with optional exclusion of variables."""
        json_files = [os.path.join(tilepath, f) for f in os.listdir(tilepath) if f.endswith('.json')]
        with ThreadPoolExecutor() as executor:
            batch_size = 10  # Adjust batch size based on system I/O limits
            results = []
            for i in range(0, len(json_files), batch_size):
                batch = json_files[i:i+batch_size]
                futures = [asyncio.get_event_loop().run_in_executor(executor, self._load_tile_from_file, file) for file in batch]
                results.extend(await asyncio.gather(*futures))
                
        # Exclude specified variables if any
        if self.exclude_vars:
            results = self._filter_excluded_vars(results)
        return results
    
    def _load_tile_from_file(self, file_path):
        """Helper function to load tile data from a single file."""
        with open(file_path, "r") as file:
            metadata = json.load(file)        
            # Extract the bounding box and create a GeoDataFrame
            bbox = box(*metadata["bounds"][:4])
            geo = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")
            coords = self._get_features(geo)
            # Dynamically include exclude_vars in the returned dictionary
            exclude_flags = {var: metadata.get(var, False) for var in self.exclude_vars}
            return {
                "coords": coords,
                "json_name": file_path,
                **exclude_flags,  # Include the exclude flags dynamically
            }


    def _process_tile(self, tile, img):
        """Preprocess a single tile."""
        try:
            coords = tile["coords"]
            
            out_img, _ = mask(img, shapes=coords, crop=True)
            
            rgb = np.dstack((out_img[2], out_img[1], out_img[0]))
            rgb_rescaled = 255 * rgb / 65535 if np.max(out_img[1]) > 255 else rgb
            height, width = rgb_rescaled.shape[:2]
            tile_image = self.aug.get_transform(rgb_rescaled).apply_image(rgb_rescaled)
            tile_tensor = torch.as_tensor(tile_image.astype("float32").transpose(2, 0, 1))
            _, orig_height, orig_width = out_img.shape       
            return tile_tensor, {"orig_height": orig_height, "orig_width": orig_width, "height": height, "width": width, "json_name": tile["json_name"]}
        
        except Exception as e:
            print(f"Error processing tile {tile['json_name']}: {e}")
            return None, None

    async def _process_and_save_batch_async(self, batch, pred_subdir, tifpath):
        """Process a batch of tiles, convert masks to polygons, and save predictions asynchronously."""
        predictions = []
        with torch.no_grad():
            batch_tensors = [{"image": b["image"], "height": b["height"], "width": b["width"]} for b in batch]
            batch_predictions = self.model(batch_tensors)

            # Process predictions concurrently using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = [
                    asyncio.get_event_loop().run_in_executor(executor, self._process_and_save_single, b, pred, pred_subdir, tifpath)
                    for b, pred in zip(batch, batch_predictions)
                ]
                results = await asyncio.gather(*futures)
                predictions.extend([item for sublist in results for item in sublist])

        torch.cuda.empty_cache()
        return predictions

    def _process_and_save_single(self, b, pred, pred_subdir, tifpath):
        """Process and save a single prediction from a batch."""
        orig_height = b["orig_height"]
        orig_width = b["orig_width"]
        output_file = os.path.join(pred_subdir, f"Prediction_{os.path.basename(b['json_name'])}")

        if not pred["instances"].has("pred_masks"):
            print("Warning: no masks given, probably false model!")
            return []

        polygons = []
        scores = []
        categories = []
        for mask, score, category in zip(
            pred["instances"].pred_masks, 
            pred["instances"].scores, 
            pred["instances"].pred_classes
        ):
            # Resize mask
            resized_mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float().to(self.device),
                size=(orig_height, orig_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            
            resized_mask = resized_mask.cpu().numpy().astype(np.uint8)
            
            # Convert mask to polygon
            contours, _ = cv2.findContours(
                resized_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )            
            for contour in contours:
                if contour.size >= 8:  # Minimum polygon size
                    contour = contour.flatten().tolist()
                    if contour[:2] != contour[-2:]:  # Ensure closed polygon
                        contour.extend(contour[:2])
                    
                    # Open the metadata to extract the EPSG and raster transform
                    metadata_path = b['json_name']
                    with open(metadata_path, "r") as meta_file:
                        metadata = json.load(meta_file)

                    epsg = metadata["crs"]
                    raster_transform = Affine(*metadata["transform"])

                    # Convert polygon coordinates to geographical coordinates
                    x_coords, y_coords = xy_gpu(raster_transform, contour[1::2], contour[::2])
                    #polygon = Polygon(zip(x_coords, y_coords))

                    # Append polygon, score, and category to the list
                    polygons.append(list(zip(x_coords, y_coords)))
                    scores.append(float(score))
                    categories.append(int(category))

        # Create COCO-like JSON for each instance
        evaluations = []
        for poly, score, category in zip(polygons, scores, categories):
            evaluations.append({
                "image_id": tifpath,
                "category_id": category,
                "score": score,
                "polygon_coords": [poly]
            })
            
        asyncio.run(self._save_json_async(evaluations, output_file))
        
        return evaluations

    def _get_features(self, gdf: gpd.GeoDataFrame):
        """Extract features for rasterio masking."""
        return [json.loads(gdf.to_json())["features"][0]["geometry"]]