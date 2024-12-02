import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import json
import cv2
import torch
import rasterio
import numpy as np
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json


class Predictor(DefaultPredictor):
    def __init__(self, cfg, device_type="cpu", max_batch_size=5, output_dir='./output'):
        super().__init__(cfg)
        self.device = device_type
        self.max_batch_size = max_batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def __call__(self, tifpath, tilepath):
        pred_subdir = os.path.join(self.output_dir, os.path.basename(tifpath).replace('.tif', ''))
        os.makedirs(pred_subdir, exist_ok=True)

        # Run loading of tiles asynchronously
        tiles = asyncio.run(self._load_tiles_async(tilepath))
        print(f"Processing {len(tiles)} tiles for {tifpath}")

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

    async def _load_tiles_async(self, tilepath):
        """Load tile metadata from JSON files asynchronously."""
        json_files = [os.path.join(tilepath, f) for f in os.listdir(tilepath) if f.endswith('.json')]
        tiles = []
        # Use ThreadPoolExecutor for blocking I/O (file reading)
        with ThreadPoolExecutor() as executor:
            futures = [asyncio.get_event_loop().run_in_executor(executor, self._load_tile_from_file, file_path) for file_path in json_files]
            results = await asyncio.gather(*futures)
            tiles.extend(results)
        return tiles

    def _load_tile_from_file(self, file_path):
        """Helper function to load tile data from a single file."""
        with open(file_path, "r") as file:
            metadata = json.load(file)
            bbox = box(*metadata["bounds"][:4])
            geo = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")
            coords = self.get_features(geo)
            return {"coords": coords, "json_name": os.path.basename(file_path)}

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
            with torch.amp.autocast(self.device):
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
        output_file = os.path.join(pred_subdir, f"Prediction_{b['json_name']}")

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
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(orig_height, orig_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            
            resized_mask = resized_mask.cpu().numpy().astype(np.uint8)  # Convert to uint8 for cv2
            
            # Convert mask to polygon
            contours, _ = cv2.findContours(
                resized_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if contour.size >= 10:  # Minimum polygon size
                    contour = contour.flatten().tolist()
                    if contour[:2] != contour[-2:]:  # Ensure closed polygon
                        contour.extend(contour[:2])
                    polygons.append(contour)
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
        
        # Save to JSON asynchronously
        with open(output_file, "w") as f:
            json.dump(evaluations, f)

        return evaluations

    def get_features(self, gdf: gpd.GeoDataFrame):
        """Extract features for rasterio masking."""
        return [json.loads(gdf.to_json())["features"][0]["geometry"]]