import os
import json
import torch
import time
import aiofiles
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.mask import mask
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import asyncio
import numpy as np

class Predictor(DefaultPredictor):
    def __init__(self, cfg, device_type="cpu", max_batch_size=5, output_dir='./output', num_workers=1, memory_threshold=0.8):
        super().__init__(cfg)
        self.device = device_type
        self.max_batch_size = max_batch_size
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.memory_threshold = memory_threshold  # Threshold to control GPU memory usage

    async def __call__(self, tifpath, tilepath):
        pred_subdir = os.path.join(self.output_dir, os.path.basename(tifpath).replace('.tif', ''))
        os.makedirs(pred_subdir, exist_ok=True)

        # Load tiles asynchronously
        tiles = await self._load_tiles(tilepath)

        # Process tiles in parallel batches
        predictions = []
        batch = []
        with rasterio.open(tifpath) as img:
            # Process tiles asynchronously in smaller batches to reduce memory usage
            for i in range(0, len(tiles), self.max_batch_size):
                batch_tiles = tiles[i:i + self.max_batch_size]
                process_tile_tasks = [self._process_tile(tile, img) for tile in batch_tiles]
                results = await asyncio.gather(*process_tile_tasks)

                # Collect valid tiles
                for tile_image, tile_info in results:
                    if tile_image is not None:
                        batch.append({"image": tile_image, **tile_info})

                # Process the batch once it's filled
                if len(batch) >= self.max_batch_size:
                    # Check if there is enough free GPU memory before processing
                    if not self._check_gpu_memory():
                        print("Not enough GPU memory, waiting for some time...")
                        await asyncio.sleep(2)  # Wait and try again
                    else:
                        # Save results after each batch to avoid running out of GPU memory
                        predictions += await self._process_and_save_batch(batch, pred_subdir, tifpath)
                        batch = []  # Reset batch after processing
                #if i % 100 == 0:
                #    print(f"Processed {i}/{len(tiles)} tiles...")

            # Process remaining batch if any
            if batch:
                predictions += await self._process_and_save_batch(batch, pred_subdir, tifpath)

        return predictions

    def _check_gpu_memory(self):
        """Check if there is enough free GPU memory."""
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_percentage = free_memory / torch.cuda.get_device_properties(0).total_memory
        #print(f"Free memory: {free_memory_percentage * 100:.2f}%")
        return free_memory_percentage > self.memory_threshold  # Only proceed if enough memory is available

    async def _load_tiles(self, tilepath):
        """Load and preprocess tile metadata asynchronously."""
        json_files = [os.path.join(tilepath, f) for f in os.listdir(tilepath) if f.endswith('.json')]
        tiles = []

        async def process_file(file_path):
            """Process a single JSON file asynchronously."""
            async with aiofiles.open(file_path, mode="r") as file:
                file_content = await file.read()
                metadata = json.loads(file_content)

                # Process metadata
                bbox = box(*metadata["bounds"][:4])
                geo = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")
                coords = self.get_features(geo)
                return {"coords": coords, "json_name": os.path.basename(file_path)}

        tasks = [process_file(file_path) for file_path in json_files]
        tiles = await asyncio.gather(*tasks)

        return tiles

    async def _process_tile(self, tile, img):
        """Preprocess a single tile asynchronously."""
        try:
            coords = tile["coords"]
            out_img, _ = mask(img, shapes=coords, crop=True)
            out_sumbands = np.sum(out_img, axis=0)

            if np.mean(out_sumbands == 0) > 0.25 or np.mean(np.isnan(out_sumbands)) > 0.25:
                return None, None  # Skip invalid tiles

            rgb = np.dstack((out_img[2], out_img[1], out_img[0]))
            rgb_rescaled = (255 * rgb / 65535).astype(np.uint8) if rgb.max() > 255 else rgb
            tile_image = self.aug.get_transform(rgb_rescaled).apply_image(rgb_rescaled)
            tile_tensor = torch.as_tensor(tile_image.transpose(2, 0, 1).copy(), dtype=torch.float32)

            # Move tensor to CPU after processing to avoid OOM on GPU
            tile_tensor = tile_tensor.cpu()

            return tile_tensor, {"height": tile_tensor.shape[1], "width": tile_tensor.shape[2], "json_name": tile["json_name"]}
        except Exception as e:
            print(f"Error processing tile {tile['json_name']}: {e}")
            return None, None

    async def _process_and_save_batch(self, batch, pred_subdir, tifpath):
        """Process and save a batch of images asynchronously."""
        start_time = time.time()
        predictions = []

        with torch.no_grad():
            # Move all batch tensors to the CPU (to avoid using GPU memory for saving predictions)
            batch_tensors = [{"image": b["image"].cpu(), "height": b["height"], "width": b["width"]} for b in batch]
            batch_predictions = self.model(batch_tensors)

        # Process predictions and save them immediately to avoid using GPU memory for too long
        async def save_prediction(b, prediction):
            """Save a single prediction asynchronously."""
            output_file = os.path.join(pred_subdir, f"Prediction_{b['json_name']}.json")
            evaluations = instances_to_coco_json(prediction["instances"].to("cpu"), tifpath)
            async with aiofiles.open(output_file, "w") as dest:
                await dest.write(json.dumps(evaluations, separators=(',', ':')))

        save_tasks = [save_prediction(b, pred) for b, pred in zip(batch, batch_predictions)]
        predictions = await asyncio.gather(*save_tasks)

        # Clear GPU memory after processing the batch
        del batch_tensors
        del batch_predictions
        torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time
        #print(f"Processed and saved batch of {len(batch)} tiles in {elapsed_time:.2f}s.")
        return predictions

    def get_features(self, gdf: gpd.GeoDataFrame):
        """Parse features from GeoDataFrame for rasterio."""
        return [json.loads(gdf.to_json())["features"][0]["geometry"]]