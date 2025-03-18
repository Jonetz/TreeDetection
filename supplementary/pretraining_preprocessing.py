import os, shutil


import numpy as np
import rasterio
from glob import glob
from detectree2.preprocessing.tiling import get_features
from pathlib import Path
import cv2
import geopandas as gpd
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.mask import mask
from shapely.geometry import box
import shutil
import random

# Set variables
site_path = Path("pretraining/")
rgb_path = site_path / "rgb"
mask_path = site_path / "mask"
tiles_path = site_path / "tiles"
rgb_tiles_path = tiles_path / "rgb_tiles"
mask_tiles_path = tiles_path / "mask_tiles"

test_frac = 0.2
buffer = 200
tile_width = 250
tile_height = 250

if not os.path.exists(rgb_tiles_path):
  os.makedirs(rgb_tiles_path, exist_ok=True)

if not os.path.exists(mask_tiles_path):
  os.makedirs(mask_tiles_path, exist_ok=True)
  
# Get all .tif files in the rgb directory
tif_files = list(rgb_path.glob("*.tif"))

# Get all mask files in the mask directory
mask_files = list(mask_path.glob("*.tif"))

def tile_data(
    data: DatasetReader,
    out_dir: Path,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    dtype_bool: bool = False,
) -> None:
    """Tiles up orthomosaic for making predictions on."""
    out_path = out_dir
    os.makedirs(out_path, exist_ok=True)
    crs = CRS.from_string(data.crs.wkt)
    crs = crs.to_epsg()
    tilename = Path(data.name).stem

    total_tiles = int(
        ((data.bounds[2] - data.bounds[0]) / tile_width) * ((data.bounds[3] - data.bounds[1]) / tile_height)
    )

    tile_count = 0
    print(f"Tiling to {total_tiles} total tiles")

    for minx in np.arange(data.bounds[0], data.bounds[2] - tile_width, tile_width):
        for miny in np.arange(data.bounds[1], data.bounds[3] - tile_height, tile_height):

            tile_count += 1
            out_path_root = out_path / f"{tilename}_{int(round(minx))}_{int(round(miny))}_{tile_width}_{buffer}_{crs}"

            bbox = box(
                minx - buffer,
                miny - buffer,
                minx + tile_width + buffer,
                miny + tile_height + buffer,
            )

            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)
            coords = get_features(geo)

            out_img, out_transform = mask(data, shapes=coords, crop=True)
            out_meta = data.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "nodata": None,
            })

            if dtype_bool:
                out_meta.update({"dtype": "uint8"})

            out_tif = out_path_root.with_suffix(".tif")
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            clipped = rasterio.open(out_tif)
            arr = clipped.read()

            r = arr[0]
            g = arr[1]
            b = arr[2]

            rgb = np.dstack((b, g, r))  # BGR for cv2

            if np.max(g) > 255:
                rgb_rescaled = 255 * rgb / 65535
            else:
                rgb_rescaled = rgb

            cv2.imwrite(
                str(out_path_root.with_suffix(".png").resolve()),
                rgb_rescaled,
            )

            if tile_count % 50 == 0:
                print(f"Processed {tile_count} tiles of {total_tiles} tiles")

    print("Tiling complete")

if not rgb_tiles_path.exists():
    rgb_tiles_path.mkdir(parents=True)
if not mask_tiles_path.exists():
    mask_tiles_path.mkdir(parents=True)

for img_path in tif_files:
    print(f"Processing {img_path}")

    mask_file = mask_path / img_path.name
    if mask_file not in mask_files:
        print(f"Mask file not found for {img_path}")
        continue

    data = rasterio.open(img_path)
    tile_data(data, rgb_tiles_path, buffer, tile_width, tile_height, dtype_bool=False)

    with rasterio.open(mask_file) as src:
        mask_band = src.read(1)
        mask_profile = src.profile

        mask_band = np.clip(mask_band, 0, None)
        mask_band = np.interp(mask_band, (mask_band.min(), mask_band.max()), (0, 255))
        mask_band = mask_band.astype(np.uint8)
        mask_data_3band = np.stack([mask_band] * 3, axis=0)
        mask_profile.update(count=3)
        
        
        # Ensure the mask is resized to match the image size exactly
        image_shape = (data.height, data.width)
        mask_resized = cv2.resize(mask_data_3band.transpose(1, 2, 0), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_resized.transpose(2, 0, 1)  # Convert back to CHW format

        temp_mask_file = mask_tiles_path / f"{img_path.stem}.tmp"
        with rasterio.open(temp_mask_file, 'w', **mask_profile) as dst:
            dst.write(mask_resized)

    with rasterio.open(temp_mask_file) as temp_mask_data:
        tile_data(temp_mask_data, mask_tiles_path, buffer, tile_width, tile_height, dtype_bool=True)

    os.remove(temp_mask_file)

# Make train and test folders
for folder in [rgb_tiles_path, mask_tiles_path]:
    for subfolder in ["train", "test"]:
        subfolder_path = folder / subfolder
        if subfolder_path.exists() and subfolder_path.is_dir():
            shutil.rmtree(subfolder_path)
        subfolder_path.mkdir(parents=True, exist_ok=True)

file_names = list(rgb_tiles_path.glob("*.tif"))
file_roots = [item.stem for item in file_names]

num = list(range(len(file_roots)))
random.shuffle(num)

i = 0
for i in range(len(file_roots)):
    tile_root = file_roots[num[i]]
    print(f"Processed {i}/{len(file_roots)} files", end= '{\r}')
    if i < len(file_roots) * test_frac:        
        shutil.copy(rgb_tiles_path / f"{tile_root}.tif", rgb_tiles_path / "test" / f"{tile_root}.tif")
        if (mask_tiles_path / f"{tile_root}.tif").exists():
            i += 1
            continue
            #shutil.copy(mask_tiles_path / f"{tile_root}.tif", mask_tiles_path / "test" / f"{tile_root}.tif")
        else:
            print(f"Warning: Mask tile {tile_root}.tif does not exist")
    else:
        shutil.copy(rgb_tiles_path / f"{tile_root}.tif", rgb_tiles_path / "train" / f"{tile_root}.tif")
        if (mask_tiles_path / f"{tile_root}.tif").exists(): 
            i += 1       
            continue
            #shutil.copy(mask_tiles_path / f"{tile_root}.tif", mask_tiles_path / "train" / f"{tile_root}.tif")
        else:
            print(f"Warning: Mask tile {tile_root}.tif does not exist")

print("File distribution to train and test folders complete")

