"""
Get neighboring tiles for a given tiles in different batches and save them in a output folder.
This script is used to get the neighboring tiles for a given tile in different batches and save them in a output folder.
"""
import os
import sys
# Add the root project directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import re
from concurrent.futures import ThreadPoolExecutor
import rasterio
from TreeDetection.helpers import retrieve_neighboring_image_filenames, merge_images, crop_image, tif_geoinfo
from TreeDetection.config import get_config, Config


def crop_single_image(images_path, rgbi, meta_info, f, merged_directory=None):
    print(f"Processing {f} with {images_path} and merged_directory {merged_directory}")
    config_obj = Config()
    if merged_directory is None:
        # put it in the working directory in a subfolder
        merged_directory = os.path.join(os.getcwd(), "merged")
    local_image_names = []
    left, right, up, down = retrieve_neighboring_image_filenames(f, images_path, meta_info)

    result_directory = merged_directory
    os.makedirs(result_directory, exist_ok=True)
    f_basename = os.path.basename(f).replace(".tif", "").split("_")[0]
    f_name_end = os.path.basename(f).replace(".tif", "").split("_")[-1]
        # f_x_coord, f_y_coord = tif_geoinfo(f)
    transform = meta_info[f]
    f_x_coord, f_y_coord = transform.c, transform.f

    # We only look at the right and the bottom neighbors so that we don't process the same cropped image twice
    if right is not None:
        transform = meta_info[right]
        right_x_coord, right_y_coord = transform.c, transform.f

        try:
            with rasterio.open(f) as left_img, rasterio.open(f"{right}") as right_img:
                merged_img, merged_img_meta = merge_images(left_img, right_img)
                    

                # Write the merged file only to memory for faster processing
                with rasterio.MemoryFile() as memfile:
                    with memfile.open(**merged_img_meta) as merged_src:
                        merged_src.write(merged_img)
                        if rgbi:
                            output_filename = f"{f_basename}_{round(f_x_coord)}_{round(f_y_coord)}_{round(right_x_coord)}_{round(right_y_coord)}_{f_name_end}.tif"
                        else:
                            output_filename = f"{f_basename}_{round(f_x_coord)}{round(f_y_coord)}{round(right_x_coord)}{round(right_y_coord)}_{f_name_end}.tif"
                            # Perform cropping here
                        cropped_data, cropped_meta = crop_image(merged_src,
                                                                    (config_obj.tile_width + 2 * config_obj.buffer) *
                                                                    config_obj.overlapping_tiles_width,
                                                                    merged_src.height)
                            # Save the cropped image
                        with rasterio.open(f"{result_directory}/{output_filename}", "w", **cropped_meta) as dest:
                            dest.write(cropped_data)
                            
                        local_image_names.append(f"{result_directory}/{output_filename}")
        except Exception as e:
            config_obj.logger.error(f"Error merging images {left_img} and {right_img}: {e}")

    if down is not None:
        transform = meta_info[down]
        down_x_coord, down_y_coord = transform.c, transform.f

        try:
            with rasterio.open(f) as top_img, rasterio.open(f"{down}") as bottom_img:
                merged_img, merged_img_meta = merge_images(top_img, bottom_img)

                # Write the merged file only to memory for faster processing
                with rasterio.MemoryFile() as memfile:
                    with memfile.open(**merged_img_meta) as merged_src:
                        merged_src.write(merged_img)
                        if rgbi:
                            output_filename = f"{f_basename}_{round(f_x_coord)}_{round(f_y_coord)}_{round(down_x_coord)}_{round(down_y_coord)}_{f_name_end}.tif"
                        else:
                            output_filename = f"{f_basename}_{round(f_x_coord)}{round(f_y_coord)}{round(down_x_coord)}{round(down_y_coord)}_{f_name_end}.tif"
                            # Perform cropping here
                        cropped_data, cropped_meta = crop_image(merged_src, merged_src.width,                        
                                                                    (config_obj.tile_height + 2 * config_obj.buffer) *
                                                                    config_obj.overlapping_tiles_height)
                            # Save the cropped image
                        with rasterio.open(f"{result_directory}/{output_filename}", "w", **cropped_meta) as dest:
                            dest.write(cropped_data)

                        local_image_names.append(f"{result_directory}/{output_filename}")
        except Exception as e:
            config_obj.logger.error(f"Error merging images {top_img} and {bottom_img}: {e}")
    return local_image_names

if __name__ == "__main__":
    config, config_obj = get_config("example/config.yml")
    
    # Folders to search for correspondences
    image_dirs = ["batch1/rgb", "batch2/rgb"]
    height_dirs = ["batch1/height",  "batch1/height"]
    
    # Final output directory
    output_directory = "/locations_separated/merging"
    os.makedirs(output_directory, exist_ok=True)
    
    image_regex_pattern = re.compile(config["image_regex"])
    height_data_regex_pattern = re.compile(config["height_data_regex"])
    
    all_images = dict() 
    all_height_data = dict()
    for dir in image_dirs:
        all_images[dir] = [os.path.join(dir, f) for f in os.listdir(dir) if image_regex_pattern.search(os.path.basename(f))]
    for dir in height_dirs:
        all_height_data[dir] = [os.path.join(dir, f) for f in os.listdir(dir) if height_data_regex_pattern.search(os.path.basename(f))]
    
    print(f"Found {len(all_images)} images and {len(all_height_data)} height data files.")
    cropped_image_names = []
    meta_info = {}
    for folders in [all_images, all_height_data]:
        for batch in folders:
            for image in folders[batch]:
                transform, _, _, _ = tif_geoinfo(image)
                meta_info[image] = transform
            
    print(meta_info)
    # Get the total length of all lists in all_images
    i = 0
    total_length = len(all_images.keys()) + len(all_height_data.keys())
    for mode in [(all_images, True), (all_height_data, False)]:
        files, rgbi = mode        
        for batch in files.keys():
            # make images_path a list of all images that are not in the batch
            images_path = []
            for f in files.keys():
                if f != batch:
                    images_path.extend(files[f])
            number_of_files_found = len(files[batch])
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(crop_single_image, [images_path] * number_of_files_found, [rgbi] * number_of_files_found, \
                    [meta_info] * number_of_files_found, files[batch], [output_directory] * number_of_files_found))
            cropped_image_names.extend(results)
            i += 1
            #config_obj.logger.info(f"Processed {i}/{total_length} batches.")
                