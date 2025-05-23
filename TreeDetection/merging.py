
import concurrent
import os

import rasterio
from TreeDetection.config import Config
from TreeDetection.helpers import crop_image, merge_images, retrieve_neighboring_image_filenames, tif_geoinfo


def merge_and_crop_images(config, images_paths, height_paths):
    config_obj = Config()
    config_obj._load_into_config(config)
    logger = config["logger"]
    # Filter out images that have already been processed (can be identified by the __ in the filename)

    merged_directory = config["merged_path"]
    
    def save_cropped_images(images_path, rgbi=True):
        """
        Save the cropped images based on the neighboring images.

        Args:
            images_path (list): List of image paths.
        """
        cropped_image_names = []
        meta_info = {}
        for f in images_path:
            transform, _, _, _ = tif_geoinfo(f)
            meta_info[f] = transform
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(crop_single_image, [images_path] * len(images_path), [rgbi] * len(images_path), [meta_info] * len(images_path), images_path))
        cropped_image_names.extend([item for sublist in results for item in sublist])

        return cropped_image_names

    def crop_single_image(images_path, rgbi, meta_info, f):
        local_image_names = []
        left, right, up, down = retrieve_neighboring_image_filenames(f, images_path, meta_info)

        directory = os.path.dirname(f)
        result_directory = f"{directory}/{merged_directory}"
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
                                                                        (config["tile_width"] + 2 * config["buffer"]) *
                                                                        config["overlapping_tiles_width"],
                                                                        merged_src.height)
                                # Save the cropped image
                            with rasterio.open(f"{result_directory}/{output_filename}", "w", **cropped_meta) as dest:
                                dest.write(cropped_data)
                                
                            local_image_names.append(f"{result_directory}/{output_filename}")
            except Exception as e:
                logger.error(f"Error merging images {left_img} and {right_img}: {e}")

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
                                                                        (config["tile_height"] + 2 * config["buffer"]) *
                                                                        config["overlapping_tiles_height"])
                                # Save the cropped image
                            with rasterio.open(f"{result_directory}/{output_filename}", "w", **cropped_meta) as dest:
                                dest.write(cropped_data)

                            local_image_names.append(f"{result_directory}/{output_filename}")
            except Exception as e:
                logger.error(f"Error merging images {top_img} and {bottom_img}: {e}")
        return local_image_names

    # Here we merge and crop neighboring images
    try:
        cropped_image_filenames = save_cropped_images(images_paths, rgbi=True)
        cropped_height_filenames = save_cropped_images(height_paths, rgbi=False)

        # Include the image paths of the cropped images to the list of images to be processed
        images_paths.extend(cropped_image_filenames)
        height_paths.extend(cropped_height_filenames)
    except Exception as e:
        logger.error(f"Error merging and cropping images: {e}")