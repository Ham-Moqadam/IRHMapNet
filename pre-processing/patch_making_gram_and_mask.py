


#### 
####
#### PATCH GENERATION FOR BOTH RAADRGRAMS AND MASKS
####
####



import random
import os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

time1 = time.time()

def generate_random_slices(data_dir_masks, data_dir_radargrams, profile_name, output_dir_masks, output_dir_radargrams, num_patches, patch_size, display_time=2):
    """
    Generate random slices from mask and radargram CSV files, save them in PNG and CSV formats, and display them for a limited time.
    Apply random transformations (none, mirror, flip, both) to each patch.

    :param data_dir_masks: Directory containing the mask images or CSV files.
    :param data_dir_radargrams: Directory containing the radargram CSV files.
    :param profile_name: Base name of the image or CSV file (without extension).
    :param output_dir_masks: Directory to save the mask patches.
    :param output_dir_radargrams: Directory to save the radargram patches.
    :param num_patches: Number of random patches to generate.
    :param patch_size: Size of each patch (width, height).
    :param display_time: Time in seconds to display each patch.
    """
    ## Create the output directories if they don't exist
    os.makedirs(output_dir_masks, exist_ok=True)
    os.makedirs(output_dir_radargrams, exist_ok=True)

    ## Read the mask PNG file if it exists, otherwise read the mask CSV file
    image_path_mask_png = os.path.join(data_dir_masks, f"{profile_name}.png")
    image_path_mask_csv = os.path.join(data_dir_masks, f"{profile_name}.csv")

    try:
        if os.path.exists(image_path_mask_csv):
            image_matrix_mask = np.loadtxt(image_path_mask_csv, delimiter=",")
            image_matrix_mask = (image_matrix_mask != 0).astype(np.uint8)  # Convert to binary
        elif os.path.exists(image_path_mask_png):
            image_matrix_mask = imageio.imread(image_path_mask_png)
            if len(image_matrix_mask.shape) == 3 and image_matrix_mask.shape[2] == 4:
                image_matrix_mask = image_matrix_mask[:, :, 2]  # Convert RGBA to grayscale
            image_matrix_mask = (image_matrix_mask > 127).astype(np.uint8)  # Convert to binary
        else:
            raise FileNotFoundError(f"Neither {image_path_mask_png} nor {image_path_mask_csv} exists")

        img_height, img_width = image_matrix_mask.shape

        ## Check if patch size is valid
        if patch_size[0] > img_width or patch_size[1] > img_height:
            print(f"An error occurred: Patch size {patch_size} is larger than image dimensions {image_matrix_mask.shape}")
            return

        ## Read the radargram CSV file
        image_path_radargram_csv = os.path.join(data_dir_radargrams, f"{profile_name}_proc.csv")
        if not os.path.exists(image_path_radargram_csv):
            raise FileNotFoundError(f"Radargram file {image_path_radargram_csv} does not exist")
        
        image_matrix_radargram = np.loadtxt(image_path_radargram_csv, skiprows=1, delimiter=",")
        ## image_matrix_radargram = (image_matrix_radargram != 0).astype(np.uint8)  # Convert to binary

        ## Generate random patches
        for i in tqdm(range(num_patches), desc=f"Generating patches for {profile_name}"):
            try:
                left = random.randint(0, img_width - patch_size[0])
                top = random.randint(0, img_height - patch_size[1])
                right = left + patch_size[0]
                bottom = top + patch_size[1]

                patch_mask = image_matrix_mask[top:bottom, left:right]
                patch_radargram = image_matrix_radargram[top:bottom, left:right]
                
                ## print(top,bottom, left,right)

                ## Randomly select a transformation: none, mirror, flip, both
                transform_choice = random.choice(['none', 'mirror', 'flip', 'both'])

                if transform_choice == 'mirror':
                    patch_mask = np.fliplr(patch_mask)
                    patch_radargram = np.fliplr(patch_radargram)
                elif transform_choice == 'flip':
                    patch_mask = np.flipud(patch_mask)
                    patch_radargram = np.flipud(patch_radargram)
                elif transform_choice == 'both':
                    patch_mask = np.fliplr(np.flipud(patch_mask))
                    patch_radargram = np.fliplr(np.flipud(patch_radargram))

                ## Save the mask patch as PNG
                patch_mask_png_path = os.path.join(output_dir_masks, f"{profile_name}_patch_{i}.png")

                ## Save the mask patch as CSV
                patch_mask_csv_path = os.path.join(output_dir_masks, f"{profile_name}_patch_{i}.csv")
                pd.DataFrame(patch_mask).to_csv(patch_mask_csv_path, index=False, header=False)

                ## Save the radargram patch as PNG
                patch_radargram_png_path = os.path.join(output_dir_radargrams, f"{profile_name}_patch_{i}.png")

                ## Save the radargram patch as CSV
                patch_radargram_csv_path = os.path.join(output_dir_radargrams, f"{profile_name}_patch_{i}.csv")
                pd.DataFrame(patch_radargram).to_csv(patch_radargram_csv_path, index=False, header=False)

                ## Display the mask patch
                plt.figure("random patch", figsize=(9, 9))
                plt.imshow(patch_mask, cmap='gray')
                plt.axis('off')
                plt.show(block=False)
                plt.pause(display_time)
                plt.savefig(patch_mask_png_path, bbox_inches='tight', transparent="True", pad_inches=0)
                plt.close()

                ## Display the radargram patch
                plt.figure("random patch", figsize=(9, 9))
                plt.imshow(patch_radargram, cmap='binary')
                plt.axis('off')
                plt.show(block=False)
                plt.pause(display_time)
                plt.savefig(patch_radargram_png_path, bbox_inches='tight', transparent="True", pad_inches=0)
                plt.close()

                print(f"Saved mask patch: {patch_mask_png_path} and {patch_mask_csv_path}")
                print(f"Saved radargram patch: {patch_radargram_png_path} and {patch_radargram_csv_path}")

            except ValueError as ve:
                print(f"ValueError for patch {i}: {ve}")

    except Exception as e:
        print(f"An error occurred: {e}")



#%%
        
### ---------------------------------------------------------------------------        
        
        
        

# Example usage
data_dir_masks = "../../DATA/for_training/masks_full/"
data_dir_radargrams = "../../DATA/for_training/grams_full/"

output_dir_masks = "../../DATA/for_training/set2_masks_patches/"
output_dir_radargrams = "../../DATA/for_training/set2_grams_patches/"


### for the image processing and radon method
profile_names = [19983102, 20013122]
num_patches = 120
patch_size = (512, 512)
display_time = 0.2

for profile_name in profile_names:
    generate_random_slices(data_dir_masks, data_dir_radargrams, str(profile_name), output_dir_masks, output_dir_radargrams, num_patches, patch_size, display_time)


### for the hand-labelled method
profile_names = [19993132 , 20023150]
num_patches = 180
patch_size = (512, 512)
display_time = 0.2

for profile_name in profile_names:
    generate_random_slices(data_dir_masks, data_dir_radargrams, str(profile_name), output_dir_masks, output_dir_radargrams, num_patches, patch_size, display_time)




time2 = time.time()

print("tital_time"  ,time2-time1)








