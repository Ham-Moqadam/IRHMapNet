

import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    # Define image and mask directories
    img_dir = 'slices/grams_patches/'
    mask_dir = 'slices/masks_patches/'
    
    # Get list of image and mask names
    radargram_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    # Create empty lists for images and masks
    X = []
    Y = []
    
    for file in radargram_files:
        # Load image data
        img = np.array(np.loadtxt(f'{img_dir}{file}', delimiter=","))
        
        # Ensure image is of shape (512, 512)
        if img.shape != (512, 512):
            print(f'Warning: Radargram file {file} does not have shape (512, 512)')
            # Resize or pad if needed
            img = np.pad(img, ((0, 512 - img.shape[0]), (0, 512 - img.shape[1])), mode='constant', constant_values=0)
        
        # Add a channel dimension
        img = img[..., np.newaxis]
        X.append(img)
    
    for file in mask_files:
        # Load mask data
        img = np.array(np.loadtxt(f'{mask_dir}{file}', delimiter=","))
        
        # Ensure mask is of shape (512, 512)
        if img.shape != (512, 512):
            print(f'Warning: Mask file {file} does not have shape (512, 512)')
            # Resize or pad if needed
            img = np.pad(img, ((0, 512 - img.shape[0]), (0, 512 - img.shape[1])), mode='constant', constant_values=0)
        
        # Add a channel dimension
        img = img[..., np.newaxis]
        Y.append(img)
    
    # Convert lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # Perform train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    return X_train, Y_train, X_test, Y_test






"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data():
    # Load data from CSV files
    radargram_files = sorted(os.listdir('slices/grams_patches'))
    mask_files = sorted(os.listdir('slices/masks_patches'))
    
    X = []
    y = []

    for file in radargram_files:
        # img = np.array(pd.read_csv(f'slices/grams_patches/{file}'))
        img = np.array(np.loadtxt(f'slices/grams_patches/{file}', delimiter=","))
        print(f'Loaded radargram file {file} with shape {img.shape}')
        if img.shape != (512, 512):
            # Handle unexpected shape
            print(f'Warning: Radargram file {file} does not have shape (512, 512)')
            img = np.resize(img, (512, 512))  # Resize or pad if needed
        X.append(img)
    
    for file in mask_files:
        #img = np.array(pd.read_csv(f'slices/masks_patches/{file}'))
        img = np.array(np.loadtxt(f'slices/masks_patches/{file}', delimiter = ","))
        print(f'Loaded mask file {file} with shape {img.shape}')
        if img.shape != (512, 512):
            # Handle unexpected shape
            print(f'Warning: Mask file {file} does not have shape (512, 512)')
            img = np.resize(img, (512, 512))  # Resize or pad if needed
        y.append(img)
    
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)[..., np.newaxis]

    # Print shapes for verification
    print(f'X shape: {X.shape}, y shape: {y.shape}')

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_val, y_val


"""



"""


import numpy as np
import pandas as pd


import os
from sklearn.model_selection import train_test_split




def load_data():  
    ## ------- Define image and mask directories
    img_dir = 'slices/grams_patches/'
    mask_dir = 'slices/masks_patches/'
    
    ## ------- Get list of image and mask names
    radargram_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    ## ------ Create empty lists for images and masks
    X = []
    Y = []
    
    for file in radargram_files:
        # Load image data
        img = np.array(np.loadtxt(f'{img_dir}{file}', delimiter=","))
        # print(f'Loaded radargram file {file} with shape {img.shape}')
        
        # Ensure image is of shape (512, 512)
        if img.shape != (512, 512):
            print(f'Warning: Radargram file {file} does not have shape (512, 512)')
            # Resize or pad if needed
            img = np.pad(img, ((0, 512 - img.shape[0]), (0, 512 - img.shape[1])), mode='constant', constant_values=0)
        
        # Add a channel dimension
        img = img[..., np.newaxis]
        X.append(img)
    
    for file in mask_files:
        # Load mask data
        img = np.array(np.loadtxt(f'{mask_dir}{file}', delimiter=","))
        # print(f'Loaded mask file {file} with shape {img.shape}')
        
        # Ensure mask is of shape (512, 512)
        if img.shape != (512, 512):
            print(f'Warning: Mask file {file} does not have shape (512, 512)')
            # Resize or pad if needed
            img = np.pad(img, ((0, 512 - img.shape[0]), (0, 512 - img.shape[1])), mode='constant', constant_values=0)
        
        # Add a channel dimension
        img = img[..., np.newaxis]
        Y.append(img)
    
    ## ----- Convert lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    ## ----- Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, Y_train, X_test, Y_test


"""










