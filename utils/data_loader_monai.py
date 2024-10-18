import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    img_dir = 'slices/set2_grams_patches/'
    mask_dir = 'slices/set2_masks_patches/'

    radargram_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    X = []
    Y = []

    for file in radargram_files:
        img = np.array(np.loadtxt(f'{img_dir}{file}', delimiter=","))
        if img.shape != (512, 512):
            print(f'Warning: Radargram file {file} does not have shape (512, 512)')
            img = np.pad(img, ((0, 512 - img.shape[0]), (0, 512 - img.shape[1])), mode='constant', constant_values=0)
        img = img[np.newaxis, ...]  # Ensuring the shape is (1, 512, 512)
        print(f'Loaded image shape: {img.shape}')  # Debugging statement
        X.append(img)

    for file in mask_files:
        img = np.array(np.loadtxt(f'{mask_dir}{file}', delimiter=","))
        if img.shape != (512, 512):
            print(f'Warning: Mask file {file} does not have shape (512, 512)')
            img = np.pad(img, ((0, 512 - img.shape[0]), (0, 512 - img.shape[1])), mode='constant', constant_values=0)
        img = img[np.newaxis, ...]  # Ensuring the shape is (1, 512, 512)
        print(f'Loaded mask shape: {img.shape}')  # Debugging statement
        Y.append(img)


    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    return X_train, Y_train, X_test, Y_test

