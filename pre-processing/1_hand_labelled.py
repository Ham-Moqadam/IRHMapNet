#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:59:28 2024

@author: hmoqadam

This is to get the array of the handlabelled ones

"""


import numpy as np
import pandas as pd
import imageio.v3 as iio    
import matplotlib.pyplot as plt
import cv2

hand_labelled_link = "/home/hmoqadam/h_m/My_Participation/Writing/2.SEGMENTATION/_MODEL_/DATA/for_training/1_hand_labeled_masks/"

profile_names = [19993132, 20023150]
## --- manual cut: 
    ### --- 19993132 up to row 2300
    ### --- 20023150 up to row 2000

profile_name = str(profile_names[1]) 






image = iio.imread(hand_labelled_link + profile_name+"_orig.png")
image = image[:,:,0]
hand_label_image = image.copy()
hand_label_image [hand_label_image  >0.0] = 1.0 



# desired_width = 5913
# desired_height = 3721
# image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_NEAREST)



# Set all non-zero elements to one
# hand_label_image[hand_label_image != 0] = 1

hand_label_image = hand_label_image [:2000, :] 


## if you want to erode it, its here

kernel = np.ones((2,2),np.uint8)
erosion = cv2.erode(hand_label_image,kernel,iterations = 1)


# plt.imshow(hand_label_image)
# plt.axis('off')
# plt.savefig(hand_labelled_link+ profile_name+".png", 
#             bbox_inches='tight', transparent="True", pad_inches=0)



df = pd.DataFrame(hand_label_image)
df.to_csv(hand_labelled_link+ profile_name+".csv",
          index=False, header=False)



