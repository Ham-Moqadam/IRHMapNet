#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:16:31 2024

@author: hmoqadam
"""

import numpy as np
import functions_raddleham as fff
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from functions_raddleham import PreProcessing
import pandas as pd
import os
from functions_raddleham import PreProcessing

# Create an instance of the class
pre = PreProcessing()

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

rough_grams = ("")

smoothed_grams = ("")

a = os.listdir(rough_grams)

for i in range(len(a)):
    print(a[i])
    b = np.loadtxt(rough_grams+a[i], delimiter=',')
    print(type(b))
    ## Gaussian blur the radargram slice
    b_gaussian = ndimage.gaussian_filter(b, 1)
    b_gaussian_norm = pre.normalize_matrix(b_gaussian, 0,1)
    
    df = pd.DataFrame(b_gaussian_norm)
    df.to_csv(smoothed_grams+ a[i],
              index=False, header=False)
    print(f"gram number {i} is smoothes, normalized and saved")























