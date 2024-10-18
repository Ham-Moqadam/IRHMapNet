
## _________________________________________________________________________
##                                                                          |
##   author:    H.Moqadam                                                   |
##   date start:      16.12.2022                                            |   
##                                                                          |
##   Descriprion:         				                                    |
##      To plot all the full radargram profiles                             |
##                                                                          |
## _________________________________________________________________________|

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
from matplotlib.colors import LogNorm
import pandas as pd



time_start = time.time()

data_dir        = "/home/hmoqadam/h_m/RADAR_DATA/new_data/radargram_CSV/RAW/p_60nm/"

gram_image_dir   = "/home/hmoqadam/h_m/My_Participation/Writing/2.SEGMENTATION/_MODEL_/DATA/png_format/raw/"


files_shape         = []
files_rows          = []
files_columns       = []


names              = sorted(os.listdir(data_dir))

names_profile         = [s.replace("_proc.csv", "" ) for s in names]

print(names_profile)

for i in range (len(names)):
    print(f" plot number  --> {i} , name of the file --> {names_profile[i]}")
    
    ## -- loading the matrix
    matrix_diff = np.loadtxt(data_dir +names[i], 
                             skiprows = 1, delimiter = ",")
   

    plt.figure(figsize = (12,10))
    im = plt.imshow(matrix_diff, vmax=0.04*np.max(matrix_diff),
               norm=LogNorm(), cmap=plt.cm.Greys, interpolation='bilinear')
    plt.colorbar(im)
    plt.title(names_profile[i])
    plt.tight_layout()
    plt.savefig(gram_image_dir + str(names_profile[i]) + ".png", dpi = 400)
    plt.close()    

    a       = matrix_diff
    b       = a.shape
    files_shape.append(str(b))
    files_rows.append(str(b[0]))
    files_columns.append(str(b[1]))

    print(f"name of the file: {names_profile[i]}")
    # print(f"Runtime of the loop was {end_loop - start_loop}  seconds")
    print("end of loop number : ", i, "\n")
    

## ------- make a dataframe (and CSV file) with the shapes

"""

data_dir_CSV = "/home/hmoqadam/h_m/My_Participation/Writing/2.SEGMENTATION/_MODEL_/DATA/"

Data_frame_of_files = pd.DataFrame({"Name of the File     " : names_profile,
                                    "Shape of the Matrix   " : files_shape,
                                    "rows       ": files_rows,
                                    "columns    ": files_columns})

Data_frame_of_files.to_csv(data_dir_CSV + "??.ods")
"""






time_end = time.time()
print(f"the time of the routine running was: {time_end - time_start}")


























