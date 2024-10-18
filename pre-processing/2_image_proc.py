

### ----------------------------------------------------------------------- ###
### to make the traing data for the grand finale 
### date: 3.7.2024
### author: H.Moqadam
### taken form the old functions / module
### ----------------------------------------------------------------------- ###


## 

# !. normalize

# !. possibly sharpen the iamge

        # import cv2 as cv
        # sharpening_k1 = np.array([[-1, -1, -1],
        #                          [-1, 12, -1],
        #                          [-1, -1, -1]])
        # my_matrix = cv.filter2D(my_matrix, -1, sharpening_k1)

# !. cut off what is lower than the threshold b = (a >= cutoff).astype(np.float32)


### another thing: laplacian using kernel

        # kernel1          = (1/6) * np.array([[1,4,1],
        #                                         [4,-20,4],
        #                                         [1,4,1]])
        # laplacian_a = ndimage.convolve(a, kernel1, mode="constant")
        
        
        # for i in range(1,laplacian_Rdata.shape[0]-1):
        #     for j in range(1,laplacian_Rdata.shape[1]-1):
        #         if laplacian_Rdata[i,j] ==0 and laplacian_Rdata[i,j-1] != 0 and laplacian_Rdata[i,j+1] != 0:
        #             laplacian_Rdata[i,j] = 500
        
        # low_values_flags = laplacian_Rdata < 400  # Where values are low
        # laplacian_Rdata[low_values_flags] = 0

### 

# some denoising methods:

    
## !. erosion and then dilution using ndimage

        # from scipy import ndimage
        
        # cl_layers1 = ndimage.binary_dilation(layers,).astype(int)
        # cl_layers2 = ndimage.binary_erosion(cl_layers1,border_value=2).astype(int)
        
        # cl_layers2 = np.ma.masked_where(cl_layers2 <1, cl_layers2)
        
## !. erosion and then dilution using open CV

        #import cv2
        
        # kernel = np.ones((1,2),np.uint8)
        # erosion = cv2.erode(layers,kernel,iterations = 1)
        # plot_echogram       = ip.plot_radargram("open CV morph - erosion", (9,6),
        #                                         erosion, "binary")
        # dilate = cv2.dilate(erosion,kernel,iterations = 1)
        # plot_echogram       = ip.plot_radargram("open CV morph - opening", (9,6),
        #                                dilate, "binary")

## !. CV's own denooising for images
        #  cv2.fastNlMeansDenoisingMulti

## !. my own one noo. 1

        # the_matrix = layers
        
        # new_one = 1 *the_matrix
        
        # for k in range(1,3):
        #     print(k)
        #     for i in range (k,the_matrix.shape[0]-k):
        #         for j in range (k,the_matrix.shape[1]-k):
        #             if (new_one[i-k,j]==0 and new_one[i+k,j]==0 and
        #                 new_one[i,j-k]==0 and new_one[i,j+k]==0 and
        #                 new_one[i+k,j+k]==0 and new_one[i-k,j-k]==0 and
        #                 new_one[i+k,j-k]==0 and new_one[i-k,j+k]==0):
        #                 new_one[i,j] = 0


# !/ save the image:

        # import imageio.v3 as iio    
        # iio.imwrite('example_image.png', original_image)


## !!! save with .png because it does not introduce compression artifacts.





import numpy as np
import functions_raddleham as fff
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from functions_raddleham import PreProcessing
import pandas as pd

# Create an instance of the class
pre_processor = PreProcessing()

data_csv_path = "/home/hmoqadam/h_m/RADAR_DATA/new_data/radargram_CSV/DIFF/p_60nm/"
profile_names = [19993109, 19993110]

profile_name = str(profile_names[1])
image_matrix = np.loadtxt(data_csv_path+str(profile_name)+"_proc.csv", skiprows=1, delimiter=",")

## normalize
image = pre_processor.normalize_matrix(image_matrix, 0,1)
    
## slice the radargram for testing
image_test = image#[200:700,1000:1500]
#[600:800,600:800]

## Gaussian blur the radargram slice
image_test_gaussian = ndimage.gaussian_filter(image_test, 1)




"""

### !!! this is not good, at lears for the laplacian, the gaussian blur is necessary

## same without Gaussian blurr

## convoluve it with laplacian Operator
image_test_lap = ndimage.convolve(image_test, lap_oper)
## normalize it
image_test_lap = fff.pre_processing.normalize_matrix(image_test_lap, 0, 1)
## threshold it`
threshold = 0.7
image_test_lap_thre = np.where(image_test_lap  > threshold , 1, 0)


## plot it all together
plt.figure("sliced >> laplacian >> threshold ")
plt.subplot(2,2,1)
plt.imshow(image_test)
plt.subplot(2,2,2)
plt.imshow(image_test_lap)
plt.subplot(2,2,3)
plt.imshow(image_test_lap_thre)
"""



#%%
### ---- 1. using the laplacian method

# !!! sharpening did not help, but also was disadvantageous




## convoluve it with laplacian Operator
lap_oper = (1/6) * np.array([[1,4,1],
                    [4,-20,4],
                    [1,4,1]])
image_test_gau_lap = ndimage.convolve(image_test_gaussian, lap_oper)
## normalize it
image_test_gau_lap = pre_processor.normalize_matrix(image_test_gau_lap, 0, 1)
## threshold it
threshold = 0.70
image_test_gau_lap_thre = np.where(image_test_gau_lap  > threshold , 1, 0)


## morphological closing 
image_morph_1 = ndimage.binary_dilation(image_test_gau_lap_thre).astype(np.int)
image_morph_2 = ndimage.binary_erosion(image_morph_1 ).astype(np.int)
image_test_gau_lap_thre_closed = image_morph_2



## skeletonize and regio labelling


"""
NOTE TO MYSELF:
    HERE I CAN DO 2 THINGS TO USE THE METHOD FOR REGION LABELLING
    
    1. 
    import functions_raddleham as fff
    processor = fff.PostProcessing()    # Create an instance of the class
    
    A = processor.label_connected_comps(image, size_threshold=10, connectivity=8)


    2. 
    from functions_raddleham import PostProcessing
    processor = PostProcessing()    # Create an instance of the class
    
    A = processor.label_connected_comps(image, size_threshold=10, connectivity=8)
"""

from skimage.morphology import skeletonize
from functions_raddleham import PostProcessing

# Create an instance of the class
processor = PostProcessing()

image_test_gau_lap_thre_closed_skeleton = skeletonize(image_test_gau_lap_thre_closed)
image_test_gau_lap_thre_closed_skeleton_regioned = processor.label_connected_comps(image_test_gau_lap_thre_closed_skeleton,20,8)

image_test_gau_lap_thre_closed_regioned = processor.label_connected_comps(image_test_gau_lap_thre_closed, 40, 8)
image_test_gau_lap_thre_closed_regioned_skeleton= skeletonize(image_test_gau_lap_thre_closed_regioned).astype(np.int16)


plt.figure("laplacian steps",figsize=(14,14))
plt.suptitle("Using laplacian \n sliced >> gaussian >> laplacian >> threshold >> closed >> skeleton >> regioned ")
plt.subplot(3,3,1)
plt.title("Sliced")
plt.imshow(image_test)
plt.subplot(3,3,2)
plt.title("Blurred")
plt.imshow(image_test_gaussian)
plt.subplot(3,3,3)
plt.title("Laplacian")
plt.imshow(image_test_gau_lap)
plt.subplot(3,3,4)
plt.title("Threshold = "+ str(threshold))
plt.imshow(image_test_gau_lap_thre)
plt.subplot(3,3,5)
plt.title("Closed")
plt.imshow(image_test_gau_lap_thre_closed)
plt.subplot(3,3,6)
plt.title("Skeleton")
plt.imshow(image_test_gau_lap_thre_closed_skeleton)
plt.subplot(3,3,7)
plt.title("Skeleton + region labelling")
plt.imshow(image_test_gau_lap_thre_closed_skeleton_regioned) # !! this is the better one
plt.subplot(3,3,8)
plt.title("region labelling + Skeleton")
plt.imshow(image_test_gau_lap_thre_closed_regioned_skeleton)
# plt.tight_layout(False)
# plt.savefig("2_using_laplacian.png")


## the name 
laplacian_edge = image_test_gau_lap_thre_closed_skeleton_regioned



#%% 
### ------ 2. using thegradient in X and Y




# Step 3: Compute gradients in x and y directions
# Sobel kernels for gradient computation
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])


# Convolve smoothed image with Sobel kernels
gradient_x = ndimage.convolve(image_test_gaussian, sobel_x)
gradient_y = ndimage.convolve(image_test_gaussian, sobel_y)


# Step 4: Compute gradient magnitude and direction
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)



# Step 5: Edge detection
# Apply thresholding to detect strong edges
threshold = 0.70  # Adjust this threshold as needed
image_test_gaussian_edged = gradient_magnitude > threshold



# erosion
image_test_gaussian_edged_erode = ndimage.binary_erosion(image_test_gaussian_edged).astype(np.int)

"""

# 1. regioned >> skeletonized

image_test_gaussian_edged_erode_regioned = processor.label_connected_comps(image_test_gaussian_edged_erode,20,8)

image_test_gaussian_edged_erode_regioned_skeleton = skeletonize(image_test_gaussian_edged_erode_regioned)

plt.figure(1), plt.imshow(image_test_gaussian_edged_erode_regioned_skeleton)
"""



# 2. skeletonized >> regioned
image_test_gaussian_edged_skeleton = skeletonize(image_test_gaussian_edged_erode)

image_test_gaussian_edged_skeleton_regioned = processor.label_connected_comps(image_test_gaussian_edged_skeleton, 40,8)

# plt.figure(2), plt.imshow(image_test_gaussian_edged_skeleton_regioned )





plt.figure("grad_XY steps",figsize=(14,14))
plt.suptitle("Using X & Y Gradient:  \n sliced >> gaussian >> threshold >> eroded >> skeleton >> region")
plt.subplot(3,2,1)
plt.title("Sliced")
plt.imshow(image_test)
plt.subplot(3,2,2)
plt.title("Blurred")
plt.imshow(image_test_gaussian)
plt.subplot(3,2,3)
plt.title("Threshold = "+ str(threshold))
plt.imshow(image_test_gaussian_edged)
plt.subplot(3,2,4)
plt.title("Eroded")
plt.imshow(image_test_gaussian_edged_erode)
plt.subplot(3,2,5)
plt.title("Skeleton")
plt.imshow(image_test_gaussian_edged_skeleton)
plt.subplot(3,2,6)
plt.title("region labelling")
plt.imshow(image_test_gaussian_edged_skeleton_regioned)
# plt.tight_layout(False)
# plt.savefig("1_using_gradient.png")


## the name 
grad_XY_edge = image_test_gaussian_edged_skeleton_regioned



#%% 
### ------ 3. my own old method

# !!! try both starting with Gaussian or not Gaussian

from functions_raddleham import ImageProcessing


# Create an instance of the class
image_pr = ImageProcessing()

image_test_gaussian = ndimage.gaussian_filter(image_test, 1)


## sharpening just in cae
import cv2 as cv
sharpening_k1 = np.array([[-1, -1, -1],
                          [-1, 12, -1],
                          [-1, -1, -1]])
image_test_gaussian_sharp = cv.filter2D(image_test_gaussian, -1, sharpening_k1)


image_test_sharp = cv.filter2D(image_test, -1, sharpening_k1)


"""
plt.subplot(1,4,1)
plt.title("section")
plt.imshow(image_test)
plt.subplot(1,4,2)
plt.title("gaussian")
plt.imshow(image_test_gaussian)
plt.subplot(1,4,3)
plt.title("gaussian sharp")
plt.imshow(image_test_gaussian_sharp)
plt.subplot(1,4,4)
plt.title("sharp")
plt.imshow(image_test_sharp)
"""


#3 here choose what to use (image, blurred, sharpened ...)
image = image_test_gaussian


## gradient in Y
image_grad_y = image_pr.gradient_in_y(image)

## smoothing
image_grad_y = ndimage.convolve(image_grad_y, (1/9)*np.ones((3,3)), mode="constant")

## normalizing
image_grad_y_norm = pre_processor.normalize_matrix(image_grad_y, 0,1)




# ## gradient in Y
# image_grad_x = image_pr.gradient_in_x(image)
# ## smoothing
# image_grad_x = ndimage.convolve(image_grad_x, (1/9)*np.ones((3,3)), mode="constant")
# ## normalizing
# image_grad_x_norm = pre_processor.normalize_matrix(image_grad_x, 0,1)

# ## sum of the gradients
# image_grad_y_norm  = image_grad_x_norm + image_grad_y_norm


## thresholding
threshold = 0.725
image_grad_y_thresh = np.where(image_grad_y_norm  > threshold , 1, 0)


## morphological closing 
image_morph_1 = ndimage.binary_dilation(image_grad_y_thresh).astype(np.int)
image_morph_2 = ndimage.binary_erosion(image_morph_1 ).astype(np.int)
image_grad_y_thresh_closed = image_morph_2


## skeletonzie 
image_grad_y_thresh_closed_skeleton = skeletonize(image_grad_y_thresh_closed)


## region labelling
image_grad_y_thresh_closed_skeleton_region = processor.label_connected_comps(image_grad_y_thresh_closed_skeleton , 10, 8)


plt.imshow(image_grad_y_thresh_closed_skeleton_region)

## the name 
old_method_edge = image_grad_y_thresh_closed_skeleton_region




#%% PLOTTING THE THREE METHODS TOGETHER


plt.figure("final comparison")
plt.subplot(2,2,1)
plt.title("laplacian")
plt.imshow(laplacian_edge,cmap = "binary_r")
plt.subplot(2,2,2)
plt.title("grad XY")
plt.imshow(grad_XY_edge,cmap = "binary_r")
plt.subplot(2,2,3)
plt.title("old method")
plt.imshow(old_method_edge,cmap = "binary_r")



plt.figure("comparison with the gram")
plt.subplot(1,2,1)
plt.title("gram")
plt.imshow(image_test,cmap = "binary_r")
plt.subplot(1,2,2)
plt.title("grad XY")
plt.imshow(grad_XY_edge,cmap = "binary_r")





"""


grad_XY_edge = grad_XY_edge[:2000,:]


## --- dilation to 3
import cv2
kernel = np.ones((3,3))
grad_XY_edge = cv2.dilate(grad_XY_edge,kernel,iterations = 1)



plt.figure(333)
plt.imshow(grad_XY_edge)
plt.axis('off')
plt.savefig("/home/hmoqadam/h_m/My_Participation/Writing/2.SEGMENTATION/_MODEL_/DATA/for_training/2_image_processing_answers/"+
            profile_name+".png", dpi = 400, 
            bbox_inches='tight', transparent="True", pad_inches=0)


df = pd.DataFrame(grad_XY_edge)
df.to_csv("/home/hmoqadam/h_m/My_Participation/Writing/2.SEGMENTATION/_MODEL_/DATA/for_training/2_image_processing_answers/"+ profile_name+".csv",
          index=False, header=False)



"""







































