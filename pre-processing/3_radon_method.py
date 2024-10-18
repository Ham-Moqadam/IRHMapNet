





import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
import imageio.v2 as imageio
from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline
import pandas as pd

# #read image
# image = imageio.imread(r"/home/hmoqadam/h_m/PROJECTS/red_lines_from_ben/Stratigraph2.tif")
# if (len(image.shape) == 3): #if color image, convert to grayscale
#     image = image[:,:,1]
    

# 19983102
# 20013122
radar_profile_name = "19993132"

radargram = np.loadtxt("/home/hmoqadam/h_m/My_Participation/Writing/2.SEGMENTATION/_MODEL_/raddleham_pixelwise/results/result_first_training/"
                       +radar_profile_name+"_proc.csv", delimiter=",", skiprows=1)

image = radargram[:2000,:]

image = image - np.mean(image) #zero mean so that the sinogram of dark and bright lines is balanced
image = image / np.max(image)  #normalization
T_matrix = np.full((image.shape[0], image.shape[1]), np.nan)  # theta ((angle) matrix, filled with NaN)

#plt.close('all')
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 4)) #figure for visualizing the sinogram

tilex = 200#tile size in x direction for sub-images
tiley = 200 #tile size in y direction for sub-images
theta = np.linspace(0., 180., max([tilex,tiley]), endpoint=False) #compute the tehtas for the sinogram

## default: tilex = 20, tiley = 20, theta = 180

#compute the sinograms for sub-images
for c in range(0,image.shape[1]-tilex,tilex//2): #column index c, tilex//2 means 50% overlap between tiles
    print(f" progress {(c/image.shape[1]*100):.2f} percent done") 
    T = [0] # theta array
    Y = [0] # array for the y-coordinate
    W = [1] # array for the weights (for the spline interpolation)
    
    for r in range(0,image.shape[0]-tiley,tiley//2): #row index r
                       
        im = image[r:r+tiley,c:c+tilex] #cur out a sub-image 
        #im = im/np.mean(im,axis=0)
        #im = im - np.mean(im)
        #im = im / np.max(im)

        sinogram = (radon(im, theta=theta, circle=False))**2 #compute the sinogram using the radon transform
        max_index = np.unravel_index(np.argmax(sinogram), sinogram.shape) #detect the brightes pixel, extract its coordinates
        pr = sinogram[max_index[0],:] # array representing the sinogram of the brightes line
        #pr = np.sum(sinogram, axis=0)
        
        #center of mass (of theta of the brightest line) 
        pr_com = pr[max(0, np.argmax(pr) - 2):min(len(pr), np.argmax(pr) + 3)]
        pr_com = pr_com - np.min(pr_com)
        theta_com = theta[max(0, np.argmax(pr) - 2):min(len(pr), np.argmax(pr) + 3)]
        theta_com = np.sum(pr_com*theta_com)/np.sum(pr_com)
        
        weight = np.max(pr)-np.mean(pr)  #weight corresponds to the brightness of the line in the sinogram
        T.append(theta_com - 90)    # append theta
        W.append(weight)            #  weights
        Y.append(r+tiley//2)        # append y-coordinate
        
        
        
        #print(f"r={r}  c={c} theta={(theta_com-90):.2f} ")
    #now all rows in the selected column "c" have been analyzed
    W[0] = np.max(W)        #   normalizing the weights, maximum weight in that columen is normalized to unity
    W = W - np.median(W)    #   tiles with a weight below the median weight will be ignored
    W[W<0] = 0              #   tiles with a weight below the median weight will be ignored
    W = W/np.max(W)         #   normalize to unity

    # spline interpolation along the row entries of one column
    spline = UnivariateSpline(Y, T, w=W, k=2)  # spline interpolation, k=1 is linear, k=2 is quadratic, k=3 is cubic
    interpolation_range = np.arange(0, image.shape[0])
    T_valid = spline(interpolation_range)
    T_matrix[:,c+tilex//2] = T_valid

# finished with all the tiles (sub-images), now we need to fill the values between the selected colums using a linear interpolation
valid_positions = np.argwhere(~np.isnan(T_matrix))
y_known, x_known = valid_positions.T
values = T_matrix[valid_positions[:, 0], valid_positions[:, 1]]
x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
T_matrix = griddata((x_known, y_known), values, (x, y), method='linear')

#compute the gradient
gradient = np.tan(T_matrix*np.pi/180.0) #convert degree to rad
nan_positions = np.argwhere(np.isnan(gradient))
gradient[nan_positions[:,0], nan_positions[:,1]] = 0
fig2 = plt.figure() 
plt.imshow(gradient, vmin = -2, vmax = 2)

# red lines
all_horizons = [] ## -- me
fig3 = plt.figure()
plt.imshow(image, cmap='gray')
for y in range(10,image.shape[0],50):
    hor = np.zeros(image.shape[1])
    hor[0] = y
    for x in range(1,image.shape[1]):
        hor[x] = hor[x-1] - gradient[int(hor[x-1]),x]
        if hor[x] > image.shape[0]:
            break
    plt.plot(hor)
    all_horizons.append(hor) ## -- me
plt.show()





 
# plt.savefig("/home/hmoqadam/h_m/My_Participation/Conference_Symposium/EGU_2024/layer_slope.png",
#             dpi = 400,
#            bbox_inches='tight', transparent="True", pad_inches=0)



mask = np.array(all_horizons)
mask = np.round(mask)

# ### plotting the horizons
# for i in range(len(mask)):
#      plt.plot(mask[i,:])



# plt.imshow(image, cmap='gray')
# for i in range(len(mask)):
#     plt.plot(mask[i,:])



mask_empty = np.copy(radargram) * 0



a = np.copy(mask)

for row in range(a.shape[0]):
    for col in range(a.shape[1]):
        if 0 <= int(a[row, col]) < mask_empty.shape[0]:
            mask_empty[int(a[row, col]), col] = 1







mask_empty = mask_empty[:2000,:]

## --- dilation to 3
import cv2
kernel = np.ones((3,3))
mask_empty = cv2.dilate(mask_empty,kernel,iterations = 1)



plt.figure(333)
plt.imshow(mask_empty)
plt.axis('off')
# plt.savefig("/home/hmoqadam/h_m/My_Participation/Writing/2.SEGMENTATION/_MODEL_/raddleham_pixelwise/results/result_first_training/"+
#             radar_profile_name+".png", dpi = 400, 
#             bbox_inches='tight', transparent="True", pad_inches=0)


df = pd.DataFrame(mask_empty)
df.to_csv("/home/hmoqadam/h_m/My_Participation/Writing/2.SEGMENTATION/_MODEL_/raddleham_pixelwise/results/result_first_training/"+"mask_"+ radar_profile_name+".csv",
          index=False, header=False)




































