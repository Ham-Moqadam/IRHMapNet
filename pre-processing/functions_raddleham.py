

### ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: ###
###     date: 03.07.2024
###     author: H.Moqadam
###     description: module for the new deep learning pipeline for the 
###
###
### ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: ###


class PreProcessing:
    def normalize_matrix(self, image, lower_range, upper_range):
        """
        normalizes the image, within the given the upper and lower limit
        """
        import numpy as np
    
        a = np.array((image - np.min(image)) / (np.max(image) - np.min(image)))
        b = upper_range - lower_range
        c = lower_range
        answer = a*b + c
        
        return answer


    def getting_file_names(self, directory, sort = True):
        """
        This functuion gets the name of all the files containing in a directory
        and sorts them alphabetically
        """ 
        import os
    
        a = os.listdir(directory)
        if sort:
            a = sorted(a)
        return a    
    
    
    
    def dilate_image(self, image, kernel_size):
        """
        
        """
        import numpy as np
        import cv2 as cv

        kernel_length_1, kernel_length_2 = kernel_size
        kernel = np.ones((kernel_length_1, kernel_length_2), np.uint8)

        return cv.dilate(image, kernel)
    
    
    
    
    # def cut_grams_noise(self, read_directory, profile_names, cutoff_row , save_directory, skiprows=1,):
    #     """
    #     """
    #     import numpy as np
    #     import pandas as pd
        
    #     radargram = np.loadtxt(read_directory+str(profile_names[i])+"_proc.csv", skiprows = skiprows, delimiter = ",")

    #     radargram = radargram[:cutoff_row,:]
    #     df = pd.DataFrame(radargram)
    #     df.to_csv(save_directory+str(profile_names[i])+"_proc.csv", index=False, header=False)
        
    


    


class ImageProcessing:
    def function_a(cls):
        print('function A')
    
    def function_b(cls):
        print('function B')
        
        
    
    ### ~~~~~~~~~~~~~~ Function to take the gradient fo the image in Y-DIRECTION
    def gradient_in_y(self, image_matrix):
    	""" gradient of the image matrix in the y direction
    	"""
    	import numpy as np
    
    	grad_matrix = np.zeros(image_matrix.shape)
    	for i in range(image_matrix.shape[1]):
    		grad_matrix[:,i] = np.absolute(np.gradient(image_matrix[:,i]))
    	return grad_matrix
    ### ~~~~~~~~~~~~~~ END Function to take the gradient fo the image in Y-DIRECTION




    
    ### ~~~~~~~~~~~~~~ Function to take the gradient fo the image in X-DIRECTION
    def gradient_in_x(self, image_matrix):
    	""" gradient of the image matrix in the y direction
    	"""
    	import numpy as np
    
    	grad_matrix = np.zeros(image_matrix.shape)
    	for i in range(image_matrix.shape[0]):
    		grad_matrix[i,:] = np.absolute(np.gradient(image_matrix[i,:]))
    	return grad_matrix
    ### ~~~~~~~~~~~~~~ END Function to take the gradient fo the image in X-DIRECTION
    





    
    
"""

### frm radar image processing:
    
    def smooth_normalize_and_clean(img_matrix, cutoff):
         a       = img_matrix
         b       = ndimage.convolve(a, (1/9)*np.ones((3,3)), mode="constant")
         c       = normalize(b)
         d       = np.logical_and(c >= cutoff, c).astype(np.float32)
         return d


    
for i in range(1,laplacian_Rdata.shape[0]-1):
    for j in range(1,laplacian_Rdata.shape[1]-1):
        if laplacian_Rdata[i,j] ==0 and laplacian_Rdata[i,j-1] != 0 and laplacian_Rdata[i,j+1] != 0:
            laplacian_Rdata[i,j] = 500


"""



class PostProcessing:
    def label_connected_comps(self, image, size_threshold, connectivity):
        """
        Label the connected components
        Calculate the size of each component
        Define the size threshold
        Create a mask for components larger than the threshold
        Apply the mask to the labeled image to remove small components
        Convert the filtered image back to binary format
        
        Parameters
        ----------
        image : np.ndarray
            Binary input image.
        size_threshold : int
            Minimum size of components to keep.
        connectivity : int
            Connectivity for labeling (4 or 8).
        
        Returns
        -------
        filtered_binary_image : np.ndarray
            Binary image with small components removed.
        """
        import numpy as np
        from scipy.ndimage import label
        
        if connectivity == 4:
            structure = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]])
        elif connectivity == 8:
            structure = np.ones((3, 3), dtype=int)
        else:
            raise ValueError("Connectivity must be 4 or 8.")
        
        labeled_image, num_labels = label(image, structure=structure)
        
        component_sizes = np.bincount(labeled_image.ravel())
        large_components_mask = np.isin(labeled_image, np.where(component_sizes >= size_threshold)[0])
        filtered_image = labeled_image * large_components_mask
        filtered_binary_image = (filtered_image > 0).astype(np.uint8)
        
        return filtered_binary_image











        



