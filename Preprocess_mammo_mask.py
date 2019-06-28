# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:46:32 2018

@author: Hang Min

Prepare the mammograms for pseudo-color transformation

To normalize the INbreast mammogram to 16bit

To pad the mammograms and masks to a square


"""



import os
import numpy as np
from Preprocess_mammo import Preprocess 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from utility import padimages
#from skimage.segmentation import mark_boundaries
from skimage import io
from skimage.measure import label
from overlay import mask_overlay
import timeit

start = timeit.default_timer()

import warnings
warnings.filterwarnings("ignore")

#from skimage import data, color, io, img_as_float

image_path = "scans/raw_mammogram/"
annotation_path = 'scans/raw_annotation/'

save_image_path = "scans/preprocessed_image/"   
if not os.path.exists(save_image_path):        
    os.mkdir(save_image_path)
 
    
save_mask_path = "scans/preprocessed_mask/"    
if not os.path.exists(save_mask_path):        
    os.mkdir(save_mask_path)


file_names = os.listdir(image_path)    
file_names = sorted(file_names)


for i in range(0,len(file_names)):

    print(file_names[i])
    mammo = io.imread(image_path+file_names[i],0)
   
    lesion_mask = io.imread(annotation_path+file_names[i])
    
    if np.max(lesion_mask)>=0:
#    Extract the breast profile and crop the mammogram, breast mask and the lesion mask
#    Normalize the image into 16-bit
        breast_preprocess = Preprocess.extract_breast_profile(mammo,lesion_mask,1)
               
        mammo = breast_preprocess.image
        breast_mask = breast_preprocess.mask
        lesion_mask =breast_preprocess.lesion_mask    
        
        print ('Number of lesions: '+str(np.max(np.unique(label(lesion_mask)))))
        
#   pad the image, to ensure the aspect ratio is 1:1
        pad_mammo = padimages(mammo,file_names[i],1)
        
#    save the preprocessed image

        io.imsave(save_image_path + file_names[i],pad_mammo)
        
        #if the image has more than 1 lesion, then seperate them into different masks and number them.
        
        labelim = label(lesion_mask)
        if np.max(labelim)>0:
#            if there is at least 1 lesion.
            for l in range(1,np.max(labelim+1)):
                l_mask = np.zeros(np.shape(labelim))
                l_mask = l_mask.astype(lesion_mask.dtype)
                l_mask [labelim==l] = 255
                num_nonzero = np.where(l_mask>0)
                num_nonzero = len(num_nonzero[0])

                if num_nonzero>15:
                    print('A valid mask')
#                   Pad the mask in the same way as padding the image
                    pad_l_mask = padimages(l_mask,file_names[i],1)
                    io.imsave(save_mask_path+file_names[i][:-4]+str(l)+'.png',pad_l_mask)
                else:
                    print('Has a tiny piece of noise that is not valid for training!')
                    
        else:# if there is no lesion
            pad_lesion_mask = padimages(lesion_mask,file_names[i],1)
            io.imsave(save_mask_path+file_names[i][:-4]+str(0)+'.png',pad_lesion_mask)
       
stop = timeit.default_timer()
print('RunTime per image: ', (stop - start)/ len(file_names)) 