# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:40:36 2018

@author: uqhmin
"""
import numpy as np
import cv2
from skimage.measure import label, regionprops


class Preprocess:
    def __init__(self, rawim, im, breast_mask, lesion_mask):
        self.raw = rawim
        self.image = im
        self.mask = breast_mask
        self.lesion_mask = lesion_mask
   
    def extract_breast_profile(image,lesion_mask, if_crop):
        
        breast_mask = np.zeros(np.shape(image))
        breast_mask[image>0]=1
        
        labelim = label(breast_mask)
        props =  regionprops(labelim)
#        find the largest object as the breast
        area = 0
        ind = 1
        for i in range(0,len(props)):
            if area<props[i].filled_area:
                area = props[i].filled_area
                ind = i+1
        breast_mask = np.zeros(np.shape(image))
        breast_mask[labelim==ind]=1  
        labelim = label(breast_mask)       
        props =  regionprops(labelim)
        boundingbox = props[0].bbox
#        crop the breast mask and mammogram
        if if_crop == 1:
            breast_mask = breast_mask[boundingbox[0]:boundingbox[2],boundingbox[1]:boundingbox[3]]
            breast_raw_image = image[boundingbox[0]:boundingbox[2],boundingbox[1]:boundingbox[3]]
            lesion_mask = lesion_mask[boundingbox[0]:boundingbox[2],boundingbox[1]:boundingbox[3]]
        else:
            breast_raw_image = image
#        breast_image = rescale2uint8(breast_raw_image,breast_mask)
        breast_image = rescale2uint16(breast_raw_image,breast_mask)
        return Preprocess(breast_raw_image,breast_image,breast_mask,lesion_mask)
    
def rescale2uint8(image,breast_mask):
    intensity_in_mask = image[breast_mask>0]
#    use top 0.2 percentile to do the strech
    maxi = np.percentile(intensity_in_mask,99.8)#np.max(intensity_in_mask)
    mini = np.percentile(intensity_in_mask,0.2)#np.min(intensity_in_mask)
#        stretch the image into 0~255
    
    image = 255*(image-mini)/(maxi-mini)
    image[breast_mask==0] = 0
    image[image<0] = 0
    image[image>255] = 255
    image = np.uint8(image)
          
    return image

def rescale2uint16(image,breast_mask):
    intensity_in_mask = image[breast_mask>0]
#    use top 0.2 percentile to do the strech
    maxi = np.percentile(intensity_in_mask,99.8)#np.max(intensity_in_mask)
    mini = np.percentile(intensity_in_mask,0.2)#np.min(intensity_in_mask)
#        stretch the image into 0~255
    
    image = 65535*(image-mini)/(maxi-mini)
    image[breast_mask==0] = 0
    image[image<0] = 0
    image[image>65535] = 65535
    image = np.uint16(image)
          
    return image


        
        
    
    