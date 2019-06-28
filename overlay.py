# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:52:14 2018

@author: uqhmin
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, io, img_as_float

def mask_overlay(img, mask):
    alpha = 0.5

    img = img_as_float(img)
    rows, cols = img.shape
    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))
    img_hsv = color.rgb2hsv(img_color)
    color_mask = np.zeros((rows, cols, 3))
    color_mask1 = color_mask[:,:,1]
    color_mask1[np.where(mask>0)] = 1
    color_mask[:,:,1] = color_mask1
    color_mask_hsv = color.rgb2hsv(color_mask)
    
    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    
    img_masked = color.hsv2rgb(img_hsv)
    
    return img_masked
