# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:57:36 2019

@author: uqhmin
"""
import numpy as np
# To pad image into a square shape
def padimages(image,file_name, ratio):
    [length, width] = np.shape(image)
    if length/width>ratio:#1024/800
        print('This image needs padding.')
        add_wid = round(length*(1/ratio)-width)
        pad = np.zeros((length,add_wid))
        pad = pad.astype(image.dtype)
        if '_R_' in file_name:
        #                pad on the left
            pad_image = np.concatenate((pad,image),axis=1)
        else:
            pad_image = np.concatenate((image,pad),axis=1)
            
    return pad_image