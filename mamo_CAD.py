"""
Original code: 
    
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
Adapted by Sam Kelly, Hang Min and Devin Wilson for mammographic mass detection and segmentation.


"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.image
import glob
import scipy.misc
from PIL import Image
#import imgaug 
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.getcwd()
ROOT_DIR = ROOT_DIR+"/Mask_r_cnn/"

MAMOGRAM_IMAGE_DIR = "/scans/pseudo_color_image/" #Path of the mammograms
MAMOGRAM_MASK_DIR = "/scans/mask/"# Path of the ground truth masks


# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join("/your_path/", "logs")#Log directory for saving the weights
DEMO_SAVE_DIR = "scans/seg_mask/"# path to save the segmentation masks
if not os.path.exists(DEMO_SAVE_DIR):        
    os.mkdir(DEMO_SAVE_DIR)

############################################################
#  Configurations
############################################################


class MamogramConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mamogram"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + lesion

    # Number of training steps per epoch,set to the number of training data here
    STEPS_PER_EPOCH = 100

    # Number of validation steps after each round of training
    VALIDATION_STEPS = 10
    # Resize mode: "none" or "square"

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Skip detections with < DETECTION_MIN_CONFIDENCE
    DETECTION_MIN_CONFIDENCE = 0.965 # alter this during testing to generate different TPR at different FPI
    # 0.7 0.75 0.8 0.85 0.9 


############################################################
#  Dataset
############################################################

class MamogramDataset(utils.Dataset):

    def load_mamogram(self, subset):
        """This method loads the actual image
        subset is either "train" or "val" depending on whether the image is part of the training or validation datasets 
        """
        # Add classes. We have only one class to add.
        # These are the things that will be segmented
        self.add_class("mamogram", 1, "lesion")

        # Train or validation dataset?

        #list all the files in the directory with the mamogram images
        files = os.listdir(ROOT_DIR + MAMOGRAM_IMAGE_DIR + subset + "/")

        for fname in files:
            self.add_image("mamogram", image_id=fname, path=ROOT_DIR + MAMOGRAM_IMAGE_DIR + subset +"/"+ fname, subset=subset, fname=fname)


    def load_mask(self, image_id):
        """load the instance masks for an image.
        Returns:
        a tuple containing:
        masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.

        use dtype=np.int32
        """
        info = self.image_info[image_id]
        fname = info['fname']
       

        files = glob.glob(ROOT_DIR + MAMOGRAM_MASK_DIR + info['subset'] + fname[0:-4] + "*")

        masks = []
        for i in range(0, len(files)):
            data = skimage.io.imread(files[i])
            
            if data.ndim != 1:
                data = skimage.color.rgb2gray(data)
          
            singleMask = data
            if i == 0:
                masks = np.zeros((singleMask.shape[0], singleMask.shape[1], len(files)))
            masks[:,:,i] = singleMask

 

        instanceMaskMap = np.array(np.ones([masks.shape[-1]], dtype=np.int32)) #this is VERY important: array of class ids in the order that they appear in bigdata
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        return (masks.astype(np.bool), instanceMaskMap)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
		Taken from utils.py, any refinements we need can be done here
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = MamogramDataset()
    dataset_train.load_mamogram("/train/")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MamogramDataset()
    dataset_val.load_mamogram("/val/")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    aug = iaa.Sequential([
        iaa.OneOf([iaa.Fliplr(0.5),
                   iaa.Flipud(0.5),
                   iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,augmentation=aug,
                layers='all')

def segment(model, imPath):
    
    image = skimage.io.imread(imPath)

    fname = imPath.split('/')[-1]
    mrcnnData = model.detect([image], verbose=1)
       # documentation for model.detect:
       # """Runs the detection pipeline.

       # images: List of images, potentially of different sizes.

       # Returns a list of dicts, one dict per image. The dict contains:
       # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
       # class_ids: [N] int class IDs
       # scores: [N] float probability scores for the class IDs
       # masks: [H, W, N] instance binary masks
       # """

    mrcnnData = mrcnnData[0] #model.detect takes a list of images, but here we only provide one image so the output is a list with just one element

    masks = mrcnnData['masks']
    for i in range(0, masks.shape[2]):
        #iterate through the masks
        maskSingle = np.squeeze(masks[:, :, i])
        file_name = DEMO_SAVE_DIR + "demo_mask_" + str(i) + "_" + fname + "_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        

        scipy.misc.imsave(file_name, maskSingle.astype(np.int64)) 


    print(mrcnnData)
    print("&&&&&&&&&&&: "+str(mrcnnData['rois']))
    print("&&&&&&&&&&&: "+str(mrcnnData['class_ids']))
    print("&&&&&&&&&&&: "+str(mrcnnData['scores']))

    return

def segmentWrapper(model, directory):
    """wrapper function for segment to take many images as an input, calls segment() on everything in the directory"""
    files = os.listdir(directory)
    for f in files:
        segment(model, directory + '/' + f)

def overlayResult(image, mask):
	"""Function to overlay segmentation mask on an image.
	usage: image_var = overlayResult(image, dict['masks'] || masks_var)
	
	image: RGB or grayscale image [height, width, 1 || 3].
	mask: segmentation mask [height, width, instance_count]
	
	returns resulting image.
	"""
	# Image is already in grayscale so we skip converting it
	# May need to create 3 dimensions if single dimension image though so
	# will add this as a placeholder
	gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
	# Copy color pixels from the original color image where mask is set
	if mask.shape[-1] > 0:
		#collapse masks into one layer
		mask = (np.sum(mask, -1, keepdims=True) >= 1)
		overlay = np.where(mask, image, gray).astype(np.uint8)
	else:
		overlay = gray.astype(np.uint8)
		
	return overlay
	
	
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect breast lesions.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'segment'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=False,
                        metavar='/path/to/image',
                        help="Path to image file for segmentation")
    args = parser.parse_args()

    # Configurations
    if args.command == "train":
        config = MamogramConfig()
    else:
        class InferenceConfig(MamogramConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "segment":
        if os.path.isdir(args.image):
            segmentWrapper(model, args.image)
        else:
            segment(model, args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'segment'".format(args.command))
