from imutils import paths
import imutils
import numpy as np
import argparse
import cv2


# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="input directory of images")
ap.add_argument("-o", "--output_path", type=str, default=None, help="image output path")
args = vars(ap.parse_args())

# Get all image paths from folder
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)


###############################################################
# Complete panorama

###############################################################

