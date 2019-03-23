# python Scripts/panorama.py -i Images/Panorama
from imutils import paths
import imutils
import numpy as np
import argparse
import cv2


# Remove black pixels around content of image
def trim(img):
    # Convert to grayscale and create mask of non-black pixels
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = img_gray > 0
    # Get coordinates of non-zero elements
    coords = np.argwhere(mask)
    # Find minimum and maximum coordinates
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    # Crop image at min and max coordinates
    return img[x0:x1, y0:y1]


# Compute key points and features of image
def detect_and_describe(img):
    # Create SIFT detector
    descriptor = cv2.xfeatures2d.SIFT_create()
    # Extract key points and features
    kps, features = descriptor.detectAndCompute(img, None)
    # Convert key points to Numpy array
    kps = np.float32([kp.pt for kp in kps])
    return kps, features


# Match features of two images together
def match_key_points(key_pts_a, key_pts_b, features_a, features_b, ratio, thresh):
    # Create BruteForce (calculates Euclidean distance between all feature vectors
    # and finds the pairs that have the smallest distance) descriptor matcher
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # Perform k-NN matching which returns top two matches for each feature vector
    raw_matches = matcher.knnMatch(features_a, features_b, 2)
    matches = []

    # Loop over matches
    for m in raw_matches:
        # Ensure distance is within a certain ratio (Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # Homography matrix computation requires at least 4 matches
    if len(matches) > 4:
        # Construct two sets of points
        key_pts_a = np.float32([key_pts_a[iterator] for (_, iterator) in matches])
        key_pts_b = np.float32([key_pts_b[iterator] for (iterator, _) in matches])
        # Calculate homography matrix using RANSAC algorithm
        h, _ = cv2.findHomography(key_pts_a, key_pts_b, cv2.RANSAC, thresh)

        return h
    return None


def stitch_images(image_a, image_b, ratio, thresh):
    # Extract key points and features from images
    kps_a, features_a = detect_and_describe(image_a)
    kps_b, features_b = detect_and_describe(image_b)
    # Generate homography matrix
    homography = match_key_points(kps_a, kps_b, features_a, features_b, ratio, thresh)
    # If homography is None there aren't enough matched key points
    if homography is None:
        return None
    # Warp image_a according to the homography matrix and copy image_b into the result
    result = cv2.warpPerspective(image_a, homography, (image_a.shape[1] + image_b.shape[1],
                                                       (image_a.shape[0] + int(image_b.shape[0]/2))))
    result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b
    return result


def create_panorama(imgs, ratio=0.75, thresh=4.0):
    # Go through images in array and stitch all of them in order
    for i in range(len(imgs) - 1, 0, -1):
        res = stitch_images(imgs[i], imgs[i - 1], ratio, thresh)
        imgs[i - 1] = res
    # Return cropped result
    return trim(res)


###############################################################

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="input directory of images")
ap.add_argument("-o", "--output_path", type=str, default=None, help="image output path")
args = vars(ap.parse_args())

# Get all image paths from folder
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# Load all images into array
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

# Generate panorama and show result
imgOut = create_panorama(images)
cv2.imshow("Result", imgOut)

# Save image if specified
if args["output_path"] is not None:
    cv2.imwrite(args["output_path"], imgOut)
    print(args["output_path"])

###############################################################

cv2.waitKey(0)
cv2.destroyAllWindows()
