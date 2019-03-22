# python Scripts/panorama.py -i Images/Panorama
from imutils import paths
import imutils
import numpy as np
import argparse
import cv2


def trim(frame):

    if not np.sum(frame[0]):
        return trim(frame[1:])

    if not np.sum(frame[-1]):
        return trim(frame[:-2])

    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])

    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


def stitch(image_a, image_b, ratio, thresh):

    kps_a, features_a = detect_and_describe(image_a)
    kps_b, features_b = detect_and_describe(image_b)

    m = match_key_points(kps_a, kps_b, features_a, features_b, ratio, thresh)
    if m is None:
        return None

    matches, h, status = m
    result = cv2.warpPerspective(image_a, h, (image_a.shape[1] + image_b.shape[1], image_a.shape[0]))
    result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b

    return result


def detect_and_describe(img):
    # Create SIFT detector
    descriptor = cv2.xfeatures2d.SIFT_create()
    # Extract key points and features
    kps, features = descriptor.detectAndCompute(img, None)
    # Convert key points to Numpy array
    kps = np.float32([kp.pt for kp in kps])
    return kps, features


def match_key_points(key_pts_a, key_pts_b, features_a, features_b, ratio, thresh):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(features_a, features_b, 2)
    matches = []

    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        key_pts_a = np.float32([key_pts_a[iterator] for (_, iterator) in matches])
        key_pts_b = np.float32([key_pts_b[iterator] for (iterator, _) in matches])

        (h, status) = cv2.findHomography(key_pts_a, key_pts_b, cv2.RANSAC, thresh)

        return matches, h, status
    return None


def create_panorama(imgs, ratio=0.75, thresh=4.0):
    res = None
    for i in range(len(imgs) - 1, 0, -1):
        res = stitch(imgs[i], imgs[i - 1], ratio, thresh)
        imgs[i - 1] = res
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
    image = imutils.resize(image, width=400)
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
