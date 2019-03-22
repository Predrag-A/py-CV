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


def stitch(image_a, image_b, ratio=0.75, thresh=4.0, show_matches=False):

    kps_a, features_a = detect_and_describe(image_a)
    kps_b, features_b = detect_and_describe(image_b)

    m = match_key_points(kps_a, kps_b, features_a, features_b, ratio, thresh)
    if m is None:
        return None

    matches, h, status = m
    result = cv2.warpPerspective(image_a, h, (image_a.shape[1] + image_b.shape[1], image_a.shape[0]))
    result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b

    if show_matches:
        vis = draw_matches(image_a, image_b, kps_a, kps_b, matches, status)
        return result, vis
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


def draw_matches(img_a, img_b, kps_a, kps_b, matches, status):
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    visualization = np.zeros((max(h_a, h_b), w_a + w_b, 3), dtype="uint8")
    visualization[0:h_a, 0:w_a] = img_a
    visualization[0:h_b, w_a:] = img_b

    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            pt_a = (int(kps_a[queryIdx][0]), int(kps_a[queryIdx][1]))
            pt_b = (int(kps_b[trainIdx][0]) + w_a, int(kps_b[trainIdx][1]))
            cv2.line(visualization, pt_a, pt_b, (0, 255, 0), 1)
    return visualization


###############################################################

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
    image = imutils.resize(image, width=400)
    images.append(image)


###############################################################

for i in range(len(images)-1, 0, -1):
    res, vis = stitch(images[i], images[i-1], show_matches=True)
    images[i-1] = res
    cv2.imshow(str(i) + " Result", res)
    cv2.imshow(str(i) + " Vis", vis)

###############################################################

# cv2.imshow("Matches", vis)
# cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
