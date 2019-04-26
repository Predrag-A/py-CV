# python Scripts/object_annotation.py -i Images/oa_beagle.jpg -d Data
import numpy as np
import argparse
import cv2
import imutils


# Classify object in image
def classify(input_img, model, threshold=0.9):
    # our CNN requires fixed spatial dimensions for our input image(s)
    # so we need to ensure it is resized to 224x224 pixels while
    # performing mean subtraction (104, 117, 123) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 224, 224)
    blob = cv2.dnn.blobFromImage(input_img, 1, (224, 224), (104, 117, 123))

    # set blob as input and forward pass to obtain classification
    model.setInput(blob)
    preds = net.forward()

    # sort the indexes of the probabilities in descending order (higher
    # probability first) and grab the top prediction
    idx = np.argsort(preds[0])[::-1][0]

    # return index of object in classifier if probability is higher than threshold
    if preds[0][idx] > threshold:
        return idx
    return -1


def pyramid(input_img, scale=1.5, min_size=(30, 30)):
    # yield the original image
    yield input_img

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(input_img.shape[1] / scale)
        input_img = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if input_img.shape[0] < min_size[1] or input_img.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield input_img


def sliding_window(input_img, step_size, window_size):
    # slide a window across the image
    for y in range(0, input_img.shape[0], step_size):
        for x in range(0, input_img.shape[1], step_size):
            # yield the current window
            yield (x, y, input_img[y:y + window_size[1], x:x + window_size[0]])


###############################################################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-d", "--data", required=True, help="path to data folder")
ap.add_argument("-t", "--threshold", required=False, type=float, default=0.9, help="classification threshold")
args = vars(ap.parse_args())

# load the input image from disk and define window width and height
image = cv2.imread(args["image"])
(winW, winH) = (128, 128)

# load the class labels and serialized model from disk
rows = open(args["data"] + "/synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
net = cv2.dnn.readNetFromCaffe(args["data"] + "/bvlc_googlenet.prototxt ",
                               args["data"] + "/bvlc_googlenet.caffemodel")

# initialize colors for every class label
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

###############################################################

for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, step_size=32, window_size=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        index = classify(window, net, args["threshold"])
        if index > 0:
            color = colors[index]
            color = [int(c) for c in color]

            text = "Label: {}".format(classes[index])
            cv2.rectangle(image, (x, y), (x + winW, y + winH), color, 2)
            cv2.putText(image, text, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# display the output image
cv2.imshow("Image", image)

###############################################################

cv2.waitKey(0)
cv2.destroyAllWindows()

