# python Scripts/object_annotation.py -i Images/oa_beagle.jpg -d Data
import numpy as np
import argparse
import cv2
import imutils


def classify(input_img, model, threshold=0.9):
    blob = cv2.dnn.blobFromImage(input_img, 1, (224, 224), (104, 117, 123))

    model.setInput(blob)
    preds = net.forward()

    idx = np.argsort(preds[0])[::-1][0]

    if preds[0][idx] > threshold:
        return idx
    return -1


def pyramid(input_img, scale=1.5, min_size=(30, 30)):

    yield input_img

    while True:
        w = int(input_img.shape[1] / scale)
        input_img = imutils.resize(image, width=w)

        if input_img.shape[0] < min_size[1] or input_img.shape[1] < min_size[0]:
            break

        yield input_img


def sliding_window(input_img, step_size, window_size):
    for y in range(0, input_img.shape[0], step_size):
        for x in range(0, input_img.shape[1], step_size):
            yield (x, y, input_img[y:y + window_size[1], x:x + window_size[0]])


###############################################################

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-d", "--data", required=True, help="path to data folder")
ap.add_argument("-t", "--threshold", required=False, type=float, default=0.9, help="classification threshold")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(winW, winH) = (128, 128)

rows = open(args["data"] + "/synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
net = cv2.dnn.readNetFromCaffe(args["data"] + "/bvlc_googlenet.prototxt ",
                               args["data"] + "/bvlc_googlenet.caffemodel")

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

###############################################################

counter = 0

for resized in pyramid(image, scale=1.5, min_size=(winW, winH)):
    for (x, y, window) in sliding_window(resized, step_size=32, window_size=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        index = classify(window, net, args["threshold"])
        if index > 0:
            color = colors[index]
            color = [int(c) for c in color]

            text = "Label: {}".format(classes[index])
            cv2.rectangle(resized, (x, y), (x + winW, y + winH), color, 2)
            cv2.putText(resized, text, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image" + str(counter), resized)
    counter = counter+1

#cv2.imshow("Image", resized)

###############################################################

cv2.waitKey(0)
cv2.destroyAllWindows()

