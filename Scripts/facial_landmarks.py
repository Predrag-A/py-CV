from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


# draw third eye, lab assignment
def third_eye(points, img, color):
    dist = points[42, 0] - points[39, 0]
    eye_points = points[36:42]
    for (x, y) in eye_points:
        cv2.circle(img, (x + int(dist/2), y - dist), 3, color, -1)
    return img


# initialize Viola Jones face detector and then create the facial landmark predictor
detector = cv2.CascadeClassifier("Data/haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("Data/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

###############################################################

while True:
    ret, image = cap.read()
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    faces = detector.detectMultiScale(gray)

    # convert detected rects so dlib can use them
    rects = dlib.rectangles()
    for(x, y, w, h) in faces:
        rects.append(dlib.rectangle(x,y,x+w,y+h))

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        image = third_eye(shape, image, (255, 0, 0))

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
