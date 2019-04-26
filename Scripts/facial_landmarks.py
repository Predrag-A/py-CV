# python Scripts/facial_landmarks.py -v Data/haarcascade_frontalface_default.xml
# -d Data/shape_predictor_68_face_landmarks.dat
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2


# Draw third eye (lab assignment)
def third_eye(points, img, color):
    dist = points[42, 0] - points[39, 0]
    eye_points = points[36:42]
    for (x, y) in eye_points:
        cv2.circle(img, (x + int(dist/2), y - dist), 3, color, -1)
    return img


# Draw facial landmarks on image
def draw_landmarks(input_img, detector, predictor, third):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    faces = detector.detectMultiScale(gray)

    # convert detected rects so dlib can use them
    rects = dlib.rectangles()
    for(x, y, w, h) in faces:
        rects.append(dlib.rectangle(x, y, x+w, y+h))

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(input_img, (x, y), 1, (0, 0, 255), -1)
            if third:
                input_img = third_eye(shape, image, (255, 0, 0))
    return input_img


###############################################################

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--viola_jones", type=str, required=True, help="path to viola jones classifier")
ap.add_argument("-d", "--dlib_predictor", type=str, required=True, help="path to dlib predictor")
ap.add_argument("-t", "--third_eye", type=bool, required=False, default=False, help="drawing third eye")
args = vars(ap.parse_args())

# initialize Viola Jones face detector and then create the facial landmark predictor
det = cv2.CascadeClassifier(args["viola_jones"])
pred = dlib.shape_predictor(args["dlib_predictor"])

cap = cv2.VideoCapture(0)

###############################################################

while True:
    # Load image from video stream and resize
    ret, image = cap.read()
    image = imutils.resize(image, width=500)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", draw_landmarks(image, det, pred, args["third_eye"]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

###############################################################

cap.release()
cv2.destroyAllWindows()
