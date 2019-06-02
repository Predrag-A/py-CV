# python Scripts/aruco_estimation.py -a Aruco/ -c Data/calibration.yaml
import cv2
from cv2 import aruco
import yaml
import numpy as np
import argparse
import glob


def calibrate_camera(aruco_path, calibration_path):

    img_list = []

    for img_path in glob.glob(aruco_path + "/*.jpg"):
        img = cv2.imread(img_path)
        img_list.append(img)

    counter = []
    corners_list = []
    id_list = []
    first = True

    for im in img_list:
        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        counter.append(len(ids))

    counter = np.array(counter)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape,
                                                              None, None)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open(calibration_path, "w") as f:
        yaml.dump(data, f)


###############################################################

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--aruco", type=str, required=True, help="directory of aruco data")
ap.add_argument("-c", "--calibration", type=str, required=True, help="path of camera calibration data")
args = vars(ap.parse_args())


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

markerLength = 40
markerSeparation = 8

board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)
arucoParams = aruco.DetectorParameters_create()

calibrate_camera(args["aruco"], args["calibration"])

file = open(args["calibration"])

videoFile = args["aruco"] + "Aruco_board.mp4"
cap = cv2.VideoCapture(videoFile)

loadeddict = yaml.load(file, Loader=yaml.FullLoader)
camera_matrix = np.array(loadeddict.get('camera_matrix'))
dist_coeffs = np.array(loadeddict.get('dist_coeff'))
arucoParams = aruco.DetectorParameters_create()

###############################################################

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame_remapped_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)
        aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

        if ids is not None:
            im_with_aruco_board = aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs)
            if retval != 0:
                im_with_aruco_board = aruco.drawAxis(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 50)
        else:
            im_with_aruco_board = frame

        cv2.imshow("arucoboard", im_with_aruco_board)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    else:
        break

###############################################################

cap.release()
cv2.destroyAllWindows()
