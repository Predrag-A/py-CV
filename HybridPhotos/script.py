import argparse
import numpy as np
import cv2


# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--first_image", required=True, help="first input image")
ap.add_argument("-i2", "--second_image", required=True, help="second input image")
args = vars(ap.parse_args())

# Load images
imgFirst = cv2.imread(args["first_image"], flags=cv2.IMREAD_GRAYSCALE)
imgSecond = cv2.imread(args["second_image"], flags=cv2.IMREAD_GRAYSCALE)

imgInF = np.float32(imgFirst)

dft = cv2.dft(imgInF, flags = cv2.DFT_COMPLEX_OUTPUT)

dft = np.fft.fftshift(dft)

###############################################################
# Obrada u frekventnom domenu
###############################################################

dft = np.fft.ifftshift(dft)

imgOutF = cv2.idft(dft, flags = cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
imgOutF = np.clip(imgOutF, 0, 255)
imgOut = np.uint8(imgOutF)

cv2.imwrite("IzlaznaSlika.png", imgOut)
cv2.imshow("Output", imgOut)
cv2.waitKey(0)

cv2.destroyAllWindows()