# python HybridPhotos/script.py -i1 Images/hp_einstein.bmp -i2 Images/hp_marilyn.bmp
import argparse
import numpy as np
import cv2


def nothing(self):
    pass


def remove_frequencies(freq_img, radius, high_freq):
    freq_edited = np.copy(freq_img)
    rows = np.size(freq_edited, 0)
    cols = np.size(freq_edited, 1)
    center_rows, center_cols = int(round(rows / 2)), int(round(cols / 2))
    freq_edited[center_rows - radius:center_rows + radius, center_cols - radius:center_cols + radius] = 0
    if high_freq:
        freq_edited = freq_img - freq_edited
    return freq_edited


def combine_frequencies(high_freq_img, low_freq_img, radius):
    return remove_frequencies(high_freq_img, radius, True) + remove_frequencies(low_freq_img, radius, False)


# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--first_image", required=True, help="first input image")
ap.add_argument("-i2", "--second_image", required=True, help="second input image")
ap.add_argument("-s", "--steps", default=False, type=bool, help="show steps")
args = vars(ap.parse_args())

# Load images
imgFirst = cv2.imread(args["first_image"], flags=cv2.IMREAD_GRAYSCALE)
imgSecond = cv2.imread(args["second_image"], flags=cv2.IMREAD_GRAYSCALE)

imgInF = np.float32(imgFirst)
dftF = cv2.dft(imgInF, flags = cv2.DFT_COMPLEX_OUTPUT)
dftF = np.fft.fftshift(dftF)

imgInS = np.float32(imgSecond)
dftS = cv2.dft(imgInS, flags = cv2.DFT_COMPLEX_OUTPUT)
dftS = np.fft.fftshift(dftS)

cv2.namedWindow("Output")
cv2.createTrackbar("Radius", "Output", 0, 150, nothing)

###############################################################
# Obrada u frekventnom domenu

# magnitude_spectrum = np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))
# out_mag = np.uint8(255.0 * magnitude_spectrum / magnitude_spectrum.max())
# cv2.imshow('FFT Magnitude Before', out_mag)

while True:

    if cv2.waitKey(1) == 27:
        break

    dft = combine_frequencies(dftF, dftS, cv2.getTrackbarPos("Radius", "Output"))
    dft = np.fft.ifftshift(dft)
    imgOut = cv2.idft(dft, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    imgOut = np.clip(imgOut, 0, 255)
    imgOut = np.uint8(imgOut)
    cv2.imshow("Output", imgOut)


###############################################################

cv2.destroyAllWindows()