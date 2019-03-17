# python HybridPhotos/script.jpg -i1 Images/hp_einstein.jpg -i2 Images/hp_marilyn.jpg
import argparse
import numpy as np
import cv2


def nothing(self):
    pass


def create_circular_mask(h, w, radius):
    center = [int(w/2), int(h/2)]
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius
    return mask


def remove_frequencies(freq_img, radius, high_freq=False):
    freq_edited = np.copy(freq_img)
    mask = create_circular_mask(freq_img.shape[1], freq_img.shape[0], radius=radius)
    freq_edited[~mask] = 0
    if high_freq:
        freq_edited = freq_img - freq_edited
    return freq_edited


def combine_frequencies(high_freq_img, low_freq_img, radius):
    return remove_frequencies(high_freq_img, radius, True) + remove_frequencies(low_freq_img, radius)


# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--first_image", required=True, help="first input image")
ap.add_argument("-i2", "--second_image", required=True, help="second input image")
ap.add_argument("-o", "--output_path", default=None, help="image output path")
ap.add_argument("-s", "--small_output", default=False, help="show small image for comparison")
args = vars(ap.parse_args())

# Load images
imgFirst = cv2.imread(args["first_image"], flags=cv2.IMREAD_GRAYSCALE)
imgSecond = cv2.imread(args["second_image"], flags=cv2.IMREAD_GRAYSCALE)

height, width = imgFirst.shape[:2]

imgInF = np.float32(imgFirst)
dftF = cv2.dft(imgInF, flags=cv2.DFT_COMPLEX_OUTPUT)
dftF = np.fft.fftshift(dftF)

imgInS = np.float32(imgSecond)
dftS = cv2.dft(imgInS, flags=cv2.DFT_COMPLEX_OUTPUT)
dftS = np.fft.fftshift(dftS)

cv2.namedWindow("Output")
cv2.createTrackbar("Radius", "Output", 0, int(width/2), nothing)

###############################################################

while True:

    dft = combine_frequencies(dftF, dftS, cv2.getTrackbarPos("Radius", "Output"))
    dft = np.fft.ifftshift(dft)
    imgOut = cv2.idft(dft, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    imgOut = np.clip(imgOut, 0, 255)
    imgOut = np.uint8(imgOut)
    cv2.imshow("Output", cv2.resize(imgOut, (int(width*2), int(height*2))))

    if args["small_output"]:
        cv2.imshow("Output Smaller", cv2.resize(imgOut, (int(width/4), int(height/4))))

    if cv2.waitKey(1) == 27:
        if args["output_path"] is not None:
            cv2.imwrite(args["output_path"], imgOut)
            print(args["output_path"])
        break

###############################################################

cv2.destroyAllWindows()
