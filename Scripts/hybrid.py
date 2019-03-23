# python Scripts/hybrid.py -i1 Images/hp_einstein.jpg -i2 Images/hp_marilyn.jpg
import argparse
import numpy as np
import cv2


# Empty function for track bar
def nothing(self):
    pass


# Create circular mask for image with set radius
def create_circular_mask(h, w, radius):
    # Create two matrices of values from 0 to h and 0 to w
    y, x = np.ogrid[:h, :w]
    # Calculate center coordinates
    center_x, center_y = int(w/2), int(h/2)
    # For each (x, y) calculate distance from center and create matrix from those values
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Create binary mask where every value outside of radius is false
    mask = dist_from_center <= radius
    return mask


# Remove high (default) or low frequencies from DFT image
def remove_frequencies(freq_img, radius, low_freq=False):
    freq_edited = np.copy(freq_img)
    mask = create_circular_mask(freq_img.shape[1], freq_img.shape[0], radius=radius)
    # Set all masked values to 0 to remove high frequencies
    freq_edited[mask] = 0
    # Remove low frequencies by subtracting image with removed high
    # frequencies from original image
    if low_freq:
        freq_edited = freq_img - freq_edited
    return freq_edited


# Combine high frequencies of first image with low frequencies of second image
def combine_frequencies(high_freq_img, low_freq_img, radius, show_spectrum):
    high_removed = remove_frequencies(high_freq_img, radius)
    low_removed = remove_frequencies(low_freq_img, radius, True)
    # Show magnitude spectrum if required
    if show_spectrum:
        with np.errstate(divide="ignore"):
            # Calculate magnitude and scale to values in 0-255 range
            mag_high = np.log(cv2.magnitude(high_removed[:, :, 0], high_removed[:, :, 1]))
            mag_high = np.uint8(255.0 * mag_high / mag_high.max())

            mag_low = np.log(cv2.magnitude(low_removed[:, :, 0], low_removed[:, :, 1]))
            mag_low = np.uint8(255.0 * mag_low / mag_low.max())

            cv2.imshow("FFT Magnitude High Removed", mag_high)
            cv2.imshow("FFT Magnitude Low Removed", mag_low)

    return high_removed + low_removed


# Create hybrid image from two input grayscale images
def create_hybrid(high_freq_img, low_freq_img, radius=10, show_spectrum=False):
    # Convert first image with Discrete Fourier Transform and shift quadrants
    high_freq_float = np.float32(high_freq_img)
    dft_high = cv2.dft(high_freq_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_high = np.fft.fftshift(dft_high)

    # Convert second image with Discrete Fourier Transform and shift quadrants
    low_freq_float = np.float32(low_freq_img)
    dft_low = cv2.dft(low_freq_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_low = np.fft.fftshift(dft_low)

    # Combine frequencies of two images and reverse quadrant shifts and DFT
    dft = combine_frequencies(dft_high, dft_low, radius, show_spectrum)
    dft = np.fft.ifftshift(dft)
    res_img = cv2.idft(dft, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return res_img


###############################################################

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--first_image", required=True, help="first input image")
ap.add_argument("-i2", "--second_image", required=True, help="second input image")
ap.add_argument("-o", "--output_path", default=None, help="image output path")
ap.add_argument("-s", "--small_output", default=False, help="show small image for comparison")
ap.add_argument("-m", "--magnitude_show", default=False, help="show magnitude spectrum")
args = vars(ap.parse_args())

# Load images
imgFirst = cv2.imread(args["first_image"], flags=cv2.IMREAD_GRAYSCALE)
imgSecond = cv2.imread(args["second_image"], flags=cv2.IMREAD_GRAYSCALE)

# Get height and width and resize second image to size of first if needed
height, width = imgFirst.shape[:2]
if imgSecond.shape[0] != height or imgSecond.shape[1] != width:
    imgSecond = cv2.resize(imgSecond, (width, height))

# Create window with track bar, choose lesser of width/height for max radius
cv2.namedWindow("Output")
cv2.createTrackbar("Radius", "Output", 0, int(min(width, height)/2), nothing)

###############################################################

while True:
    imgOut = create_hybrid(imgFirst, imgSecond, cv2.getTrackbarPos("Radius", "Output"), args["magnitude_show"])
    imgOut = np.clip(imgOut, 0, 255)
    imgOut = np.uint8(imgOut)
    cv2.imshow("Output", cv2.resize(imgOut, (int(width*2), int(height*2)), cv2.INTER_AREA))

    # Display smaller version of output image
    if args["small_output"]:
        cv2.imshow("Output Smaller", cv2.resize(imgOut, (int(width/4), int(height/4))))

    # Exit while loop with ESC key and save image if specified
    if cv2.waitKey(1) == 27:
        if args["output_path"] is not None:
            cv2.imwrite(args["output_path"], imgOut)
            print(args["output_path"])
        break

###############################################################

cv2.destroyAllWindows()
