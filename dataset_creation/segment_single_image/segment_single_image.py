####################################################################################################
    # Imports
####################################################################################################
import os
import cv2
import numpy as np

####################################################################################################
    # Main Function
####################################################################################################

def main():
    # Load image
    directory = os.path.dirname(os.path.realpath(__file__))
    img_location = os.path.join(directory, "img_00.jpg")
    img = cv2.imread(img_location, cv2.IMREAD_GRAYSCALE)

    # Process image
    img = cv2.medianBlur(img, 41)
    save_img(img, "01_blur")

    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    save_img(img, "02_thresh")

    kernel = 11
    img = cv2.erode(img, np.ones((kernel, kernel),np.uint8), iterations = 10)
    save_img(img, "03_erode")

    img = cv2.dilate(img, np.ones((kernel, kernel),np.uint8), anchor=(0, 1), iterations = 6)
    save_img(img, "04_dilate")

####################################################################################################
    # Auxiliary Functions
####################################################################################################
def save_img(img, string):
    directory = os.path.dirname(os.path.realpath(__file__))
    save_location = os.path.join(directory, "img_" + string + ".jpg")
    cv2.imwrite(save_location, img)

if __name__ == "__main__":
    main()
    print('Finished')