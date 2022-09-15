####################################################################################################
    # Imports
####################################################################################################
import os
import cv2
import numpy as np
import random

####################################################################################################
    # Main Function
####################################################################################################
# Input
SOURCE_IMG_DIR = "/home/seb/Datasets/sopra/source_images"
BACKGR_IMG_DIR = "/home/seb/Datasets/augmentation_test/coco/val2017"
# Output
AUGM_IMG_DIR = "/home/seb/Datasets/sopra/augmented_images"
MASK_DIR =     "/home/seb/Datasets/sopra/augmented_masks"
OVERLAY_DIR =  "/home/seb/Datasets/sopra/mask_source_overlay"

def main():
    source_img_list = os.listdir(SOURCE_IMG_DIR)
    backgr_img_list = os.listdir(BACKGR_IMG_DIR)

    loop_through = 3
    source_img_list = loop_through*source_img_list
    image_count = len(source_img_list)

    random.shuffle(source_img_list)
    random.shuffle(backgr_img_list)

    for i, (img_name, backgr_img_name) in enumerate(zip(source_img_list, backgr_img_list)):
        print(str(i) + "/" + str(image_count) + ": merged '" + img_name + "' with background '" + backgr_img_name + "'")
        img = cv2.imread(os.path.join(SOURCE_IMG_DIR, img_name), cv2.IMREAD_COLOR)
        backgr = cv2.imread(os.path.join(BACKGR_IMG_DIR, backgr_img_name), cv2.IMREAD_COLOR)
        mask = process_img(img, blur=31, kernel=3, iterations=15)
        img, mask = overlay_img_on_backgr(img, mask, backgr, min_scale=0.2, blur_kernel=3, blur_std=3, p_occl = 0)
        overlay = overlay_mask_on_source(img, mask)

        # Save images
        save_img(img, dir=AUGM_IMG_DIR, name="img_{0}.png".format(i))
        save_img(mask, dir=MASK_DIR, name="mask_{0}.png".format(i))
        save_img(overlay, dir=OVERLAY_DIR, name="overlay_{0}.png".format(i))


####################################################################################################
    # Auxiliary Functions
####################################################################################################
def save_img(img, dir, name):
    name = name.replace("jpg", "png")
    save_location = os.path.join(dir, name)
    cv2.imwrite(save_location, img)

def process_img(img, blur, kernel, iterations):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, blur)

    ret, img = cv2.threshold(img, img.min(), img.max(), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret, img = cv2.threshold(img, 110, 255, cv2.THRESH_TOZERO)

    img = cv2.dilate(img, np.ones((kernel, kernel),np.uint8), iterations = iterations)
    img = cv2.erode(img, np.ones((kernel, kernel),np.uint8), iterations = iterations+1)

    # Choose largest blob
    edges = cv2.Canny(img, 200, 300)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for cnt in contours:
        areas.append(cv2.contourArea(cnt))
    largest_contour_idx = np.argmax(areas)
    img = np.zeros(img.shape)
    cv2.drawContours(img, contours, largest_contour_idx, 255, cv2.FILLED)

    return np.uint8(img)

def overlay_mask_on_source(source_img, mask):
    source_img[(mask==255)] = [0, 0, 255]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(source_img, 1.0, mask, 0.3, 0.0)
    return overlay

def overlay_img_on_backgr(img, mask, backgr, min_scale, blur_kernel, blur_std, p_occl):
    x_lim, y_lim = find_box(mask)
    backgr_width = backgr.shape[1]
    backgr_height = backgr.shape[0]

    # Change brightness of img
    max_brightness_change = 30
    brightness_change = round(np.random.randint(2*max_brightness_change) - 1.5*max_brightness_change)
    img = cv2.add(img, brightness_change)

    # Change hue
    max_hue_change = 3
    hue_change = round(max_hue_change*np.random.randn())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[...,0] = cv2.add(img[...,0], hue_change)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # Apply mask to img
    mask_RGB = np.repeat(mask[...,None], 3, axis=2)
    img = img * np.bool8(mask_RGB)

    # Crop mask & img
    mask = mask[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
    img = img[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]

    # Shrink mask to maximum size given by background image
    mask_width = mask.shape[1]
    mask_height = mask.shape[0]
    if mask_height > backgr_height:
        scale = backgr_height / mask_height
        mask_width = np.floor(mask_width * scale)
        mask_height = np.floor(mask_height * scale)
    if mask_width > backgr_width:
        scale = backgr_width / mask_width
        mask_width = np.floor(mask_width * scale)
        mask_height = np.floor(mask_height * scale)

    # Random shrink
    scale = 1 - (1 - min_scale)*np.random.rand()
    mask_width = round(mask_width * scale)
    mask_height = round(mask_height * scale)

    # Apply shrink
    mask = cv2.resize(mask, (mask_width, mask_height), cv2.INTER_AREA)
    img = cv2.resize(img, (mask_width, mask_height), cv2.INTER_AREA)

    # Introduce occlusions
    # clean_mask = np.copy(mask)
    # band_probability = p_occl
    # max_bandwidth = 0.1 * img.shape[0]
    # if np.random.rand() < band_probability:
    #     bandwidth = round(np.random.rand() * max_bandwidth)
    #     band_location = round(np.random.rand() * (img.shape[0] - bandwidth))
    #     img[band_location:band_location+bandwidth] = 0
    #     mask[band_location:band_location+bandwidth] = 0

    # Choose where to overlay image
    x_shift_range = backgr.shape[1] - mask.shape[1]
    y_shift_range = backgr.shape[0] - mask.shape[0]

    x_shift = round(np.random.rand() * x_shift_range)
    y_shift = round(np.random.rand() * y_shift_range)

    shifted_img = np.zeros((backgr.shape[0], backgr.shape[1], backgr.shape[2]), dtype='u1')
    shifted_mask = np.zeros((backgr.shape[0], backgr.shape[1]), dtype='u1')
    # shifted_clean_mask = np.zeros((backgr.shape[0], backgr.shape[1]), dtype='u1')

    shifted_img[y_shift:y_shift+img.shape[0], x_shift:x_shift+img.shape[1], :] = img
    shifted_mask[y_shift:y_shift+mask.shape[0], x_shift:x_shift+mask.shape[1]] = mask
    # shifted_clean_mask[y_shift:y_shift+clean_mask.shape[0], x_shift:x_shift+clean_mask.shape[1]] = clean_mask

    # Overlay image
    backgr[shifted_mask==255, :] = shifted_img[shifted_mask==255]

    # Add blur
    blurred_backgr = cv2.GaussianBlur(backgr, (blur_kernel, blur_kernel), blur_std)
    blurred_mask = cv2.GaussianBlur(shifted_mask, (blur_kernel, blur_kernel), blur_std)
    blurred_region = (blurred_mask < 255) * (blurred_mask > 0)
    backgr[blurred_region] = blurred_backgr[blurred_region]
     
    return backgr, shifted_mask

def find_box(mask):
    x = np.sum(mask, 0) != 0
    x_cum = np.cumsum(x) * (x!=0)
    y = np.sum(mask, 1) != 0
    y_cum = np.cumsum(y) * (y!=0)

    x_min = np.argmax(x)
    x_max = np.argmax(x_cum)
    y_min = np.argmax(y)
    y_max = np.argmax(y_cum)
    return (x_min, x_max), (y_min, y_max)


if __name__ == "__main__":
    main()
    print('Finished')