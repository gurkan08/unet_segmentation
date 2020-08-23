
"""
import cv2
import os
import numpy as np

save_dir = "dataset/binary_masks"

def read_img(img_dir):

    img_name = os.path.basename(img_dir)
    img = cv2.imread(img_dir)
    h, w, c = img.shape
    blank_image = img

    for _h in range(h):
        for _w in range(w):
            for _c in range(c):
                if img[_h, _w, _c] == 0:
                    blank_image[_h, _w, _c] = 0
                elif img[_h, _w, _c] != 0:
                    blank_image[_h, _w, _c] = 255

    # save binary mask
    cv2.imwrite(os.path.join(save_dir, img_name), np.array(blank_image))


if __name__ == '__main__':

    mask_dir = "dataset/masks"
    for root, dirs, files in os.walk(mask_dir):
       for name in files:
           img_dir = os.path.join(root, name)
           img = read_img(img_dir)

"""

