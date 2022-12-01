#Use this file to fill the imperfect mask. @Hewei

import cv2;
import numpy as np;

# This function map the pixels in [0,255] to [0,#classes-1], in our case, #class=2, i.e. free or not free
def Gray2Classes(img):
    mask=img>0
    img[mask]=1
    return img
    
 
# Read image
im_in = cv2.imread("/home/hewei/TUM/ADLR/dataset/02691156/1a74b169a76e651ebc0909d98a1ff2b4-3.png", cv2.IMREAD_GRAYSCALE)
 
# Threshold.
 
th, im_th = cv2.threshold(im_in, 254, 255, cv2.THRESH_BINARY)
im_th_inv=cv2.bitwise_not(im_th)
 
# # Copy the thresholded image.
im_floodfill = im_th.copy()
 
# # Mask used to flood filling.
# # Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# # Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 0)
# # Combine the two images to get the foreground.
im_out = im_th_inv | im_floodfill

  
# Display images.
cv2.imshow("Thresholded Image", im_th)
cv2.imshow("The Inv of Thresholded Image", im_th_inv)
cv2.imshow("Floodfilled Image", im_floodfill) 

cv2.imshow("Foreground", im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
