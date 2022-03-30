import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread("Images_and_Videos/land.jpg")

blank = np.zeros(img.shape[:2], dtype = 'uint8')

#cv.imshow("blank", blank)

circle = cv.circle(blank.copy(),(400,200),50,(255,255,255),-1)

mask = cv.bitwise_not(circle)
mask_2 = cv.bitwise_xor(blank,circle)
cv.imshow("mask",mask)

mask_3 = cv.bitwise_not(mask_2)
cv.imshow("mask_3",mask_3)

# dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

masked = cv.bitwise_and(img,img,mask = mask_2 )
masked_2 = cv.bitwise_or(img,img, mask = mask_3)
cv.imshow("mask",masked)
cv.imshow("masked_2",masked_2)












cv.waitKey(0)