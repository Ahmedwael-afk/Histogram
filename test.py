import cv2 as cv
import numpy as np

gray = cv.imdecode(np.fromfile("Images_and_Videos/ghost.png", dtype=np.uint8), 1)
gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
gray = cv.resize(gray, (640, 420))

# Fourier transform
img_dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(img_dft)  # Move frequency domain from upper left to middle
orignal_fourier= np.log(np.abs(dft_shift))
cv.imshow("org",orignal_fourier)
cv.waitKey(0)