import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#### Histo of image ####
image = cv.imread("Images_and_Videos/anime.jpg")
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
w,h = gray.shape
Histogram = np.zeros(256)
gray_1D = np.reshape(gray,(1,(w*h)))
for i in range (w*h):
	Histogram[gray_1D[0,i]] += 1 
plt.plot(Histogram)
plt.show()

#### Equalization ####

for i in range (256):
	Cm = sum(Histogram)