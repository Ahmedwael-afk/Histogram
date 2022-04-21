import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


                    
                    #### Histo of image ####

image = cv.imread("Images_and_Videos/FanArtCompetition.jpg")
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
w,h = gray.shape
Histogram = np.zeros(256)
gray_1D = np.reshape(gray,(1,(w*h)))
for i in range (w*h):
	Histogram[gray_1D[0,i]] += 1
Histogram 

#plt.plot(Histogram)
plt.plot(Histogram)


           #### Equalization ####

Histogram_norm = Histogram / (w*h)
Cm = np.zeros(256)
for i in range (256):
	Cm[i] = sum(Histogram_norm[0:i+1])
scale = 255*Cm
New_levels = np.round(scale)
Equalized_Histogram = np.zeros(256)
#Equalized_Histogram[int(New_levels[gray_1D[0,w*h-1]])] 
for i in range(w*h):
	Equalized_Histogram[int(New_levels[gray_1D[0,i]])] += 1 
#plt.plot(Equalized_Histogram)
plt.plot(Equalized_Histogram)


           #### Showing image after equalization ####

final = np.zeros_like(gray)
for i in range (w):
	for j in range (h):
		final[i,j] = New_levels[gray[i,j]]

cv.imshow("final",final)
cv.waitKey(0)










#plt.show()