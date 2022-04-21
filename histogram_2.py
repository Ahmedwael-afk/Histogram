import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


image = cv.imread("Images_and_Videos/girl.jpg")
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#print(gray[250,450])
#cv.imshow("gray",gray)
#img = gray.shape
pre_histogram = np.zeros(256)
w,h = gray.shape[:2]
#print(w*h)
image_1D = np.reshape(gray,(1,(w*h)))
#print(image_1D[0,200900])
#print(image_1D.shape)
for i in range (1,w*h):
	pre_histogram[image_1D[0,i]] += 1

plt.plot(pre_histogram)

#print(pre_histogram[:])
normalized_histo = pre_histogram[:]/(w*h)
#print(normalized_histo)
#print(normalized_histo[20],normalized_histo[255])
sum_histo = np.zeros(256)
for i in range (256):
    sum_histo[i] = sum(normalized_histo[0:i+1])

print(sum_histo[255])

new_levels = np.uint8(255*sum_histo)
print(new_levels[255])
new_levels = np.round(new_levels)
plt.plot(new_levels)
plt.show()

print(new_levels[gray[3,100]])

New_image = np.zeros(256)

plt.plot(New_image)
plt.show()




cv.waitKey(0)