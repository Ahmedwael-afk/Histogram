import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("Images_and_Videos/anime.jpg")
blue,g,r = cv.split(img)
lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)
l,a,b = cv.split(lab)
#cv.imshow("B",r)

hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
h,s,v = cv.split(hsv)
#cv.imshow("v",v)


# hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# hue = hsv_image[:,:,0].astype(float)
# sat = hsv_image[:,:,1].astype(float)
# val = hsv_image[:,:,2].astype(float)
# hue[hue == 0] = np.nan
# histr_lab = cv.calcHist([hsv_image], [0], None, [255], [0, 255])
# histr_lab1 = cv.calcHist([hsv], [1], None, [255], [0, 255])
# histr_lab2 = cv.calcHist([hsv], [2], None, [255], [0, 255])



cv.waitKey(0)



w,h = v.shape
print(v.shape)
Histogram = np.zeros(256)
gray_1D = np.reshape(b,(1,(w*h)))
for i in range (w*h):
    Histogram[gray_1D[0,i]] += 1

#plt.plot(Histogram)

unique, count = np.unique(Histogram,return_counts = True)
plt.bar(unique,count)


Histogram_norm = Histogram / (w*h)
Cm = np.zeros(256)
for i in range (256):
    Cm[i] = sum(Histogram_norm[0:i+1])
scale = 255*Cm
New_levels = np.round(scale)

final = np.zeros_like(gray_1D)             #Can show Equalized image by 2 ways, 1)using 2 nested loops and 2x2 array
for i in range (w*h):                      #or 2)using 1 loop, 1xsize array then reshape it to image dimensions 
    final[0,i] = New_levels[gray_1D[0,i]]

final = np.reshape(final,(w,h))
unique_2, count_2 = np.unique(final,return_counts = True)
plt.bar(unique_2,count_2)


cv.imshow("final",final)

cv.waitKey(0)

plt.show()