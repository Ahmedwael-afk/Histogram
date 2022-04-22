import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        # self.graphWidget_2 = pg.PlotWidget()
        # self.setCentralWidget(self.graphWidget_2)

        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30,32,34,32,33,31,29,32,35,45]

        # plot data: x, y values
        


                    
                    #### Histo of image ####

        image = cv.imread("Images_and_Videos/FanArtCompetition.jpg")
        #cv.imshow("original", image)
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        w,h = gray.shape
        Histogram = np.zeros(256)
        gray_1D = np.reshape(gray,(1,(w*h)))
        for i in range (w*h):
            Histogram[gray_1D[0,i]] += 1

                   #### Plotting
        # plt.plot(Histogram)
        # plt.show()


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
        #plt.plot(Equalized_Histogram)


                   #### Showing image after equalization ####

        # final = np.zeros_like(gray)                     
        # for i in range (w):
        #     for j in range (h):
        #         final[i,j] = New_levels[gray[i,j]]

        

        final = np.zeros_like(gray_1D)             #Can show Equalized image by 2 ways, 1)using 2 nested loops and 2x2 array
        for i in range (w*h):                      #or 2)using 1 loop, 1xsize array then reshape it to image dimensions 
            final[0,i] = New_levels[gray_1D[0,i]]

        final = np.reshape(final,(w,h))
        # cv.imshow("final",final)

        unique, count = np.unique(final, return_counts=True)
        self.graphWidget.plot(unique,count)
        cv.waitKey(0)
        plt.show()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()