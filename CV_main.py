from Gui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import sys, os
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

class AppWindow(QtWidgets.QMainWindow,Ui_MainWindow): #Test    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.actionBrowse.triggered.connect(self.browse)
        self.Image_Combo.activated.connect(self.pick_image_img)
        self.Color_Combo.activated.connect(self.apply_color_space)
        self.Img_to_filters_spatial.clicked.connect(self.Picking_Image_Filters_Spatial)
        self.Img_to_filters_freq.clicked.connect(self.pressed_freq)
        self.Spatial_Combo.activated.connect(self.Picking_Image_Spatial)
        self.Freq_Combo.activated.connect(self.Picking_Image_Freq)
        self.pushButton_Histogram.clicked.connect(self.Histogram_helper)
        self.pushButton_reset.clicked.connect(self.reset)



    def reset(self):
        # for i in range (1,14):
        #     x = "Image_"+str(i)
        #     self.x.clear()
        self.Image_1.clear()
        self.Image_2.clear()
        self.Image_3.clear()
        self.Image_4.clear()
        self.Image_5.clear()
        self.Image_6.clear()
        self.Image_7.clear()
        self.Image_8.clear()
        self.Image_9.clear()
        self.Image_10.clear()
        self.Image_11.clear()
        self.Image_14.clear()
        self.Image_15.clear()
        self.Image_Combo.setCurrentIndex(0)
        self.Color_Combo.setCurrentIndex(0)
        self.Spatial_Combo.setCurrentIndex(0)
        self.Freq_Combo.setCurrentIndex(0)


    def browse(self):                   ##Browse for an image on local files
        self.Image_of_combo_img, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', " ", "JPG Files (*.jpg; *.jpeg);; PNG Fiels (*.png);; BMP Files (*.bmp);; All Files (*)") 
        self.Image_1.setPixmap(QtGui.QPixmap(self.Image_of_combo_img))

    def pick_image_img(self):               ##Combobox of all images
        if self.Image_Combo.currentIndex() == 0:
            self.Image_1.clear()
        else:
            if self.Image_Combo.currentIndex() == 1:
                self.Image_of_combo_img = "Images_and_Videos/mountains.jpg"
            elif self.Image_Combo.currentIndex() == 2:
                self.Image_of_combo_img = "Images_and_Videos/space.jpg"
            elif self.Image_Combo.currentIndex() == 3:
               self.Image_of_combo_img = "Images_and_Videos/land.jpg"
            elif self.Image_Combo.currentIndex() == 4:
                self.Image_of_combo_img = "Images_and_Videos/star.jpg"
            self.Image_1.setPixmap(QtGui.QPixmap(self.Image_of_combo_img))
        

    def apply_color_space(self):           ##Changes color space and sends the pic to image_2
        self.image = cv.imread(self.Image_of_combo_img)
        if self.Color_Combo.currentIndex() == 0:
            cv.imwrite("Cache/rgb.jpg",self.image)
            self.Image_of_combo_color = "Cache/rgb.jpg"
        elif self.Color_Combo.currentIndex() == 1: 
            self.gray = cv.cvtColor(self.image,cv.COLOR_BGR2GRAY)
            cv.imwrite("Cache/gray.jpg",self.gray)
            self.Image_of_combo_color = "Cache/gray.jpg"
        elif self.Color_Combo.currentIndex() == 2:
            self.bgr = cv.cvtColor(self.image,cv.COLOR_BGR2RGB)
            cv.imwrite("Cache/bgr.jpg",self.bgr)
            self.Image_of_combo_color = "Cache/bgr.jpg"
        elif self.Color_Combo.currentIndex() == 3:
            self.LAB = cv.cvtColor(self.image,cv.COLOR_BGR2LAB)
            cv.imwrite("Cache/LAB.jpg",self.LAB)
            self.Image_of_combo_color = "Cache/LAB.jpg"
        elif self.Color_Combo.currentIndex() == 4:
            self.HSV = cv.cvtColor(self.image,cv.COLOR_BGR2HSV)
            cv.imwrite("Cache/HSV.jpg",self.HSV)
            self.Image_of_combo_color = "Cache/HSV.jpg"
        self.Image_2.setPixmap(QtGui.QPixmap(self.Image_of_combo_color))

    def Picking_Image_Filters_Spatial(self):      ##Sends Image after changing color space to Image_3 & Image_4
        self.Image_3.setPixmap(QtGui.QPixmap(self.Image_of_combo_color))



    def Picking_Image_Spatial(self):       ##Apply different filters depending on combobox
        self.image_spatial = cv.imread(self.Image_of_combo_color)
        if self.Spatial_Combo.currentIndex() == 0:
            self.Image_3.clear()
        else:
            if self.Spatial_Combo.currentIndex() == 1:
                self.blur = cv.GaussianBlur(self.image_spatial,(5,5),cv.BORDER_DEFAULT)
                cv.imwrite("Cache/blur.jpg",self.blur)
                self.Image_of_combo_spatial = "Cache/blur.jpg"
            elif self.Spatial_Combo.currentIndex() == 2:
                self.blur_2 = cv.GaussianBlur(self.image_spatial,(3,3),cv.BORDER_DEFAULT)
                self.edges = cv.Canny(self.blur_2,100,150)
                cv.imwrite("Cache/edge.jpg",self.edges)
                self.Image_of_combo_spatial = "Cache/edge.jpg"
            elif self.Spatial_Combo.currentIndex() == 3:
                self.average = cv.blur(self.image_spatial,(5,5))                
                cv.imwrite("Cache/Average.jpg",self.average)
                self.Image_of_combo_spatial = "Cache/Average.jpg"
            elif self.Spatial_Combo.currentIndex() == 4:
                self.kernel2 = np.matrix('-1 -1 -1;-1 8 -1;-1 -1 -1', np.float64)
                self.Laplacian = cv.filter2D(src=self.image_spatial, ddepth=-1, kernel=self.kernel2)
                cv.imwrite("Cache/Laplacian.jpg",self.Laplacian)
                self.Image_of_combo_spatial = "Cache/Laplacian.jpg"
            elif self.Spatial_Combo.currentIndex() == 5:
                self.median = cv.medianBlur(self.image_spatial,5)
                cv.imwrite("Cache/median.jpg",self.median)
                self.Image_of_combo_spatial = "Cache/median.jpg"       
            self.Image_3.setPixmap(QtGui.QPixmap(self.Image_of_combo_spatial))


    def pressed_freq(self):
        self.image_freq = cv.imread(self.Image_of_combo_color,0)
        # self.image_freq = cv.cvtColor(self.image_freq_2, cv.COLOR_RGB2GRAY)
        self.dft = cv.dft(np.float32(self.image_freq), flags=cv.DFT_COMPLEX_OUTPUT)
        self.dft_shift = np.fft.fftshift(self.dft)
        self.magnitude_spectrum = 20 * np.log((cv.magnitude(self.dft_shift[:, :, 0], self.dft_shift[:, :, 1]))+1)    
        self.fig = plt.figure(figsize=(12, 12))
        self.ax1 = self.fig.add_subplot(2,2,1)
        self.ax1.imshow(self.image_freq, cmap = "Greys_r")
        self.ax1.title.set_text('Input Image')
        self.ax2 = self.fig.add_subplot(2,2,2)
        self.ax2.imshow(self.magnitude_spectrum, cmap = "Greys_r")
        self.ax2.title.set_text('FFT of image')
        plt.imsave("Cache/input_image.jpg", self.image_freq, cmap = "Greys_r")
        plt.imsave('Cache/dft.jpg', self.magnitude_spectrum, cmap = "Greys_r" )
        #plt.savefig('Cache/dft.jpg',bbox_inches = 'tight')
        self.image_to_be_filtered_frequency_1 = "Cache/input_image.jpg"
        self.image_to_be_filtered_frequency_2 = "Cache/dft.jpg"
        self.Image_4.setPixmap(QtGui.QPixmap(self.image_to_be_filtered_frequency_1))
        self.Image_5.setPixmap(QtGui.QPixmap(self.image_to_be_filtered_frequency_2))


    def Picking_Image_Freq(self):
        self.image_freq_2 = cv.imread(self.image_to_be_filtered_frequency_2)
        if self.Freq_Combo.currentIndex() == 0:
            self.Image_4.clear()
            self.Image_5.clear()
        elif self.Freq_Combo.currentIndex() == 1:
            rows,cols = self.image_freq_2.shape[:2]
            crow, ccol = int(rows / 2), int(cols / 2)
            mask_1 = np.ones((rows, cols,2), np.uint8)
            r = 80
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask_1[mask_area] = 0
            fshift = self.dft_shift * mask_1
            fshift_mask_mag = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv.idft(f_ishift)
            img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            fig_1 = plt.figure(figsize=(12, 12))
            ax1 = fig_1.add_subplot(4,4,1)
            ax1.imshow(self.image_freq, cmap='gray')
            ax1.title.set_text('Input Image')
            ax2 = fig_1.add_subplot(4,4,2)
            ax2.imshow(self.image_freq_2, cmap='gray')
            ax2.title.set_text('FFT of image')
            plt.savefig("Cache/inverse_dft_1.jpg", bbox_inches = 'tight')
            fig_2 = plt.figure(figsize=(12, 12))
            ax3 = fig_2.add_subplot(4,4,1)
            ax3.imshow(fshift_mask_mag, cmap='gray')
            ax3.title.set_text('FFT + Mask')
            ax4 = fig_2.add_subplot(4,4,2)
            ax4.imshow(img_back, cmap='gray')
            ax4.title.set_text('After inverse FFT')
            plt.savefig("Cache/inverse_dft.jpg",bbox_inches = 'tight')
            self.image_freq_3 = "Cache/inverse_dft_1.jpg"
            self.image_freq_4 = "Cache/inverse_dft.jpg"
            self.Image_4.setPixmap(QtGui.QPixmap(self.image_freq_3))
            self.Image_5.setPixmap(QtGui.QPixmap(self.image_freq_4))

        elif self.Freq_Combo.currentIndex() == 2:
            rows, cols = self.image_freq_2.shape[:2]
            crow, ccol = int(rows / 2), int(cols / 2)
            mask_2 = np.zeros((rows, cols, 2), np.uint8)
            r = 80
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask_2[mask_area] = 1
            fshift = self.dft_shift * mask_2
            fshift_mask_mag = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv.idft(f_ishift)
            img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            fig_1 = plt.figure(figsize=(12, 12))
            ax1 = fig_1.add_subplot(4,4,1)
            ax1.imshow(cv.imread(self.image_to_be_filtered_frequency_1), cmap='gray')
            ax1.title.set_text('Input Image')
            ax2 = fig_1.add_subplot(4,4,2)
            ax2.imshow(cv.imread(self.image_to_be_filtered_frequency_2), cmap='gray')
            ax2.title.set_text('FFT of image')
            plt.savefig("Cache/inverse_dft_3.jpg", bbox_inches = 'tight')
            fig_2 = plt.figure(figsize=(12, 12))
            ax3 = fig_2.add_subplot(4,4,1)
            ax3.imshow(fshift_mask_mag, cmap='gray')
            ax3.title.set_text('FFT + Mask')
            ax4 = fig_2.add_subplot(4,4,2)
            ax4.imshow(img_back, cmap='gray')
            ax4.title.set_text('After inverse FFT')
            plt.savefig("Cache/inverse_dft_4.jpg",bbox_inches = 'tight')
            self.image_freq_5 = "Cache/inverse_dft_3.jpg"
            self.image_freq_6 = "Cache/inverse_dft_4.jpg"
            self.Image_4.setPixmap(QtGui.QPixmap(self.image_freq_5))
            self.Image_5.setPixmap(QtGui.QPixmap(self.image_freq_6))

            


            # img= cv.imread(self.image_to_be_filtered_frequency)
            # blank = np.zeros(img.shape[:2], dtype = 'uint8')
            # circle = cv.circle(blank.copy(),(400,200),50,255,-1)
            # self.mask = cv.bitwise_not(circle)
            # if self.Freq_Combo.currentIndex() == 1:
            #     masked = cv.bitwise_and(img,img, mask = self.mask)
            #     cv.imwrite("Low_pass_freq.jpg", masked)
            #     self.image_of_combo_freq = "Low_pass_freq.jpg"
                #fshift = self.dft_shift * self.mask
                #fshift_mask_mag = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
               # f_ishift = np.fft.ifftshift(masked)
                # img_back = cv.idft(masked)
                # img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
                # ig = plt.figure(figsize=(12, 12))
                #ax1 = fig.add_subplot(2,2,1)
                #ax1.imshow(img, cmap='gray')
                #ax1.title.set_text('Input Image')
                #ax2 = fig.add_subplot(2,2,2)
                #ax2.imshow(magnitude_spectrum, cmap='gray')
                #ax2.title.set_text('FFT of image')
                # ax3 = fig.add_subplot(2,2,3)
                # ax3.imshow(fshift_mask_mag, cmap='gray')
                # ax3.title.set_text('FFT + Mask')
                # ax4 = fig.add_subplot(2,2,4)
                # ax4.imshow(img_back, cmap='gray')
                # ax4.title.set_text('After inverse FFT')
                # plt.savefig("dft_after_filter.jpg", bbox_inches = 'tight')
                # self.image_of_combo_freq = "dft_after_filter.jpg" 

    def Histogram_helper(self):
        img,org_hist, equalized_image = self.Histogram(self.Image_of_combo_img)
        cv.imwrite("Cache/gray_2.jpg",img)
        cv.imwrite("Cache/final.jpg",equalized_image)
        self.Image_7.clear()
        unique_org_hist, count_org_hist = np.unique(img,return_counts=True)
        self.Image_7.plot(unique_org_hist,count_org_hist)                      #### Histogram plotting

        unique_equa_hist, count_equa_hist = np.unique(equalized_image, return_counts=True)
        self.Image_9.clear()
        self.Image_9.plot(unique_equa_hist, count_equa_hist)

        self.Image_6.setPixmap(QtGui.QPixmap("Cache/gray_2.jpg"))    #### Org_image plotting
        self.Image_8.setPixmap(QtGui.QPixmap("Cache/final.jpg"))     #### Equalized_image plotting

        if self.Image_of_combo_color == "Cache/gray.jpg" or "Cache/rgb.jpg":
            filtered_img,filtered_hist,filtered_equa = self.Histogram(self.Image_of_combo_spatial)
            cv.imwrite("Cache/filtered_img.jpg",filtered_img)
            cv.imwrite("Cache/filtered_equalized.jpg",filtered_equa)
            self.Image_11.clear()
            unique_filtered_hist, count_filtered_hist = np.unique(filtered_img,return_counts=True)
            self.Image_11.plot(unique_filtered_hist,count_filtered_hist)                      #### Histogram plotting

            self.Image_15.clear()
            unique_equa_filtered_hist, count_equa_filtered_hist = np.unique(filtered_equa, return_counts=True)
            self.Image_15.plot(unique_equa_filtered_hist, count_equa_filtered_hist)

            self.Image_10.setPixmap(QtGui.QPixmap("Cache/filtered_img.jpg"))           #### Org_image plotting
            self.Image_14.setPixmap(QtGui.QPixmap("Cache/filtered_equalized.jpg"))     #### Equalized_image plotting

        #elif self.Image_of_combo_color == "Cache/bgr.jpg":


    def Histogram(self,Input):
        x = np.shape(cv.imread(Input))
        if x[-1] == 3:
            hsv = cv.cvtColor(cv.imread(Input),cv.COLOR_BGR2HSV)
            h,s,v =cv.split(hsv)
            self.gray = v
        else:
            self.gray = cv.imread(Input)
        w,h = self.gray.shape
        Histogram = np.zeros(256)
        gray_1D = np.reshape(self.gray,(1,(w*h)))
        for i in range (w*h):
            Histogram[gray_1D[0,i]] += 1

        #### Equalized Histogram

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

        return self.gray,Histogram,final




app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
w = AppWindow()

w.show() # Create the widget and show it
sys.exit(app.exec_()) # Run the app