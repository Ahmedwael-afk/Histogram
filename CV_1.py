from Gui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import sys

class AppWindow(QtWidgets.QMainWindow,Ui_MainWindow): #Test    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.actionBrowse.triggered.connect(self.browse)
        self.Image_Combo.activated.connect(self.Picking_Image_img)
        self.Color_Combo.activated.connect(self.Picking_Image_color)
        self.Img_to_filters_spatial.clicked.connect(self.Picking_Image_Filters_Spatial)
        self.Img_to_filters_freq.clicked.connect(self.Picking_Image_Filters_Freq)
        self.Spatial_Combo.activated.connect(self.Picking_Filter_Spatial)
        self.Freq_Combo.activated.connect(self.Picking_Filter_Freq)


    def browse(self):                   ##Browse for an image on local files
        self.Image_of_combo_img, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', " ", "JPG Files (*.jpg; *.jpeg);; PNG Fiels (*.png);; BMP Files (*.bmp);; All Files (*)") 
        self.Image_1.setPixmap(QtGui.QPixmap(self.Image_of_combo_img))

    def Picking_Image_img(self):          ##Call Function (pick_image_1) and picks img_1 from combobox
        self.Image_1.setPixmap(QtGui.QPixmap(self.pick_image_img()))

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
             # elif self.Image_Combo.currentIndex() == 5:
             #    self.browse()
            return self.Image_of_combo_img
        
        
    def Picking_Image_color(self):         ##Calls function (apply_color_space)
        self.Image_2.setPixmap(QtGui.QPixmap(self.apply_color_space()))

    def apply_color_space(self):           ##Changes color space and sends the pic to image_2
        self.image = cv.imread(self.Image_of_combo_img)
        if self.Color_Combo.currentIndex() == 0:
            self.Image_2.clear()
        else:
            if self.Color_Combo.currentIndex() == 1: 
                self.gray = cv.cvtColor(self.image,cv.COLOR_BGR2GRAY)
                cv.imwrite("Cache/gray.jpg",self.gray)
                self.Image_of_combo_color = "Cache/gray.jpg"
            elif self.Color_Combo.currentIndex() == 2:
                self.rgb = cv.cvtColor(self.image,cv.COLOR_BGR2RGB)
                cv.imwrite("Cache/rgb.jpg",self.rgb)
                self.Image_of_combo_color = "Cache/rgb.jpg"
            elif self.Color_Combo.currentIndex() == 3:
                self.LAB = cv.cvtColor(self.image,cv.COLOR_BGR2LAB)
                cv.imwrite("Cache/LAB.jpg",self.LAB)
                self.Image_of_combo_color = "Cache/LAB.jpg"
            elif self.Color_Combo.currentIndex() == 4:
                self.HSV = cv.cvtColor(self.image,cv.COLOR_BGR2HSV)
                cv.imwrite("Cache/HSV.jpg",self.HSV)
                self.Image_of_combo_color = "Cache/HSV.jpg"
            return self.Image_of_combo_color

    def Picking_Image_Filters_Spatial(self):      ##Sends Image after changing color space to Image_3 & Image_4
        self.Image_3.setPixmap(QtGui.QPixmap(self.pressed_spatial()))

    def pressed_spatial(self):                   ##Function of pushButton (Done)
        self.image = cv.imread(self.Image_of_combo_color)
        cv.imwrite("Cache/image_to_be_filtered_spatial.jpg",self.image)
        self.image_to_be_filtered_spatial = "Cache/image_to_be_filtered_spatial.jpg"
        return self.image_to_be_filtered_spatial




    def Picking_Filter_Spatial(self):      ##Changes Image after applying spatial filter.
        self.Image_3.setPixmap(QtGui.QPixmap(self.Picking_Image_Spatial()))

    def Picking_Image_Spatial(self):       ##Apply different filters depending on combobox
        self.image_spatial = cv.imread(self.image_to_be_filtered_spatial)
        if self.Spatial_Combo.currentIndex() == 0:
            pass
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
            return self.Image_of_combo_spatial

    def Picking_Image_Filters_Freq(self):
        self.Image_4.setPixmap(QtGui.QPixmap(self.pressed_freq()))

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
        plt.imsave('Cache/dft.jpg', self.magnitude_spectrum, cmap = "Greys_r" )
        #plt.savefig('Cache/dft.jpg',bbox_inches = 'tight')
        self.image_to_be_filtered_frequency = "Cache/dft.jpg"
        return self.image_to_be_filtered_frequency


    def Picking_Filter_Freq(self):      ##Changes Image after applying frequency filter.
        self.Image_4.setPixmap(QtGui.QPixmap(self.Picking_Image_Freq()))


    def Picking_Image_Freq(self):
        self.image_freq_2 = cv.imread(self.image_to_be_filtered_frequency)
        if self.Freq_Combo.currentIndex() == 0:
            self.Image_4.clear()
        else:
            rows, cols = self.image_freq_2.shape[:2]
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.ones((rows, cols,2), np.uint8)
            r = 80
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask[mask_area] = 0

            fshift = self.dft_shift * mask

            fshift_mask_mag = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

            f_ishift = np.fft.ifftshift(fshift)


            img_back = cv.idft(f_ishift)

            img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(self.image_freq_2, cmap='gray')
            ax1.title.set_text('Input Image')
            ax2 = fig.add_subplot(2,2,2)
            ax2.imshow(self.magnitude_spectrum, cmap='gray')
            ax2.title.set_text('FFT of image')
            ax3 = fig.add_subplot(2,2,3)
            ax3.imshow(fshift_mask_mag, cmap='gray')
            ax3.title.set_text('FFT + Mask')
            ax4 = fig.add_subplot(2,2,4)
            ax4.imshow(img_back, cmap='gray')
            ax4.title.set_text('After inverse FFT')
            plt.savefig("Cache/inverse_dft.jpg",bbox_inches = 'tight')
            self.image_freq_2 = "Cache/inverse_dft.jpg"
            return self.image_freq_2


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



              
            return self.image_of_combo_freq


    





app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
w = AppWindow()

w.show() # Create the widget and show it
sys.exit(app.exec_()) # Run the app