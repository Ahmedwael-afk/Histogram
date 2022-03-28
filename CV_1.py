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
        self.Img_to_filters.clicked.connect(self.Picking_Image_Filters_Spatial)
        self.Img_to_filters.clicked.connect(self.Picking_Image_Filters_Freq)
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
        cv.imwrite("to_be_filtered.jpg",self.image)
        self.image_to_be_filtered = "to_be_filtered.jpg"
        return self.image_to_be_filtered

    def Picking_Image_Filters_Freq(self):
        self.Image_4.setPixmap(QtGui.QPixmap(self.pressed_freq()))

    def pressed_freq(self):
        self.image_freq = cv.imread(self.Image_of_combo_color)
        # cv.imwrite("to_freq_filters.jpg",self.image_freq)
        # self.image_to_freq_filters = "to_freq_filters"
        dft = cv.dft(np.float32(self.image_freq), flags=cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log((cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))+1)
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(img)
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(magnitude_spectrum)
        ax2.title.set_text('FFT of image')
        plt.show()



    def Picking_Filter_Spatial(self):      ##Changes Image after applying spatial filter.
        self.Image_3.setPixmap(QtGui.QPixmap(self.Picking_Image_Spatial()))

    def Picking_Image_Spatial(self):       ##Apply different filters depending on combobox
        self.image_spatial = cv.imread(self.image_to_be_filtered)
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



    def Picking_Filter_Freq(self):      ##Changes Image after applying frequency filter.
        self.Image_4.setPixmap(QtGui.QPixmap(self.Picking_Image_Freq()))


    def Picking_Image_Freq(self):
        self.image_freq = self.image_to_be_filtered
        dft = cv.dft(np.float32(self.image_freq), flags=cv.DFT_COMPLEX_OUTPUT)
        if self.Freq_Combo.currentIndex() == 0:
            self.Image_4.clear()
        else:
            if self.Freq_Combo.currentIndex() == 1:
                pass
            return self.image_of_combo_freq


    





app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
w = AppWindow()

w.show() # Create the widget and show it
sys.exit(app.exec_()) # Run the app