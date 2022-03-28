import urllib.request
import random

def Img_downloader(url):
	file_name = str(random.randrange(1,1000))+"sora"+".jpg"
	image = urllib.request.urlretrieve(url,file_name)

Img_downloader("https://i.pinimg.com/564x/fe/fa/64/fefa64cb19c33b10b85efc6dd2eaefaf.jpg")